import json
import pandas as pd

from typing import Iterable, List, Tuple

from raster_loader.errors import (
    IncompatibleRasterException,
    import_error_databricks,
)

from raster_loader.utils import ask_yes_no_question, batched

from raster_loader.io.common import (
    rasterio_metadata,
    rasterio_windows_to_records,
    get_number_of_blocks,
    check_metadata_is_compatible,
    update_metadata,
)
from raster_loader.io.datawarehouse import DataWarehouseConnection

try:
    from databricks.connect import DatabricksSession
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        LongType,
    )
except ImportError:  # pragma: no cover
    _has_databricks = False
else:
    _has_databricks = True


class DatabricksConnection(DataWarehouseConnection):
    def __init__(self, host, token, cluster_id):
        if not _has_databricks:
            import_error_databricks()

        self.host = host
        self.token = token
        self.cluster_id = cluster_id

        self.client = self.get_connection()

    def get_connection(self):
        # Initialize DatabricksSession
        session = DatabricksSession.builder.remote(
            host=self.host, token=self.token, cluster_id=self.cluster_id
        ).getOrCreate()
        session.conf.set("spark.databricks.session.timeout", "6h")
        return session

    def execute(self, sql):
        return self.client.sql(sql)

    def execute_to_dataframe(self, sql):
        df = self.execute(sql)
        return df.toPandas()

    def create_schema_if_not_exists(self, fqn):
        schema_name = fqn.split(".")[1]  # Extract schema from FQN
        self.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")

    def create_table_if_not_exists(self, fqn):
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {fqn} (
                BLOCK BIGINT,
                METADATA STRING,
                {self.band_columns}
            ) USING DELTA
            """
        )

    def band_rename_function(self, band_name: str):
        return band_name.upper()

    def write_metadata(
        self,
        metadata,
        append_records,
        fqn,
    ):
        schema = StructType(
            [
                StructField("BLOCK", LongType(), True),
                StructField("METADATA", StringType(), True),
            ]
        )

        data = [(0, json.dumps(metadata))]

        metadata_df = self.client.createDataFrame(data, schema)

        # Write to table
        metadata_df.write.format("delta").mode("append").saveAsTable(fqn)

        return True

    def get_metadata(self, fqn):
        query = f"""
            SELECT METADATA
            FROM {fqn}
            WHERE BLOCK = 0
        """
        result = self.execute_to_dataframe(query)
        if result.empty:
            return None
        return json.loads(result.iloc[0]["METADATA"])

    def check_if_table_exists(self, fqn):
        schema_name, table_name = fqn.split(".")[1:3]  # Extract schema and table
        sql = f"""
            SELECT *
            FROM {schema_name}.INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME = '{table_name}'
        """
        df = self.execute(sql)
        return df.count() > 0

    def check_if_table_is_empty(self, fqn):
        df = self.client.table(fqn)
        return df.count() == 0

    def upload_records(
        self,
        records: Iterable,
        fqn: str,
        overwrite: bool,
    ):
        records_list = []
        for record in records:
            if "METADATA" in record:
                del record["METADATA"]
            records_list.append(record)

        data_df = pd.DataFrame(records_list)
        spark_df = self.client.createDataFrame(data_df)

        mode = "overwrite" if overwrite else "append"
        spark_df.write.format("delta").mode(mode).saveAsTable(fqn)

        return True

    def upload_raster(
        self,
        file_path: str,
        fqn: str,
        bands_info: List[Tuple[int, str]] = None,
        chunk_size: int = None,
        overwrite: bool = False,
        append: bool = False,
        cleanup_on_failure: bool = False,
    ) -> bool:
        print("Loading raster file to Databricks...")

        bands_info = bands_info or [(1, None)]
        append_records = False

        try:
            if (
                self.check_if_table_exists(fqn)
                and not self.check_if_table_is_empty(fqn)
                and not overwrite
            ):
                append_records = append or ask_yes_no_question(
                    f"Table {fqn} already exists and is not empty. "
                    f"Append records? [yes/no] "
                )

                if not append_records:
                    exit()

            # Prepare band columns
            self.band_columns = ", ".join(
                [
                    f"{self.band_rename_function(band_name or f'band_{band}')} BINARY"
                    for band, band_name in bands_info
                ]
            )

            # Create schema and table if not exists
            self.create_schema_if_not_exists(fqn)
            self.create_table_if_not_exists(fqn)

            metadata = rasterio_metadata(
                file_path, bands_info, self.band_rename_function
            )

            records_gen = rasterio_windows_to_records(
                file_path,
                self.band_rename_function,
                bands_info,
            )

            total_blocks = get_number_of_blocks(file_path)

            if chunk_size is None:
                ret = self.upload_records(records_gen, fqn, overwrite)
                if not ret:
                    raise IOError("Error uploading to Databricks.")
            else:
                from tqdm.auto import tqdm

                print(f"Writing {total_blocks} blocks to Databricks...")
                with tqdm(total=total_blocks) as pbar:
                    if total_blocks < chunk_size:
                        chunk_size = total_blocks
                    isFirstBatch = True
                    for records in batched(records_gen, chunk_size):
                        ret = self.upload_records(
                            records, fqn, overwrite and isFirstBatch
                        )
                        pbar.update(len(records))
                        if not ret:
                            raise IOError("Error uploading to Databricks.")
                        isFirstBatch = False

            print("Writing metadata to Databricks...")
            if append_records:
                old_metadata = self.get_metadata(fqn)
                check_metadata_is_compatible(metadata, old_metadata)
                update_metadata(metadata, old_metadata)

            self.write_metadata(metadata, append_records, fqn)

        except IncompatibleRasterException as e:
            raise IOError(f"Error uploading to Databricks: {e.message}")

        except KeyboardInterrupt:
            delete = cleanup_on_failure or ask_yes_no_question(
                "Would you like to delete the partially uploaded table? [yes/no] "
            )

            if delete:
                self.delete_table(fqn)

            raise KeyboardInterrupt

        except Exception as e:
            delete = cleanup_on_failure or ask_yes_no_question(
                (
                    "Error uploading to Databricks. "
                    "Would you like to delete the partially uploaded table? [yes/no] "
                )
            )

            if delete:
                self.delete_table(fqn)

            raise IOError(f"Error uploading to Databricks: {e}")

        print("Done.")
        return True

    def delete_table(self, fqn):
        self.execute(f"DROP TABLE IF EXISTS {fqn}")

    def get_records(self, fqn: str, limit=10) -> pd.DataFrame:
        query = f"SELECT * FROM {fqn} LIMIT {limit}"
        df = self.execute_to_dataframe(query)
        return df

    def insert_in_table(
        self,
        rows: List[dict],
        fqn: str,
    ) -> bool:
        data_df = pd.DataFrame(rows)
        spark_df = self.client.createDataFrame(data_df)
        spark_df.write.format("delta").mode("append").saveAsTable(fqn)
        return True

    def quote_name(self, name):
        return f"`{name}`"
