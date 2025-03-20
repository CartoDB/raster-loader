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
    def __init__(self, host, token, cluster_id, catalog, schema):
        if not _has_databricks:
            import_error_databricks()

        self.host = host
        self.token = token
        self.cluster_id = cluster_id
        self.catalog = catalog
        self.schema = schema

        self.client = self.get_connection()

    def get_connection(self):
        # Initialize DatabricksSession
        session = DatabricksSession.builder.remote(
            host=self.host, token=self.token, cluster_id=self.cluster_id
        ).getOrCreate()
        session.conf.set("spark.databricks.session.timeout", "6h")
        return session

    def get_table_fqn(self, table):
        return f"`{self.catalog}`.{self.schema}.{table}"

    def execute(self, sql):
        return self.client.sql(sql)

    def execute_to_dataframe(self, sql):
        df = self.execute(sql)
        return df.toPandas()

    def create_schema_if_not_exists(self):
        self.execute(f"CREATE SCHEMA IF NOT EXISTS `{self.catalog}`.{self.schema}")

    def create_table_if_not_exists(self, table):
        self.execute(
            f"""
            CREATE TABLE IF NOT EXISTS `{self.catalog}`.{self.schema}.{table} (
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
        table,
    ):
        # Create a DataFrame with the metadata
        schema = StructType(
            [
                StructField("BLOCK", LongType(), True),
                StructField("METADATA", StringType(), True),
            ]
        )

        data = [(0, json.dumps(metadata))]

        metadata_df = self.client.createDataFrame(data, schema)

        # Write to table
        fqn = self.get_table_fqn(table)
        metadata_df.write.format("delta").mode("append").saveAsTable(fqn)

        return True

    def get_metadata(self, table):
        fqn = self.get_table_fqn(table)
        query = f"""
            SELECT METADATA
            FROM {fqn}
            WHERE BLOCK = 0
        """
        result = self.execute_to_dataframe(query)
        if result.empty:
            return None
        return json.loads(result.iloc[0]["METADATA"])

    def check_if_table_exists(self, table):
        sql = f"""
            SELECT *
            FROM `{self.catalog}`.INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = '{self.schema}'
              AND TABLE_NAME = '{table}'
        """
        df = self.execute(sql)
        # If the count is greater than 0, the table exists
        return df.count() > 0

    def check_if_table_is_empty(self, table):
        fqn = self.get_table_fqn(table)
        df = self.client.table(fqn)
        return df.count() == 0

    def upload_records(
        self,
        records: Iterable,
        table: str,
        overwrite: bool,
    ):
        fqn = self.get_table_fqn(table)
        records_list = []
        for record in records:
            # Remove 'METADATA' from records, as it's handled separately
            if "METADATA" in record:
                del record["METADATA"]
            records_list.append(record)

        data_df = pd.DataFrame(records_list)
        spark_df = self.client.createDataFrame(data_df)

        if overwrite:
            mode = "overwrite"
        else:
            mode = "append"

        spark_df.write.format("delta").mode(mode).saveAsTable(fqn)

        return True

    def upload_raster(
        self,
        file_path: str,
        table: str,
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
                self.check_if_table_exists(table)
                and not self.check_if_table_is_empty(table)
                and not overwrite
            ):
                append_records = append or ask_yes_no_question(
                    f"Table `{self.catalog}`.{self.schema}.{table} already exists "
                    "and is not empty. Append records? [yes/no] "
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
            self.create_schema_if_not_exists()
            self.create_table_if_not_exists(table)

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
                ret = self.upload_records(records_gen, table, overwrite)
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
                            records, table, overwrite and isFirstBatch
                        )
                        pbar.update(len(records))
                        if not ret:
                            raise IOError("Error uploading to Databricks.")
                        isFirstBatch = False

            print("Writing metadata to Databricks...")
            if append_records:
                old_metadata = self.get_metadata(table)
                check_metadata_is_compatible(metadata, old_metadata)
                update_metadata(metadata, old_metadata)

            self.write_metadata(metadata, append_records, table)

        except IncompatibleRasterException as e:
            raise IOError(f"Error uploading to Databricks: {e.message}")

        except KeyboardInterrupt:
            delete = cleanup_on_failure or ask_yes_no_question(
                "Would you like to delete the partially uploaded table? [yes/no] "
            )

            if delete:
                self.delete_table(table)

            raise KeyboardInterrupt

        except Exception as e:
            delete = cleanup_on_failure or ask_yes_no_question(
                (
                    "Error uploading to Databricks. "
                    "Would you like to delete the partially uploaded table? [yes/no] "
                )
            )

            if delete:
                self.delete_table(table)

            raise IOError(f"Error uploading to Databricks: {e}")

        print("Done.")
        return True

    def delete_table(self, table):
        fqn = self.get_table_fqn(table)
        self.execute(f"DROP TABLE IF EXISTS {fqn}")

    def get_records(self, table: str, limit=10) -> pd.DataFrame:
        fqn = self.get_table_fqn(table)
        query = f"SELECT * FROM {fqn} LIMIT {limit}"
        df = self.execute_to_dataframe(query)
        return df

    def insert_in_table(
        self,
        rows: List[dict],
        table: str,
    ) -> bool:
        fqn = self.get_table_fqn(table)
        data_df = pd.DataFrame(rows)
        spark_df = self.client.createDataFrame(data_df)
        spark_df.write.format("delta").mode("append").saveAsTable(fqn)
        return True

    def quote_name(self, name):
        return f"`{name}`"
