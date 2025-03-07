import json
import pandas as pd
import rasterio
import os

from itertools import chain
from raster_loader.errors import import_error_databricks, IncompatibleRasterException
from raster_loader.utils import ask_yes_no_question, batched
from raster_loader.io.common import (
    check_metadata_is_compatible,
    get_number_of_blocks,
    get_number_of_overviews_blocks,
    rasterio_metadata,
    rasterio_overview_to_records,
    rasterio_windows_to_records,
    update_metadata,
)

from typing import Iterable, List, Tuple

try:
    from databricks import sql as databricks_sql
    from databricks.connect import DatabricksSession
except ImportError:  # pragma: no cover
    _has_databricks = False
else:
    _has_databricks = True

from raster_loader.io.datawarehouse import DataWarehouseConnection


class DatabricksConnection(DataWarehouseConnection):
    def __init__(self, server_hostname, http_path, access_token, cluster_id=None):
        if not _has_databricks:
            import_error_databricks()

        self.server_hostname = server_hostname
        self.http_path = http_path
        self.access_token = access_token
        self.connection = None
        self.cursor = None

        # Initialize Spark session with cluster_id
        try:
            self.spark = DatabricksSession.builder.remote(
                host=f"https://{server_hostname}",
                token=access_token,
                cluster_id=cluster_id
                or os.getenv(
                    "DATABRICKS_CLUSTER_ID"
                ),  # Try environment variable if not provided
            ).getOrCreate()
        except Exception as e:
            print(f"Error initializing Spark session: {e}")
            raise

    def _ensure_connection(self):
        if self.connection is None:
            self.connection = databricks_sql.connect(
                server_hostname=self.server_hostname,
                http_path=self.http_path,
                access_token=self.access_token,
            )
            self.cursor = self.connection.cursor()

    def execute(self, sql_query):
        self._ensure_connection()
        self.cursor.execute(sql_query)
        return self.cursor.fetchall()

    def execute_to_dataframe(self, sql_query):
        self._ensure_connection()
        self.cursor.execute(sql_query)
        return pd.DataFrame(self.cursor.fetchall())

    def quote(self, q):
        if isinstance(q, str):
            q = q.replace("'", "''")
            return f"'{q}'"
        return str(q)

    def quote_name(self, name):
        """Quote a table name with proper escaping for Databricks."""
        parts = name.replace("`", "").split(".")
        return ".".join(f"`{part}`" for part in parts)

    def upload_records(self, records: Iterable, fqn: str, overwrite: bool = False):
        records_list = []
        for record in records:
            if "metadata" in record:
                del record["metadata"]
            records_list.append(record)

        if not records_list:
            print("No records to upload.")
            return True

        try:
            # Convert to Pandas DataFrame
            data_df = pd.DataFrame(records_list)

            # Convert Pandas DataFrame to Spark DataFrame
            spark_df = self.spark.createDataFrame(data_df)

            # Write to Delta table with corrected settings
            write_mode = "overwrite" if overwrite else "append"
            (
                spark_df.repartition(200)
                .write.format("delta")
                .mode(write_mode)
                .saveAsTable(fqn)
            )

            return True
        except Exception as e:
            print(f"Error uploading records: {str(e)}")
            return False

    def upload_raster(
        self,
        file_path: str,
        fqn: str,
        bands_info: List[Tuple[int, str]] = None,
        chunk_size: int = None,
        overwrite: bool = False,
        append: bool = False,
        cleanup_on_failure: bool = False,
        exact_stats: bool = False,
        basic_stats: bool = False,
        compress: bool = False,
        compression_level: int = 6,
    ):
        """Write a raster file to a Databricks table."""
        print("Loading raster file to Databricks...")

        bands_info = bands_info or [(1, None)]
        append_records = False
        fqn = self.quote_name(fqn)

        try:
            if self.check_if_table_exists(fqn) and not self.check_if_table_is_empty(
                fqn
            ):
                if overwrite:
                    self.delete_table(fqn)
                else:
                    append_records = append or ask_yes_no_question(
                        f"Table {fqn} already exists "
                        "and is not empty. Append records? [yes/no] "
                    )

                    if not append_records:
                        exit()

            metadata = rasterio_metadata(
                file_path,
                bands_info,
                self.band_rename_function,
                exact_stats,
                basic_stats,
                compress=compress,
            )

            overviews_records_gen = rasterio_overview_to_records(
                file_path,
                self.band_rename_function,
                bands_info,
                compress=compress,
                compression_level=compression_level,
            )

            windows_records_gen = rasterio_windows_to_records(
                file_path,
                self.band_rename_function,
                bands_info,
                compress=compress,
                compression_level=compression_level,
            )

            records_gen = chain(overviews_records_gen, windows_records_gen)

            if append_records:
                old_metadata = self.get_metadata(fqn)
                check_metadata_is_compatible(metadata, old_metadata)
                update_metadata(metadata, old_metadata)

            number_of_blocks = get_number_of_blocks(file_path)
            number_of_overview_tiles = get_number_of_overviews_blocks(file_path)
            total_blocks = number_of_blocks + number_of_overview_tiles

            if chunk_size is None:
                success = self.upload_records(records_gen, fqn, overwrite)
                if not success:
                    raise IOError("Error uploading to Databricks.")
            else:
                from tqdm.auto import tqdm

                processed_blocks = 0
                print(
                    f"Writing {number_of_blocks} blocks and {number_of_overview_tiles} "
                    "overview tiles to Databricks..."
                )
                with tqdm(total=total_blocks) as pbar:
                    if total_blocks < chunk_size:
                        chunk_size = total_blocks
                    isFirstBatch = True

                    for records in batched(records_gen, chunk_size):
                        ret = self.upload_records(
                            records, fqn, overwrite and isFirstBatch
                        )

                        num_records = len(records)
                        processed_blocks += num_records
                        pbar.update(num_records)
                        if not ret:
                            raise IOError("Error uploading to Databricks.")
                        isFirstBatch = False

                    empty_blocks = total_blocks - processed_blocks
                    pbar.update(empty_blocks)

                print("Number of empty blocks: ", empty_blocks)

            print("Writing metadata to Databricks...")
            self.write_metadata(metadata, append_records, fqn)

        except IncompatibleRasterException as e:
            raise IOError("Error uploading to Databricks: {}".format(e.message))

        except KeyboardInterrupt:
            delete = cleanup_on_failure or ask_yes_no_question(
                "Would you like to delete the partially uploaded table? [yes/no] "
            )

            if delete:
                self.delete_table(fqn)

            raise KeyboardInterrupt

        except rasterio.errors.CRSError as e:
            raise e

        except Exception as e:
            delete = cleanup_on_failure or ask_yes_no_question(
                "Error uploading to Databricks. "
                "Would you like to delete the partially uploaded table? [yes/no] "
            )

            if delete:
                self.delete_table(fqn)

            import traceback

            print(traceback.print_exc())
            raise IOError("Error uploading to Databricks: {}".format(e))
        finally:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()

        print("Done.")
        return True

    def check_if_table_exists(self, fqn: str):
        try:
            self.execute(f"DESCRIBE TABLE {fqn}")
            return True
        except Exception:
            return False

    def check_if_table_is_empty(self, fqn: str):
        try:
            result = self.execute(f"SELECT COUNT(*) FROM {fqn}")
            return result[0][0] == 0
        except Exception:
            return True

    def get_metadata(self, fqn):
        rows = self.execute(
            f"""
            SELECT metadata FROM {fqn} WHERE block = 0
            """
        )
        if not rows:
            return None
        return json.loads(rows[0][0])

    def write_metadata(self, metadata, append_records, fqn):
        if append_records:
            self.execute(
                f"""
                UPDATE {fqn}
                SET metadata = {self.quote(json.dumps(metadata))}
                WHERE block = 0
            """
            )
        else:
            self.execute(
                f"""
                    ALTER TABLE {fqn}
                    ADD COLUMN metadata STRING;
                """
            )
            return self.insert_in_table(
                [
                    {
                        # store metadata in the record with this block number
                        "block": 0,
                        "metadata": json.dumps(metadata),
                    }
                ],
                fqn,
            )
