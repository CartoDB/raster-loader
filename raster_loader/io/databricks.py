import json
import pandas as pd
import rasterio

from itertools import chain
from raster_loader.lib.errors import (
    import_error_databricks,
    IncompatibleRasterException,
)
from raster_loader.lib.utils import ask_yes_no_question, batched
from raster_loader.io.common import (
    check_metadata_is_compatible,
    get_number_of_blocks,
    get_number_of_overviews_blocks,
    rasterio_metadata,
    rasterio_overview_to_records,
    rasterio_windows_to_records,
    update_metadata,
)

from typing import Dict, Iterable, List, Tuple

try:
    from databricks.connect import DatabricksSession
except ImportError:  # pragma: no cover
    _has_databricks = False
else:
    _has_databricks = True

from raster_loader.io.datawarehouse import DataWarehouseConnection


class DatabricksConnection(DataWarehouseConnection):
    def __init__(self, server_hostname, access_token, cluster_id, parallelism=200):
        # Validate required parameters
        if not server_hostname:
            raise ValueError("server_hostname cannot be null or empty")
        if not access_token:
            raise ValueError("access_token cannot be null or empty")
        if not cluster_id:
            raise ValueError("cluster_id cannot be null or empty")
        if not _has_databricks:
            import_error_databricks()

        # Normalize server_hostname by removing any 'https://' prefix
        self.server_hostname = server_hostname.replace("https://", "")
        self.access_token = access_token
        self.cluster_id = cluster_id
        self.parallelism = parallelism

        try:
            self.spark = DatabricksSession.builder.remote(
                host=f"https://{self.server_hostname}",
                token=access_token,
                cluster_id=cluster_id,
            ).getOrCreate()
        except Exception as e:
            print(f"Error initializing Spark session: {e}")
            raise

    def execute(self, query: str) -> list:
        """Execute a SQL query and return results as a list of rows"""
        try:
            return self.spark.sql(query).collect()
        except Exception as e:
            print(f"Error executing query: {e}")
            raise

    def execute_to_dataframe(self, query: str) -> "pd.DataFrame":
        """Execute a SQL query and return results as a pandas DataFrame"""
        try:
            return self.spark.sql(query).toPandas()
        except Exception as e:
            print(f"Error executing query: {e}")
            raise

    def quote(self, q):
        if isinstance(q, str):
            q = q.replace("'", "''")
            return f"'{q}'"
        return str(q)

    def quote_name(self, name):
        """Quote a table name with proper escaping for Databricks."""
        parts = name.replace("`", "").split(".")
        return ".".join(f"`{part}`" for part in parts)

    def upload_records(
        self,
        records: Iterable,
        fqn: str,
        overwrite: bool = False,
        parallelism: int = 1000,
    ):
        # Convert to Pandas DataFrame
        data_df = pd.DataFrame(records)

        # Drop metadata column if it exists
        if "metadata" in data_df.columns:
            data_df = data_df.drop(columns=["metadata"])

        if data_df.empty:
            print("No records to upload.")
            return True

        try:
            # Convert Pandas DataFrame to Spark DataFrame
            spark_df = self.spark.createDataFrame(data_df)

            # Write to Delta table
            write_mode = "overwrite" if overwrite else "append"
            (
                spark_df.repartition(parallelism)
                .write.format("delta")
                .mode(write_mode)
                .saveAsTable(fqn)
            )

            return True
        except Exception as e:
            print(f"Error uploading records: {str(e)}")
            return False

    def wait_for_cluster(self):
        """Wait for the Databricks cluster to be ready."""
        print("Waiting for Databricks cluster to be ready...")
        try:
            # Execute a simple SQL query that doesn't affect any tables
            self.execute("SELECT 1")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Databricks cluster: {str(e)}")

    def upload_raster(
        self,
        file_path: str,
        fqn: str,
        bands_info: List[Tuple[int, str]] = None,
        chunk_size: int = None,
        parallelism: int = 1000,
        overwrite: bool = False,
        append: bool = False,
        cleanup_on_failure: bool = False,
        exact_stats: bool = False,
        basic_stats: bool = False,
        compress: bool = False,
        compression_level: int = 6,
        band_valuelabels: List[Dict[int, str]] = [],
    ):
        """Write a raster file to a Databricks table."""
        # Wait for cluster to be ready before starting the upload
        self.wait_for_cluster()

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
                band_valuelabels=band_valuelabels,
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
                success = self.upload_records(records_gen, fqn, overwrite, parallelism)
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
                            records, fqn, overwrite and isFirstBatch, parallelism
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
