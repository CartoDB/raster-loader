import json
import rasterio
import pandas as pd

from typing import Iterable, List, Tuple

from raster_loader.errors import (
    IncompatibleRasterException,
    import_error_snowflake,
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
    from snowflake.connector.pandas_tools import write_pandas
    import snowflake.connector
except ImportError:  # pragma: no cover
    _has_snowflake = False
else:
    _has_snowflake = True


class SnowflakeConnection(DataWarehouseConnection):
    def __init__(self, username, password, account, database, schema, token, role):
        if not _has_snowflake:
            import_error_snowflake()

        # TODO: Write a proper static factory for this
        if token is None:
            self.client = snowflake.connector.connect(
                user=username,
                password=password,
                account=account,
                database=database.upper(),
                schema=schema.upper(),
                role=role.upper() if role is not None else None,
            )
        else:
            self.client = snowflake.connector.connect(
                authenticator="oauth",
                token=token,
                account=account,
                database=database.upper(),
                schema=schema.upper(),
                role=role.upper() if role is not None else None,
            )

    def band_rename_function(self, band_name: str):
        return band_name

    def write_metadata(
        self,
        metadata,
        append_records,
        fqn,
    ):
        fqn = fqn.upper()
        if append_records:
            query = f"""
                UPDATE {fqn}
                SET metadata = (
                    SELECT TO_JSON(
                        PARSE_JSON(
                            {self.quote(json.dumps(metadata))}
                        )
                    )
                ) WHERE block = 0
            """

            self.execute(query)

            return True
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
                        "BLOCK": 0,
                        "METADATA": json.dumps(metadata),
                    }
                ],
                fqn,
            )

    def get_metadata(self, fqn: str):
        query = f"""
            SELECT metadata
            FROM {fqn.upper()}
            WHERE block = 0
        """
        result = self.execute(query)
        if len(result) == 0:
            return None
        return json.loads(result[0][0])

    def upload_records(
        self,
        records: Iterable,
        fqn: str,
        overwrite: bool,
    ):
        records_list = []
        for record in records:
            del record["METADATA"]
            records_list.append(record)

        data_df = pd.DataFrame(records_list)

        database, schema, table = fqn.upper().split(".")

        return write_pandas(
            conn=self.client,
            df=data_df,
            table_name=table,
            database=database,
            schema=schema,
            chunk_size=10000,
            auto_create_table=True,
            overwrite=overwrite,
        )[0]

    def execute(self, sql):
        return self.client.cursor().execute(sql).fetchall()

    def execute_to_dataframe(self, sql):
        return self.client.cursor().execute(sql).fetch_pandas_all()

    def check_if_table_exists(self, fqn: str):  # pragma: no cover
        database, schema, table = fqn.upper().split(".")
        query = f"""
            SELECT *
            FROM {database}.INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = '{schema}'
            AND TABLE_NAME = '{table}';
            """
        res = self.execute(query)

        return len(res) > 0

    def check_if_table_is_empty(
        self,
        fqn: str,
    ):  # pragma: no cover
        database, schema, table = fqn.split(".")
        query = f"""
            SELECT ROW_COUNT
            FROM {database}.INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = '{schema.upper()}'
            AND TABLE_NAME = '{table.upper()}';
            """
        res = self.execute(query)
        return res[0] == 0

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
        print("Loading raster file to Snowflake...")

        bands_info = bands_info or [(1, None)]

        append_records = False

        fqn = fqn.upper()

        try:
            if (
                self.check_if_table_exists(fqn)
                and not self.check_if_table_is_empty(fqn)
                and not overwrite
            ):
                append_records = append or ask_yes_no_question(
                    f"Table {fqn} already exists "
                    "and is not empty. Append records? [yes/no] "
                )

                if not append_records:
                    exit()

            metadata = rasterio_metadata(file_path, bands_info, lambda x: x.upper())

            records_gen = rasterio_windows_to_records(
                file_path,
                lambda x: x.upper(),
                bands_info,
            )

            total_blocks = get_number_of_blocks(file_path)

            if chunk_size is None:
                ret = self.upload_records(records_gen, fqn, overwrite)
                if not ret:
                    raise IOError("Error uploading to Snowflake.")
            else:
                from tqdm.auto import tqdm

                print(f"Writing {total_blocks} blocks to Snowflake...")
                with tqdm(total=total_blocks) as pbar:
                    if total_blocks < chunk_size:
                        chunk_size = total_blocks
                    isFirstBatch = True
                    for records in batched(records_gen, chunk_size):
                        ret = self.upload_records(
                            records, fqn, overwrite and isFirstBatch
                        )
                        pbar.update(chunk_size)
                        if not ret:
                            raise IOError("Error uploading to Snowflake.")
                        isFirstBatch = False
                    pbar.update(1)

            print("Writing metadata to Snowflake...")
            if append_records:
                old_metadata = self.get_metadata(fqn)
                check_metadata_is_compatible(metadata, old_metadata)
                update_metadata(metadata, old_metadata)

            self.write_metadata(metadata, append_records, fqn)

        except IncompatibleRasterException as e:
            raise IOError("Error uploading to Snowflake: {}".format(e.message))

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
                (
                    "Error uploading to Snowflake. "
                    "Would you like to delete the partially uploaded table? [yes/no] "
                )
            )

            if delete:
                self.delete_table(fqn)

            raise IOError("Error uploading to Snowflake: {}".format(e))

        print("Done.")
        return True
