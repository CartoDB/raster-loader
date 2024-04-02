import time
import json
import pandas as pd
import rasterio
import re

from raster_loader import __version__
from raster_loader.errors import import_error_bigquery, IncompatibleRasterException
from raster_loader.utils import ask_yes_no_question, batched
from raster_loader.io.common import (
    rasterio_metadata,
    rasterio_windows_to_records,
    get_number_of_blocks,
    check_metadata_is_compatible,
    update_metadata,
)

from typing import Iterable, List, Tuple
from functools import partial

try:
    from google.cloud import bigquery
    from google.auth.credentials import Credentials

except ImportError:  # pragma: no cover
    _has_bigquery = False
else:
    _has_bigquery = True

from raster_loader.io.datawarehouse import DataWarehouseConnection

if _has_bigquery:

    class AccessTokenCredentials(Credentials):
        def __init__(self, access_token):
            super(AccessTokenCredentials, self).__init__()
            self._access_token = access_token

        def refresh(self, request):
            pass

        def apply(self, headers, token=None):
            headers["Authorization"] = f"Bearer {self._access_token}"

else:

    class Credentials:
        def __init__(self):
            import_error_bigquery()

    class AccessTokenCredentials:
        def __init__(self, access_token):
            import_error_bigquery()


class BigQueryConnection(DataWarehouseConnection):
    def __init__(self, project, credentials: Credentials = None):
        if not _has_bigquery:  # pragma: no cover
            import_error_bigquery()

        self.client = bigquery.Client(project=project, credentials=credentials)

    def execute(self, sql):
        return self.client.query(sql).result()

    def execute_to_dataframe(self, sql):
        return self.client.query(sql).to_dataframe()

    def quote(self, q):
        if isinstance(q, str):
            q = q.replace("\\", "\\\\")
            return f"'''{q}'''"
        return str(q)

    def quote_name(self, name):
        return f"`{name}`"

    def upload_records(self, records: Iterable, fqn):
        records = list(records)

        data_df = pd.DataFrame(records)

        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("block", bigquery.enums.SqlTypeNames.INT64),
                bigquery.SchemaField("metadata", bigquery.enums.SqlTypeNames.STRING),
            ],
            clustering_fields=["block"],
        )

        return self.client.load_table_from_dataframe(
            dataframe=data_df,
            destination=fqn,
            job_id_prefix=f"{fqn.split('.')[-1]}_",
            job_config=job_config,
        )

    def upload_raster(
        self,
        file_path: str,
        fqn: str,
        bands_info: List[Tuple[int, str]] = [(1, None)],
        chunk_size: int = None,
        overwrite: bool = False,
        append: bool = False,
        cleanup_on_failure: bool = False,
    ):
        """Write a raster file to a BigQuery table."""
        print("Loading raster file to BigQuery...")

        append_records = False

        try:
            if self.check_if_table_exists(fqn) and not self.check_if_table_is_empty(
                fqn
            ):
                if overwrite:
                    self.delete_bigquery_table(fqn)
                else:
                    append_records = append or ask_yes_no_question(
                        f"Table {fqn} already exists "
                        "and is not empty. Append records? [yes/no] "
                    )

                    if not append_records:
                        exit()

            metadata = rasterio_metadata(
                file_path, bands_info, self.band_rename_function
            )

            records_gen = rasterio_windows_to_records(
                file_path,
                self.band_rename_function,
                bands_info,
            )

            if append_records:
                old_metadata = self.get_metadata(fqn)
                check_metadata_is_compatible(metadata, old_metadata)
                update_metadata(metadata, old_metadata)

            total_blocks = get_number_of_blocks(file_path)
            if chunk_size is None:
                job = self.upload_records(records_gen, fqn)
                # raise error if job went wrong (blocking call)
                job.result()
            else:
                from tqdm.auto import tqdm

                jobs = []
                errors = []
                print(f"Writing {total_blocks} blocks to BigQuery...")
                with tqdm(total=total_blocks) as pbar:
                    if total_blocks < chunk_size:
                        chunk_size = total_blocks

                    def done_callback(job):
                        pbar.update(job.num_records or 0)
                        try:
                            job.result()
                        except Exception as e:
                            errors.append(e)
                        try:
                            jobs.remove(job)
                        except ValueError:
                            # job already removed because failed
                            pass

                    for records in batched(records_gen, chunk_size):
                        job = self.upload_records(records, fqn)
                        job.num_records = len(records)

                        job.add_done_callback(partial(lambda job: done_callback(job)))
                        jobs.append(job)

                        # do not continue to schedule jobs if there are errors
                        if len(errors):
                            raise Exception(errors)

                    # wait for end of jobs or any error
                    while len(jobs) > 0 and len(errors) == 0:
                        time.sleep(1)

                    if len(errors):
                        raise Exception(errors)

                    pbar.update(1)

            print("Writing metadata to BigQuery...")
            self.write_metadata(metadata, append_records, fqn)

            print("Updating labels...")
            self.update_labels(fqn, self.get_labels(__version__))

        except IncompatibleRasterException as e:
            raise IOError("Error uploading to BigQuery: {}".format(e.message))

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
                    "Error uploading to BigQuery. "
                    "Would you like to delete the partially uploaded table? [yes/no] "
                )
            )

            if delete:
                self.delete_table(fqn)

            raise IOError("Error uploading to BigQuery: {}".format(e))

        print("Done.")
        return True

    def delete_bigquery_table(self, fqn: str):
        try:
            self.client.delete_table(fqn, not_found_ok=True)
            return True
        except Exception:
            return False

    def check_if_table_exists(self, fqn: str):
        try:
            self.client.get_table(fqn)
            return True
        except Exception:
            return False

    def check_if_table_is_empty(self, fqn: str):
        table = self.client.get_table(fqn)
        return table.num_rows == 0

    def get_metadata(self, fqn):
        rows = self.execute(
            f"""
                SELECT metadata FROM {self.quote_name(fqn)} WHERE block = 0
            """
        )

        rows = list(rows)
        if len(rows) == 0:
            return None

        return json.loads(rows[0]["metadata"])

    def get_labels(self, version: str):
        return {
            "raster_loader": re.sub(r"[^a-z0-9_-]", "_", version.lower()),
        }

    def update_labels(self, fqn, labels):
        table = self.client.get_table(fqn)
        table.labels = labels
        table = self.client.update_table(table, ["labels"])

    def write_metadata(
        self,
        metadata,
        append_records,
        fqn,
    ):
        if append_records:
            self.execute(
                f"""
                UPDATE {self.quote_name(fqn)}
                SET metadata = (
                    SELECT TO_JSON_STRING(
                        PARSE_JSON(
                            {self.quote(json.dumps(metadata))},
                            wide_number_mode=>'round'
                        )
                    )
                ) WHERE block = 0
            """
            )

            return True
        else:
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
