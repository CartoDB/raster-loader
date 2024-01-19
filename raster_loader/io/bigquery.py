import json
import time
import rasterio
import pandas as pd

from typing import Iterable, List, Tuple
from functools import partial

from raster_loader.errors import import_error_bigquery, IncompatibleRasterException

from raster_loader.utils import ask_yes_no_question, batched, bigquery_sql_quote

from raster_loader.io.common import (
    rasterio_metadata,
    rasterio_windows_to_records,
    get_number_of_blocks,
    check_metadata_is_compatible,
    update_metadata,
)

try:
    from google.cloud import bigquery
except ImportError:  # pragma: no cover
    _has_bigquery = False
else:
    _has_bigquery = True


def table_columns(bands: List[str]) -> List[Tuple[str, str, str]]:
    # TODO: upgrade BQ client version and use 'JSON' type for 'attrs'
    columns = [("block", "INTEGER", "REQUIRED"), ("metadata", "STRING", "NULLABLE")]
    columns += [(band_name, "BYTES", "NULLABLE") for band_name in bands]
    return columns


def write_metadata(
    metadata,
    append_records,
    project_id,
    dataset_id,
    table_id,
    client=None,
):
    """Write the metadata of a raster file to a BigQuery table.

    Parameters
    ----------
    metadata : dict
        The metadata of the raster file.
    append_records : bool
        Whether the table already contains records.
    project_id : str
        BigQuery project name.
    dataset_id : str
        BigQuery dataset name.
    table_id : str
        BigQuery table name.
    client : google.cloud.bigquery.client.Client
        The BigQuery client.

    Returns
    -------
    bool
        True if the metadata was written.
    """
    """Requires bigquery."""
    if not _has_bigquery:  # pragma: no cover
        import_error_bigquery()

    if client is None:
        client = bigquery.Client(project=project_id)
    if append_records:
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        query = f"""
            UPDATE `{table_ref}`
            SET metadata = (
                SELECT TO_JSON_STRING(
                    PARSE_JSON(
                        {bigquery_sql_quote(json.dumps(metadata))},
                        wide_number_mode=>'round'
                    )
                )
            ) WHERE block = 0
        """

        job = client.query(query)
        job.result()

        return True
    else:
        return insert_in_table(
            [
                {
                    "block": 0,  # store metadata in the record with this block number
                    "metadata": json.dumps(metadata),
                }
            ],
            project_id,
            dataset_id,
            table_id,
            client=client,
        )


def get_metadata(project_id, dataset_id, table_id, client=None):
    """Get the metadata of the raster contained in a BigQuery table.

    Parameters
    ----------
    project_id : str
        BigQuery project name.
    dataset_id : str
        BigQuery dataset name.
    table_id : str
        BigQuery table name.
    client : google.cloud.bigquery.client.Client
        The BigQuery client.

    Returns
    -------
    dict
        The metadata of the table.
    """

    """Requires bigquery."""
    if not _has_bigquery:  # pragma: no cover
        import_error_bigquery()

    if client is None:
        client = bigquery.Client(project=project_id)
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    query = f"""
        SELECT metadata
        FROM `{table_ref}`
        WHERE block = 0
    """
    job = client.query(query)
    result = job.result()
    return json.loads(next(result)["metadata"])


def records_to_bigquery(
    records: Iterable, table_id: str, dataset_id: str, project_id: str, client=None
):
    """Write a record to a BigQuery table."""

    """Requires bigquery."""
    if not _has_bigquery:  # pragma: no cover
        import_error_bigquery()

    if client is None:  # pragma: no cover
        client = bigquery.Client(project=project_id)

    records = list(records)

    data_df = pd.DataFrame(records)

    return client.load_table_from_dataframe(
        dataframe=data_df,
        destination=f"{project_id}.{dataset_id}.{table_id}",
        job_id_prefix=f"{table_id}_",
    )


def table_to_records(
    table_id: str, dataset_id: str, project_id: str, limit=10
) -> pd.DataFrame:  # pragma: no cover
    """Read a BigQuery table into a records pandas.DataFrame.

    Requires the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable set to the path
    of a JSON file containing your BigQuery credentials (see `the GCP documentation
    <https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key>`_
    for more information).

    Parameters
    ----------
    table_id : str
        BigQuery table name.
    dataset_id : str
        BigQuery dataset name.
    project_id : str
        BigQuery project name.
    limit : int, optional
        Max number of records to return, by default 10.

    Returns
    -------
    pandas.DataFrame
        Records as a pandas.DataFrame.

    """

    """Requires bigquery."""
    if not _has_bigquery:  # pragma: no cover
        import_error_bigquery()

    client = bigquery.Client(project=project_id)

    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` LIMIT {limit}"

    return client.query(query).result().to_dataframe()


def create_table(
    project_id: str,
    dataset_id: str,
    table_id: str,
    columns: List[Tuple[str, str, str]],
    clustering: List[str],
    client=None,
) -> bool:  # pragma: no cover
    """Requires bigquery."""

    if not _has_bigquery:  # pragma: no cover
        import_error_bigquery()

    if client is None:
        client = bigquery.Client(project=project_id)

    schema = [
        bigquery.SchemaField(column_name, column_type, mode=column_mode)
        for [column_name, column_type, column_mode] in columns
    ]

    table = bigquery.Table(f"{project_id}.{dataset_id}.{table_id}", schema=schema)
    table.clustering_fields = clustering
    client.create_table(table)

    return True


def insert_in_table(
    rows: List[dict],
    project_id: str,
    dataset_id: str,
    table_id: str,
    client=None,
) -> bool:
    """Insert rows in a BigQuery table.

    Requires the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable set to the path
    of a JSON file containing your BigQuery credentials (see `the GCP documentation
    <https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key>`_
    for more information).

    Parameters
    ----------
    rows : List[dict],
        Rows to be inserted.
    project_id : str
        BigQuery project name.
    dataset_id : str
        BigQuery dataset name.
    table_id : str
        BigQuery table name.
    client : google.cloud.bigquery.client.Client, optional
        BigQuery client, by default None

    Returns
    -------
    bool
        True if the rows were inserted

    """

    """Requires bigquery."""
    if not _has_bigquery:  # pragma: no cover
        import_error_bigquery()

    if client is None:
        client = bigquery.Client(project=project_id)

    # Note that client.insert_rows_json(f"{project_id}.{dataset_id}.{table_id}", rows)
    # is prone to race conditions when a table with the same name
    # has been recently deleted (as we do for overwrite)
    # see https://github.com/googleapis/python-bigquery/issues/1396
    # So we'll run an INSERT query instead.

    columns = rows[0].keys()
    values = ",".join(
        [
            "("
            + ",".join([bigquery_sql_quote(row[column]) for column in columns])
            + ")"
            for row in rows
        ]
    )
    job = client.query(
        f"""
        INSERT INTO `{project_id}.{dataset_id}.{table_id}`({','.join(columns)})
        VALUES {values}
        """
    )
    job.result()

    return True


def delete_table(
    project_id: str,
    dataset_id: str,
    table_id: str,
    client=None,
) -> bool:  # pragma: no cover
    """Delete a BigQuery table.

    Parameters
    ----------
    project_id : str
        BigQuery project name.
    dataset_id : str
        BigQuery dataset name.
    table_id : str
        BigQuery table name.
    client : google.cloud.bigquery.client.Client, optional
        BigQuery client, by default None

    Returns
    -------
    bool
        True if the table was deleted.

    """

    """Requires bigquery."""
    if not _has_bigquery:  # pragma: no cover
        import_error_bigquery()

    if client is None:
        client = bigquery.Client(project=project_id)

    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    client.delete_table(table_ref, not_found_ok=True)

    return True


def check_if_table_exists(
    project_id: str,
    dataset_id: str,
    table_id: str,
    client=None,
):  # pragma: no cover
    """Check if a BigQuery table exists.

    Parameters
    ----------
    project_id : str
        BigQuery project name.
    dataset_id : str
        BigQuery dataset name.
    table_id : str
        BigQuery table name.
    client : google.cloud.bigquery.client.Client
        The BigQuery client.

    Returns
    -------
    bool
        True if the table exists, False otherwise.
    """

    """Requires bigquery."""
    if not _has_bigquery:  # pragma: no cover
        import_error_bigquery()

    if client is None:
        client = bigquery.Client(project=project_id)

    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    try:
        client.get_table(table_ref)
        return True
    except Exception:
        return False


def check_if_table_is_empty(
    project_id: str,
    dataset_id: str,
    table_id: str,
    client,
):  # pragma: no cover
    """Check if a BigQuery table is empty.

    Parameters
    ----------
    project_id : str
        BigQuery project name.
    dataset_id : str
        BigQuery dataset name.
    table_id : str
        BigQuery table name.
    client : google.cloud.bigquery.client.Client
        The BigQuery client.

    Returns
    -------
    bool
        True if the table is empty, False otherwise.
    """

    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    table = client.get_table(table_ref)
    return table.num_rows == 0


def rasterio_to_table(
    file_path: str,
    table_id: str,
    dataset_id: str,
    project_id: str,
    bands_info: List[Tuple[int, str]] = [(1, None)],
    chunk_size: int = None,
    client=None,
    overwrite: bool = False,
    append: bool = False,
) -> bool:
    """Write a rasterio-compatible raster file to a BigQuery table.
    Compatible file formats include TIFF and GeoTIFF. See
    `the GDAL website <https://gdal.org/drivers/raster/index.html>`_ for a full list.

    Requires the ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable set to the path
    of a JSON file containing your BigQuery credentials (see `the GCP documentation
    <https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key>`_
    for more information).

    Parameters
    ----------
    file_path : str
        Path to the rasterio-compatible raster file.
    table_id : str
        BigQuery table name.
    dataset_id : str
        BigQuery dataset name.
    project_id : str
        BigQuery project name.
    bands_info : [(int, str)], optional
        Band number(s) and column name(s) to read from the
        raster file, by default [(1, None)].
        If column_name is None default column name is band_<band_num>.
    chunk_size : int, optional
        Number of records to write to BigQuery at a time, the default (None)
        writes all records in a single batch.
    client : [bigquery.Client()], optional
        BigQuery client, by default None
    overwrite : bool, optional
        Overwrite the table if it already contains data, by default False

    Returns
    -------
    bool
        True if upload was successful.
    """

    """Requires bigquery."""
    if not _has_bigquery:  # pragma: no cover
        import_error_bigquery()

    """Write a raster file to a BigQuery table."""
    print("Loading raster file to BigQuery...")

    if client is None:  # pragma: no cover
        client = bigquery.Client(project=project_id)

    append_records = False

    try:
        create = False
        if check_if_table_exists(project_id, dataset_id, table_id, client):
            if overwrite:
                delete_table(project_id, dataset_id, table_id, client)
                create = True

            elif not check_if_table_is_empty(project_id, dataset_id, table_id, client):
                append_records = append or ask_yes_no_question(
                    f"Table {table_id} already exists in dataset {dataset_id} "
                    "and is not empty. Append records? [yes/no] "
                )

                if not append_records:
                    exit()
        else:
            create = True

        def table_creator(columns, clustering):
            if create:
                create_table(
                    project_id, dataset_id, table_id, columns, clustering, client
                )

        metadata = rasterio_metadata(file_path, bands_info, lambda x: x)

        records_gen = rasterio_windows_to_records(
            file_path,
            table_creator,
            table_columns,
            lambda x: x,
            bands_info,
        )

        if append_records:
            old_metadata = get_metadata(project_id, dataset_id, table_id, client)
            check_metadata_is_compatible(metadata, old_metadata)
            update_metadata(metadata, old_metadata)

        total_blocks = get_number_of_blocks(file_path)
        if chunk_size is None:
            job = records_to_bigquery(
                records_gen, table_id, dataset_id, project_id, client=client
            )
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
                    job = records_to_bigquery(
                        records, table_id, dataset_id, project_id, client=client
                    )
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
        write_metadata(
            metadata,
            append_records,
            project_id,
            dataset_id,
            table_id,
            client=client,
        )

    except IncompatibleRasterException as e:
        raise IOError("Error uploading to BigQuery: {}".format(e.message))

    except KeyboardInterrupt:
        delete = ask_yes_no_question(
            "Would you like to delete the partially uploaded table? [yes/no] "
        )

        if delete:
            delete_table(project_id, dataset_id, table_id, client)

        raise KeyboardInterrupt

    except rasterio.errors.CRSError as e:
        raise e

    except Exception as e:
        delete = ask_yes_no_question(
            (
                "Error uploading to BigQuery. "
                "Would you like to delete the partially uploaded table? [yes/no] "
            )
        )

        if delete:
            delete_table(project_id, dataset_id, table_id, client)

        raise IOError("Error uploading to BigQuery: {}".format(e))

    print("Done.")
    return True
