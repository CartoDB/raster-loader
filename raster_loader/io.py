import json
import sys
from typing import Iterable

from affine import Affine
import numpy as np
import pandas as pd
import pyproj

from raster_loader.utils import ask_yes_no_question


should_swap = {"=": sys.byteorder == "little", "<": True, ">": False, "|": False}


def array_to_record(
    arr: np.ndarray,
    geotransform: Affine,
    row_off: int = 0,
    col_off: int = 0,
    value_field: str = "band_1",
    crs: str = "EPSG:4326",
    band: int = 1,
) -> dict:

    height, width = arr.shape

    lon_NW, lat_NW = geotransform * (col_off, row_off)
    lon_NE, lat_NE = geotransform * (col_off + width, row_off)
    lon_SE, lat_SE = geotransform * (col_off + width, row_off + height)
    lon_SW, lat_SW = geotransform * (col_off, row_off + height)

    # required to append dtype to value field name for storage
    dtype_str = str(arr.dtype)
    value_field = "_".join([value_field, dtype_str])

    attrs = {
        "band": band,
        "value_field": value_field,
        "dtype": dtype_str,
        "crs": crs,
        "gdal_transform": geotransform.to_gdal(),
        "row_off": row_off,
        "col_off": col_off,
    }

    if should_swap[arr.dtype.byteorder]:
        arr_bytes = np.ascontiguousarray(arr.byteswap()).tobytes()
    else:
        arr_bytes = np.ascontiguousarray(arr).tobytes()

    record = {
        "lat_NW": lat_NW,
        "lon_NW": lon_NW,
        "lat_NE": lat_NE,
        "lon_NE": lon_NE,
        "lat_SE": lat_SE,
        "lon_SE": lon_SE,
        "lat_SW": lat_SW,
        "lon_SW": lon_SW,
        "block_height": height,
        "block_width": width,
        "attrs": json.dumps(attrs),
        value_field: arr_bytes,
    }

    return record


def record_to_array(record: dict, value_field: str = None) -> np.ndarray:
    """Convert a record to a numpy array."""

    if value_field is None:
        value_field = json.loads(record["attrs"])["value_field"]

    # determine dtype
    try:
        dtype_str = value_field.split("_")[-1]
        dtype = np.dtype(dtype_str)
        dtype = dtype.newbyteorder(">")
    except TypeError:
        raise TypeError(f"Invalid dtype: {dtype_str}")

    # determine shape
    shape = (record["block_height"], record["block_width"])

    arr = np.frombuffer(record[value_field], dtype=dtype)
    arr = arr.reshape(shape)

    return arr


def import_rasterio():  # pragma: no cover
    try:
        import rasterio

        return rasterio
    except ImportError:

        msg = (
            "Rasterio is not installed.\n"
            "Please install rasterio to use this function.\n"
            "See https://rasterio.readthedocs.io/en/latest/installation.html\n"
            "for installation instructions.\n"
            "Alternatively, run `pip install rasterio` to install from pypi."
        )
        raise ImportError(msg)


def rasterio_windows_to_records(
    file_path: str, band: int = 1, input_crs: str = None
) -> Iterable:
    """Open a raster file with rasterio."""
    rasterio = import_rasterio()

    with rasterio.open(file_path) as raster_dataset:

        raster_crs = raster_dataset.crs.to_string()

        if input_crs is None:
            input_crs = raster_crs
        elif input_crs != raster_crs:
            print(f"WARNING: Input CRS({input_crs}) != raster CRS({raster_crs}).")

        if not input_crs:  # pragma: no cover
            raise ValueError("Unable to find valid input_crs.")

        for _, window in raster_dataset.block_windows():
            rec = array_to_record(
                raster_dataset.read(band, window=window),
                raster_dataset.transform,
                window.row_off,
                window.col_off,
                crs=input_crs,
                band=band,
            )

            if input_crs.upper() != "EPSG:4326":
                rec = reproject_record(rec, input_crs, "EPSG:4326")

            yield rec


def import_bigquery():  # pragma: no cover
    try:
        from google.cloud import bigquery

        return bigquery
    except ImportError:

        msg = (
            "Google Cloud BigQuery is not installed.\n"
            "Please install Google Cloud BigQuery to use this function.\n"
            "See https://googleapis.dev/python/bigquery/latest/index.html\n"
            "for installation instructions.\n"
            "OR, run `pip install google-cloud-bigquery` to install from pypi."
        )
        raise ImportError(msg)


def records_to_bigquery(
    records: Iterable, table_id: str, dataset_id: str, project_id: str, client=None
):
    """Write a record to a BigQuery table."""

    bigquery = import_bigquery()

    if client is None:  # pragma: no cover
        client = bigquery.Client(project=project_id)

    data_df = pd.DataFrame(records)

    job = client.load_table_from_dataframe(
        data_df, f"{project_id}.{dataset_id}.{table_id}"
    )

    if job:  # pragma: no cover
        job.result()

        if job.errors:
            raise Exception(job.errors)


def bigquery_to_records(
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
    bigquery = import_bigquery()

    client = bigquery.Client(project=project_id)

    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` LIMIT {limit}"

    return client.query(query).result().to_dataframe()


def reproject_record(record: dict, src_crs: str, dst_crs: str = "EPSG:4326") -> dict:
    """Inplace reproject the bounds (lon_NW, lat_NW, etc.) of a record."""

    rasterio = import_rasterio()

    src_crs = rasterio.crs.CRS.from_string(src_crs)
    dst_crs = rasterio.crs.CRS.from_string(dst_crs)

    for lon_col, lat_col in [
        ("lon_NW", "lat_NW"),
        ("lon_NE", "lat_NE"),
        ("lon_SW", "lat_SW"),
        ("lon_SE", "lat_SE"),
    ]:

        transformer = pyproj.Transformer.from_crs(src_crs, dst_crs)
        x, y = transformer.transform(record[lon_col], record[lat_col])
        record[lon_col] = x
        record[lat_col] = y

    return record


def delete_bigquery_table(
    table_id: str,
    dataset_id: str,
    project_id: str,
    client=None,
) -> bool:  # pragma: no cover
    """Delete a BigQuery table.

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
    client : google.cloud.bigquery.client.Client, optional
        BigQuery client, by default None

    Returns
    -------
    bool
        True if the table was deleted.

    """

    bigquery = import_bigquery()

    if client is None:
        client = bigquery.Client(project=project_id)

    table_ref = client.dataset(dataset_id).table(table_id)
    client.delete_table(table_ref, not_found_ok=True)

    return True


def check_if_bigquery_table_exists(
    dataset_id: str,
    table_id: str,
    client,
):  # pragma: no cover
    """Check if a BigQuery table exists.

    Parameters
    ----------
    dataset_id : str
        The BigQuery dataset id.
    table_id : str
        The BigQuery table id.
    client : google.cloud.bigquery.client.Client
        The BigQuery client.

    Returns
    -------
    bool
        True if the table exists, False otherwise.
    """

    table_ref = client.dataset(dataset_id).table(table_id)
    try:
        client.get_table(table_ref)
        return True
    except Exception:
        return False


def check_if_bigquery_table_is_empty(
    dataset_id: str,
    table_id: str,
    client,
):  # pragma: no cover
    """Check if a BigQuery table is empty.

    Parameters
    ----------
    dataset_id : str
        The BigQuery dataset id.
    table_id : str
        The BigQuery table id.
    client : google.cloud.bigquery.client.Client
        The BigQuery client.

    Returns
    -------
    bool
        True if the table is empty, False otherwise.
    """

    table_ref = client.dataset(dataset_id).table(table_id)
    table = client.get_table(table_ref)
    return table.num_rows == 0


def rasterio_to_bigquery(
    file_path: str,
    table_id: str,
    dataset_id: str,
    project_id: str,
    band: int = 1,
    chunk_size: int = None,
    input_crs: int = None,
    client=None,
    overwrite: bool = False,
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
    band : int, optional
        Band number to read from the raster file, by default 1
    chunk_size : int, optional
        Number of records to write to BigQuery at a time, by default None
    input_crs : int, optional
        Input CRS, by default None
    client : [bigquery.Client()], optional
        BigQuery client, by default None
    overwrite : bool, optional
        Overwrite the table if it already contains data, by default False

    Returns
    -------
    bool
        True if upload was successful.
    """

    if isinstance(input_crs, int):
        input_crs = "EPSG:{}".format(input_crs)

    """Write a raster file to a BigQuery table."""
    print("Loading raster file to BigQuery...")

    records_gen = rasterio_windows_to_records(file_path, band, input_crs)

    if client is None:  # pragma: no cover
        client = import_bigquery().Client(project=project_id)

    try:
        if check_if_bigquery_table_exists(dataset_id, table_id, client):
            if overwrite:
                delete_bigquery_table(table_id, dataset_id, project_id, client)

            elif not check_if_bigquery_table_is_empty(dataset_id, table_id, client):
                append_recors = ask_yes_no_question(
                    f"Table {table_id} already exists in dataset {dataset_id} "
                    "and is not empty. Append records? [yes/no] "
                )

                if not append_recors:
                    exit()
        if chunk_size is None:
            records_to_bigquery(
                records_gen, table_id, dataset_id, project_id, client=client
            )
        else:
            from tqdm.auto import tqdm

            total_blocks = get_number_of_blocks(file_path)

            records = []
            with tqdm(total=total_blocks) as pbar:
                for record in records_gen:
                    records.append(record)

                    if len(records) >= chunk_size:
                        records_to_bigquery(
                            records, table_id, dataset_id, project_id, client=client
                        )
                        pbar.update(chunk_size)
                        records = []

                if len(records) > 0:
                    records_to_bigquery(
                        records, table_id, dataset_id, project_id, client=client
                    )
                    pbar.update(len(records))
    except KeyboardInterrupt:
        delete_table = ask_yes_no_question(
            "Would you like to delete the partially uploaded table? [yes/no] "
        )

        if delete_table:
            delete_bigquery_table(table_id, dataset_id, project_id, client)

        raise KeyboardInterrupt

    except Exception as e:
        delete_table = ask_yes_no_question(
            (
                "Error uploading to BigQuery. "
                "Would you like to delete the partially uploaded table? [yes/no] "
            )
        )

        if delete_table:
            delete_bigquery_table(table_id, dataset_id, project_id, client)

        raise IOError("Error uploading to BigQuery: {}".format(e))

    print("Done.")
    return True


def get_number_of_blocks(file_path: str) -> int:
    """Get the number of blocks in a raster file."""
    rasterio = import_rasterio()

    with rasterio.open(file_path) as raster_dataset:
        return len(list(raster_dataset.block_windows()))


def size_mb_of_rasterio_band(file_path: str, band: int = 1) -> int:
    """Get the size in MB of a rasterio band."""
    rasterio = import_rasterio()

    with rasterio.open(file_path) as raster_dataset:
        W = raster_dataset.width
        H = raster_dataset.height
        S = np.dtype(raster_dataset.dtypes[band - 1]).itemsize
        return (W * H * S) / 1024 / 1024


def print_band_information(file_path: str):
    """Print out information about the bands in a raster file."""
    rasterio = import_rasterio()

    with rasterio.open(file_path) as raster_dataset:
        print("Number of bands: {}".format(raster_dataset.count))
        print("Band types: {}".format(raster_dataset.dtypes))
        print(
            "Band sizes (MB): {}".format(
                [
                    size_mb_of_rasterio_band(file_path, band + 1)
                    for band in range(raster_dataset.count)
                ]
            )
        )


def get_block_dims(file_path: str) -> tuple:
    """Get the dimensions of a raster file's blocks."""
    rasterio = import_rasterio()

    with rasterio.open(file_path) as raster_dataset:
        return raster_dataset.block_shapes[0]
