import json

from typing import Iterable

from affine import Affine
import numpy as np
import pandas as pd

import pyproj


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
        value_field: arr.tobytes(),  # add in endian flag?
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
    except ValueError:
        raise ValueError(f"Invalid dtype: {dtype_str}")

    # determine shape
    shape = (record["block_height"], record["block_width"])

    arr = np.frombuffer(record[value_field], dtype=dtype)
    arr = arr.reshape(shape)

    return arr


def import_rasterio():
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


def _record_to_file(
    record: dict,
    file_path: str,
    value_field: str = "band_1_int8",
    driver: str = "GTiff",
) -> None:
    """Write a record to a file."""

    rasterio = import_rasterio()

    arr = record_to_array(record, value_field)

    h, w = arr.shape[-2:]

    xrange = (record["lon_NW"], record["lon_NE"])
    yrange = (record["lat_SW"], record["lat_NW"])

    transform = rasterio.transform.from_bounds(*xrange, *yrange, w, h)

    with rasterio.open(
        file_path,
        "w",
        driver=driver,
        height=h,
        width=w,
        count=1,
        dtype=arr.dtype,
        transform=transform,
    ) as dst:
        dst.write(arr, 1)
        print("Wrote to {}".format(file_path))


def bigquery_to_file(
    table_id: str,
    dataset_id: str,
    project_id: str,
    file_path: str,
    driver: str = "GTiff",
):
    """Write BigQuery Raster to File"""
    import os

    path, ext = os.path.splitext(file_path)

    df = bigquery_to_records(table_id, dataset_id, project_id)

    for i, record in df.iterrows():
        out_path = path + "_part" + str(i) + ext
        _record_to_file(record, out_path, driver=driver)


def rasterio_to_record(file_path: str, band: int = 1, input_crs: str = None) -> dict:
    """Open a raster file with rasterio."""
    rasterio = import_rasterio()

    with rasterio.open(file_path) as raster_dataset:

        if input_crs is None:
            input_crs = raster_dataset.crs.to_string()

        return array_to_record(
            raster_dataset.read(band),
            raster_dataset.transform,
            crs=input_crs,
            band=band,
        )


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

        if not input_crs:
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


def import_bigquery():
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

    # TODO: Need to test it and see if the load_table style is better..
    bigquery = import_bigquery()

    if client is None:
        client = bigquery.Client(project=project_id)

    data_df = pd.DataFrame(records)

    client.load_table_from_dataframe(data_df, f"{project_id}.{dataset_id}.{table_id}")


def bigquery_to_records(
    table_id: str, dataset_id: str, project_id: str, limit=10
) -> pd.DataFrame:
    """Read a BigQuery table into a records pandas.DataFrame."""
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

        x, y = pyproj.transform(src_crs, dst_crs, record[lon_col], record[lat_col])
        record[lon_col] = x
        record[lat_col] = y

    return record


def rasterio_to_bigquery(
    file_path: str,
    table_id: str,
    dataset_id: str,
    project_id: str,
    band: int = 1,
    chunk_size: int = None,
    input_crs: int = None,
    client=None,
) -> bool:
    """Write a rasterio-compatible file to a BigQuery table.

    Parameters
    ----------
    file_path : str
        Path to the raster file.
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

    Returns
    -------
    bool
        True if successful.

    Notes
    -----
    - TODO: If BigQuery has error then user should be notified
    - TODO: Make generally more error resilient
    """

    if isinstance(input_crs, int):
        input_crs = "EPSG:{}".format(input_crs)

    """Write a raster file to a BigQuery table."""
    print("Loading raster file to BigQuery...")

    records_gen = rasterio_windows_to_records(file_path, band, input_crs)

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

    print("Done.")
    return True


def get_number_of_blocks(file_path: str) -> int:
    """Get the number of blocks in a raster file."""
    rasterio = import_rasterio()

    with rasterio.open(file_path) as raster_dataset:
        return len(list(raster_dataset.block_windows()))


def print_gdalinfo(file_path: str):
    """Print out the output of gdalinfo."""
    import subprocess

    print("Running gdalinfo...")
    subprocess.run(["gdalinfo", file_path])


def size_mb_of_each_block(file_path: str) -> int:
    """Get the size in MB of each block in a raster file."""
    rasterio = import_rasterio()

    with rasterio.open(file_path) as raster_dataset:
        height = raster_dataset.block_shapes[0][0]
        width = raster_dataset.block_shapes[0][1]
        size = np.dtype(raster_dataset.dtypes[0]).itemsize
        return (height * width * size) / 1024 / 1024


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
