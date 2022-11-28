from types import GeneratorType

from affine import Affine
import numpy as np


def array_to_record(
    arr: np.ndarray,
    geotransform: Affine,
    row_off: int = 0,
    col_off: int = 0,
    value_field: str = "band_1",
) -> dict:

    height, width = arr.shape

    lon_NW, lat_NW = geotransform * (col_off, row_off)
    lon_NE, lat_NE = geotransform * (col_off + width, row_off)
    lon_SE, lat_SE = geotransform * (col_off + width, row_off + height)
    lon_SW, lat_SW = geotransform * (col_off, row_off + height)

    # required to append dtype to value field name for storage
    dtype_str = str(arr.dtype)
    value_field = "_".join([value_field, dtype_str])

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
        value_field: arr.tobytes(),
    }

    return record


def import_rasterio():
    try:
        import rasterio  # noqa

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


def rasterio_to_record(file_path: str, band: int = 1) -> dict:
    """Open a raster file with rasterio."""
    rasterio = import_rasterio()

    with rasterio.open(file_path) as raster_dataset:
        return array_to_record(
            raster_dataset.read(band),
            raster_dataset.transform,
            raster_dataset.height,
            raster_dataset.width,
        )


def rasterio_windows_to_records(file_path: str, band: int = 1) -> GeneratorType:
    """Open a raster file with rasterio."""
    rasterio = import_rasterio()

    with rasterio.open(file_path) as raster_dataset:
        for _, window in raster_dataset.block_windows():
            yield array_to_record(
                raster_dataset.read(band, window=window),
                raster_dataset.transform,
                window.row_off,
                window.col_off,
                window.height,
                window.width,
            )


def import_bigquery():
    try:
        from google.cloud import bigquery  # noqa
    except ImportError:

        msg = (
            "Google Cloud BigQuery is not installed.\n"
            "Please install Google Cloud BigQuery to use this function.\n"
            "See https://googleapis.dev/python/bigquery/latest/index.html\n"
            "for installation instructions.\n"
            "OR, run `pip install google-cloud-bigquery` to install from pypi."
        )
        raise ImportError(msg)


def record_to_bigquery(record: dict, table_id: str, dataset_id: str, project_id: str):
    """Write a record to a BigQuery table."""
    bigquery = import_bigquery()

    client = bigquery.Client(project=project_id)

    table_ref = client.dataset(dataset_id).table(table_id)
    table = client.get_table(table_ref)

    errors = client.insert_rows_json(table, [record])
    if errors:
        raise RuntimeError(errors)


'''
def import_redshift():
    try:
        import psycopg2
    except ImportError:

        msg = ("Psycopg2 is not installed.\n"
               "Please install Psycopg2 to use this function.\n"
               "See https://pypi.org/project/psycopg2/\n"
               "for installation instructions.\n"
               "Alternatively, run `pip install psycopg2` to install from pypi.")
        raise ImportError(msg)

def record_to_redshift(record: dict, table_name: str, schema_name: str, conn_str: str):
    """Write a record to a Redshift table.
    """
    import_redshift()

    with psycopg2.connect(conn_str) as conn:
        with conn.cursor() as cur:
            columns = ', '.join(record.keys())
            values = ', '.join([f"'{value}'" for value in record.values()])
            sql = f"INSERT INTO {schema_name}.{table_name} ({columns}) VALUES ({values})"  # noqa
            cur.execute(sql)
            conn.commit()


def import_snowflake():
    try:
        import snowflake.connector
    except ImportError:

        msg = ("Snowflake is not installed.\n"
               "Please install Snowflake to use this function.\n"
               "See https://docs.snowflake.com/en/user-guide/python-connector.html\n"
               "for installation instructions.\n"
               "Alternatively, run `pip install snowflake-connector-python` to install from pypi.")
        raise ImportError(msg)


def record_to_snowflake(record: dict, table_name: str, schema_name: str, conn_str: str):
    """Write a record to a Snowflake table.
    """
    import_snowflake()

    with snowflake.connector.connect(**conn_str) as conn:
        with conn.cursor() as cur:
            columns = ', '.join(record.keys())
            values = ', '.join([f"'{value}'" for value in record.values()])
            sql = f"INSERT INTO {schema_name}.{table_name} ({columns}) VALUES ({values})"
            cur.execute(sql)
            conn.commit()

'''
