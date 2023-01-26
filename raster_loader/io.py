import json
import sys
from itertools import islice
from typing import Iterable

from affine import Affine
import numpy as np
import pandas as pd
import pyproj

try:
    import rio_cogeo
except ImportError:  # pragma: no cover
    _has_rio_cogeo = False
else:
    _has_rio_cogeo = True

try:
    import rasterio
except ImportError:  # pragma: no cover
    _has_rasterio = False
else:
    _has_rasterio = True

try:
    import quadbin
except ImportError:  # pragma: no cover
    _has_quadbin = False
else:
    _has_quadbin = True

try:
    from google.cloud import bigquery
except ImportError:  # pragma: no cover
    _has_bigquery = False
else:
    _has_bigquery = True

from raster_loader.utils import ask_yes_no_question

should_swap = {"=": sys.byteorder != "little", "<": False, ">": True, "|": False}


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:  # pragma: no cover
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):  # noqa
        yield batch


def array_to_record(
    arr: np.ndarray,
    transformer: pyproj.Transformer,
    geotransform: Affine,
    row_off: int = 0,
    col_off: int = 0,
    value_field: str = "band_1",
    crs: str = "EPSG:4326",
    band: int = 1,
) -> dict:
    height, width = arr.shape

    lon_NW, lat_NW = transformer.transform(*(geotransform * (col_off, row_off)))
    lon_NE, lat_NE = transformer.transform(*(geotransform * (col_off + width, row_off)))
    lon_SE, lat_SE = transformer.transform(
        *(geotransform * (col_off + width, row_off + height))
    )
    lon_SW, lat_SW = transformer.transform(
        *(geotransform * (col_off, row_off + height))
    )

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


def array_to_quadbin_record(
    arr: np.ndarray,
    transformer: pyproj.Transformer,
    geotransform: Affine,
    resolution: int,
    row_off: int = 0,
    col_off: int = 0,
    value_field: str = "band_1",
    crs: str = "EPSG:4326",
    band: int = 1,
) -> dict:
    """Requires quadbin."""
    if not _has_quadbin:  # pragma: no cover
        import_error_quadbin()

    height, width = arr.shape

    x, y = transformer.transform(
        *(geotransform * (col_off + width * 0.5, row_off + height * 0.5))
    )

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
        "quadbin": quadbin.point_to_cell(x, y, resolution),
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
        dtype = dtype.newbyteorder("<")
    except TypeError:
        raise TypeError(f"Invalid dtype: {dtype_str}")

    # determine shape
    shape = (record["block_height"], record["block_width"])

    arr = np.frombuffer(record[value_field], dtype=dtype)
    arr = arr.reshape(shape)

    return arr


def import_error_bigquery():  # pragma: no cover
    msg = (
        "Google Cloud BigQuery is not installed.\n"
        "Please install Google Cloud BigQuery to use this function.\n"
        "See https://googleapis.dev/python/bigquery/latest/index.html\n"
        "for installation instructions.\n"
        "OR, run `pip install google-cloud-bigquery` to install from pypi."
    )
    raise ImportError(msg)


def import_error_rasterio():  # pragma: no cover
    msg = (
        "Rasterio is not installed.\n"
        "Please install rasterio to use this function.\n"
        "See https://rasterio.readthedocs.io/en/latest/installation.html\n"
        "for installation instructions.\n"
        "Alternatively, run `pip install rasterio` to install from pypi."
    )
    raise ImportError(msg)


def import_error_rio_cogeo():  # pragma: no cover
    msg = (
        "Cloud Optimized GeoTIFF (COG) plugin for Rasterio is not installed.\n"
        "Please install rio-cogeo to use this function.\n"
        "See https://cogeotiff.github.io/rio-cogeo/\n"
        "for installation instructions.\n"
        "Alternatively, run `pip install rio-cogeo` to install from pypi."
    )
    raise ImportError(msg)


def import_error_quadbin():  # pragma: no cover
    msg = (
        "Quadbin is not installed.\n"
        "Please install quadbin to use this function.\n"
        "See https://github.com/CartoDB/quadbin-py\n"
        "for installation instructions.\n"
        "Alternatively, run `pip install quadbin` to install from pypi."
    )
    raise ImportError(msg)


def rasterio_windows_to_records(
    file_path: str, band: int = 1, input_crs: str = None, output_quadbin: bool = False
) -> Iterable:
    if output_quadbin:
        """Open a raster file with rio-cogeo."""
        raster_info = rio_cogeo.cog_info(file_path).dict()

        """Check if raster is quadbin compatible."""
        if "GoogleMapsCompatible" != raster_info.get("Tags", {}).get(
            "Tiling Scheme", {}
        ).get("NAME"):
            msg = (
                "To use the output_quadbin option, "
                "the input raster must be a GoogleMapsCompatible raster.\n"
                "You can make your raster compatible "
                "by converting it using the following command:\n"
                "gdalwarp your_raster.tif -of COG "
                "-co TILING_SCHEME=GoogleMapsCompatible -co COMPRESS=DEFLATE "
                "your_compatible_raster.tif"
            )
            raise ValueError(msg)

        resolution = raster_info["GEO"]["MaxZoom"]

    """Requires rasterio."""
    if not _has_rasterio:  # pragma: no cover
        import_error_rasterio()

    """Open a raster file with rasterio."""
    with rasterio.open(file_path) as raster_dataset:

        raster_crs = raster_dataset.crs.to_string()

        if input_crs is None:
            input_crs = raster_crs
        elif input_crs != raster_crs:
            msg = "Input CRS conflicts with input raster metadata."
            err = rasterio.errors.CRSError(msg)
            raise err

        if not input_crs:  # pragma: no cover
            msg = "Unable to find valid input_crs."
            err = rasterio.errors.CRSError(msg)
            raise err

        transformer = pyproj.Transformer.from_crs(
            input_crs, "EPSG:4326", always_xy=True
        )

        for _, window in raster_dataset.block_windows():

            if output_quadbin:
                rec = array_to_quadbin_record(
                    raster_dataset.read(band, window=window),
                    transformer,
                    raster_dataset.transform,
                    resolution,
                    window.row_off,
                    window.col_off,
                    crs=input_crs,
                    band=band,
                )

            else:
                rec = array_to_record(
                    raster_dataset.read(band, window=window),
                    transformer,
                    raster_dataset.transform,
                    window.row_off,
                    window.col_off,
                    crs=input_crs,
                    band=band,
                )

            yield rec


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

    job_config = bigquery.LoadJobConfig()

    if "quadbin" in data_df.keys():
        # Cluster table by quadbin
        job_config.clustering_fields = ["quadbin"]

    return client.load_table_from_dataframe(
        data_df, f"{project_id}.{dataset_id}.{table_id}", job_config=job_config
    )


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

    """Requires bigquery."""
    if not _has_bigquery:  # pragma: no cover
        import_error_bigquery()

    client = bigquery.Client(project=project_id)

    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}` LIMIT {limit}"

    return client.query(query).result().to_dataframe()


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

    """Requires bigquery."""
    if not _has_bigquery:  # pragma: no cover
        import_error_bigquery()

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
    output_quadbin: bool = False,
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
    output_quadbin : bool, optional
        Upload the raster to the BigQuery table in a quadbin format (input raster must
        be a GoogleMapsCompatible raster)

    Returns
    -------
    bool
        True if upload was successful.
    """

    """Requires bigquery."""
    if not _has_bigquery:  # pragma: no cover
        import_error_bigquery()

    if isinstance(input_crs, int):
        input_crs = "EPSG:{}".format(input_crs)

    """Write a raster file to a BigQuery table."""
    print("Loading raster file to BigQuery...")

    records_gen = rasterio_windows_to_records(
        file_path, band, input_crs, output_quadbin
    )

    if client is None:  # pragma: no cover
        client = bigquery.Client(project=project_id)

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
            job = records_to_bigquery(
                records_gen, table_id, dataset_id, project_id, client=client
            )
            # raise error if job went wrong (blocking call)
            job.result()
        else:
            from tqdm.auto import tqdm

            total_blocks = get_number_of_blocks(file_path)

            jobs = []
            with tqdm(total=total_blocks) as pbar:
                for records in batched(records_gen, chunk_size):

                    try:
                        # raise error if job went wrong (blocking call)
                        jobs.pop().result()
                    except IndexError:
                        pass

                    jobs.append(
                        records_to_bigquery(
                            records, table_id, dataset_id, project_id, client=client
                        )
                    )
                    pbar.update(chunk_size)

            # raise error if the last job went wrong (blocking call)
            jobs.pop().result()

    except KeyboardInterrupt:
        delete_table = ask_yes_no_question(
            "Would you like to delete the partially uploaded table? [yes/no] "
        )

        if delete_table:
            delete_bigquery_table(table_id, dataset_id, project_id, client)

        raise KeyboardInterrupt

    except rasterio.errors.CRSError as e:
        raise e

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

    """Requires rasterio."""
    if not _has_rasterio:  # pragma: no cover
        import_error_rasterio()

    with rasterio.open(file_path) as raster_dataset:
        return len(list(raster_dataset.block_windows()))


def size_mb_of_rasterio_band(file_path: str, band: int = 1) -> int:
    """Get the size in MB of a rasterio band."""

    """Requires rasterio."""
    if not _has_rasterio:  # pragma: no cover
        import_error_rasterio()

    with rasterio.open(file_path) as raster_dataset:
        W = raster_dataset.width
        H = raster_dataset.height
        S = np.dtype(raster_dataset.dtypes[band - 1]).itemsize
        return (W * H * S) / 1024 / 1024


def print_band_information(file_path: str):
    """Print out information about the bands in a raster file."""

    """Requires rasterio."""
    if not _has_rasterio:  # pragma: no cover
        import_error_rasterio()

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

    """Requires rasterio."""
    if not _has_rasterio:  # pragma: no cover
        import_error_rasterio()

    with rasterio.open(file_path) as raster_dataset:
        return raster_dataset.block_shapes[0]
