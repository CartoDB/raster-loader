import time
from functools import partial
import json
import sys
from typing import Iterable
from typing import Callable
from typing import List
from typing import Tuple

from affine import Affine
import numpy as np
import pandas as pd
import pyproj
import shapely

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

from raster_loader.utils import ask_yes_no_question, batched
from raster_loader.geo import (
    raster_bounds,
)

should_swap = {"=": sys.byteorder != "little", "<": False, ">": True, "|": False}


def band_field_name(custom_name: str, band: int) -> str:
    return custom_name or "band_" + str(band)


def array_to_record(
    arr: np.ndarray,
    value_field: str,
    transformer: pyproj.Transformer,
    geotransform: Affine,
    resolution: int,
    row_off: int = 0,
    col_off: int = 0,
) -> dict:
    """Requires quadbin."""
    if not _has_quadbin:  # pragma: no cover
        import_error_quadbin()

    height, width = arr.shape

    x, y = transformer.transform(
        *(geotransform * (col_off + width * 0.5, row_off + height * 0.5))
    )

    block = quadbin.point_to_cell(x, y, resolution)

    if should_swap[arr.dtype.byteorder]:
        arr_bytes = np.ascontiguousarray(arr.byteswap()).tobytes()
    else:
        arr_bytes = np.ascontiguousarray(arr).tobytes()

    record = {
        "block": block,
        "metadata": None,
        value_field: arr_bytes,
    }

    return record


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


def import_error_quadbin():  # pragma: no cover
    msg = (
        "Quadbin is not installed.\n"
        "Please install quadbin to use this function.\n"
        "See https://github.com/CartoDB/quadbin-py\n"
        "for installation instructions.\n"
        "Alternatively, run `pip install quadbin` to install from pypi."
    )
    raise ImportError(msg)


def raster_band_type(raster_dataset: rasterio.io.DatasetReader, band: int) -> str:
    types = {
        i: dtype for i, dtype in zip(raster_dataset.indexes, raster_dataset.dtypes)
    }
    return str(types[band])


def table_columns(bands: List[str]) -> List[Tuple[str, str, str]]:
    # TODO: upgrade BQ client version and use 'JSON' type for 'attrs'
    columns = [("block", "INTEGER", "REQUIRED"), ("metadata", "STRING", "NULLABLE")]
    columns += [(band_name, "BYTES", "NULLABLE") for band_name in bands]
    return columns


def rasterio_metadata(
    file_path: str,
    bands_info: List[Tuple[int, str]],
):
    """Requires rasterio."""
    if not _has_rasterio:  # pragma: no cover
        import_error_rasterio()

    """Open a raster file with rasterio."""
    raster_info = rio_cogeo.cog_info(file_path).dict()

    """Check if raster is compatible."""
    if "GoogleMapsCompatible" != raster_info.get("Tags", {}).get(
        "Tiling Scheme", {}
    ).get("NAME"):
        msg = (
            "The input raster must be a GoogleMapsCompatible raster.\n"
            "You can make your raster compatible "
            "by converting it using the following command:\n"
            "gdalwarp -of COG -co TILING_SCHEME=GoogleMapsCompatible "
            "-co COMPRESS=DEFLATE -co OVERVIEWS=NONE -co ADD_ALPHA=NO "
            "-co RESAMPLING=NEAREST <input_raster>.tif <output_raster>.tif"
        )
        raise ValueError(msg)

    metadata = {}
    resolution = raster_info["GEO"]["MaxZoom"]
    metadata["resolution"] = resolution
    metadata["minresolution"] = resolution
    metadata["maxresolution"] = resolution
    metadata["nodata"] = raster_info["Profile"]["Nodata"]

    with rasterio.open(file_path) as raster_dataset:
        raster_crs = raster_dataset.crs.to_string()

        transformer = pyproj.Transformer.from_crs(
            raster_crs, "EPSG:4326", always_xy=True
        )

        bands_metadata = []
        for band, band_name in bands_info:
            meta = {
                "band": band,
                "type": raster_band_type(raster_dataset, band),
                "band_name": band_field_name(band_name, band),
            }
            bands_metadata.append(meta)

        # compute whole bounds for metadata
        bounds_geog = raster_bounds(raster_dataset, transformer, "wkt")
        bounds_polygon = shapely.Polygon(shapely.wkt.loads(bounds_geog))
        bounds_coords = list(bounds_polygon.bounds)
        center_coords = list(*bounds_polygon.centroid.coords)
        center_coords.append(resolution)
        width = raster_info["Profile"]["Width"]
        height = raster_info["Profile"]["Height"]
        # assuming all windows have the same dimensions
        a_window = next(raster_dataset.block_windows())
        block_width = a_window[1].width
        block_height = a_window[1].height

        metadata["bands"] = [
            {"type": e["type"], "band_name": e["band_name"]} for e in bands_metadata
        ]
        metadata["bounds"] = bounds_coords
        metadata["center"] = center_coords
        metadata["width"] = width
        metadata["height"] = height
        metadata["block_width"] = block_width
        metadata["block_height"] = block_height
        metadata["num_blocks"] = int(width * height / block_width / block_height)
        metadata["num_pixels"] = width * height

    return metadata


def rasterio_windows_to_records(
    file_path: str,
    create_table: Callable,
    bands_info: List[Tuple[int, str]],
) -> Iterable:
    invalid_names = [
        name for _, name in bands_info if name and name.lower() in ["block", "metadata"]
    ]
    if invalid_names:
        raise ValueError(f"Invalid band names: {', '.join(invalid_names)}")

    """Open a raster file with rio-cogeo."""
    raster_info = rio_cogeo.cog_info(file_path).dict()

    """Check if raster is compatible."""
    if "GoogleMapsCompatible" != raster_info.get("Tags", {}).get(
        "Tiling Scheme", {}
    ).get("NAME"):
        msg = (
            "The input raster must be a GoogleMapsCompatible raster.\n"
            "You can make your raster compatible "
            "by converting it using the following command:\n"
            "gdalwarp -of COG -co TILING_SCHEME=GoogleMapsCompatible "
            "-co COMPRESS=DEFLATE -co OVERVIEWS=NONE -co ADD_ALPHA=NO "
            "-co RESAMPLING=NEAREST <input_raster>.tif <output_raster>.tif"
        )
        raise ValueError(msg)

    resolution = raster_info["GEO"]["MaxZoom"]

    """Requires rasterio."""
    if not _has_rasterio:  # pragma: no cover
        import_error_rasterio()

    """Open a raster file with rasterio."""
    with rasterio.open(file_path) as raster_dataset:
        raster_crs = raster_dataset.crs.to_string()

        transformer = pyproj.Transformer.from_crs(
            raster_crs, "EPSG:4326", always_xy=True
        )

        columns = table_columns(
            [band_field_name(band_name, band) for band, band_name in bands_info]
        )
        clustering = ["block"]
        create_table(columns, clustering)

        for _, window in raster_dataset.block_windows():
            record = {}
            for band, band_name in bands_info:
                newrecord = array_to_record(
                    raster_dataset.read(band, window=window),
                    band_field_name(band_name, band),
                    transformer,
                    raster_dataset.transform,
                    resolution,
                    window.row_off,
                    window.col_off,
                )

                # add the new columns generated by array_to_record
                # but leaving unchanged the index e.g. the block column
                record.update(newrecord)

            yield record


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


def create_bigquery_table(
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


def sql_quote(value: any) -> str:
    if isinstance(value, str):
        value = value.replace("\\", "\\\\")
        return f"'''{value}'''"
    return str(value)


def insert_in_bigquery_table(
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
            "(" + ",".join([sql_quote(row[column]) for column in columns]) + ")"
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


def delete_bigquery_table(
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


def check_if_bigquery_table_exists(
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


def check_if_bigquery_table_is_empty(
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


def rasterio_to_bigquery(
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
        create_table = False
        if check_if_bigquery_table_exists(project_id, dataset_id, table_id, client):
            if overwrite:
                delete_bigquery_table(project_id, dataset_id, table_id, client)
                create_table = True

            elif not check_if_bigquery_table_is_empty(
                project_id, dataset_id, table_id, client
            ):
                append_records = append or ask_yes_no_question(
                    f"Table {table_id} already exists in dataset {dataset_id} "
                    "and is not empty. Append records? [yes/no] "
                )

                if not append_records:
                    exit()
        else:
            create_table = True

        def table_creator(columns, clustering):
            if create_table:
                create_bigquery_table(
                    project_id, dataset_id, table_id, columns, clustering, client
                )

        metadata = rasterio_metadata(file_path, bands_info)

        records_gen = rasterio_windows_to_records(
            file_path,
            table_creator,
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

    except KeyboardInterrupt:
        delete_table = ask_yes_no_question(
            "Would you like to delete the partially uploaded table? [yes/no] "
        )

        if delete_table:
            delete_bigquery_table(project_id, dataset_id, table_id, client)

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
            delete_bigquery_table(project_id, dataset_id, table_id, client)

        raise IOError("Error uploading to BigQuery: {}".format(e))

    print("Done.")
    return True


def check_metadata_is_compatible(metadata, old_metadata):
    if metadata["resolution"] != old_metadata["resolution"]:
        raise ValueError(
            "Cannot append records to a table with a different resolution "
            f"({metadata['resolution']} != {old_metadata['resolution']})."
        )
    if metadata["nodata"] != old_metadata["nodata"]:
        raise ValueError(
            "Cannot append records to a table with a different nodata"
            f"({metadata['nodata']} != {old_metadata['nodata']})."
        )

    if (
        metadata["block_width"] != old_metadata["block_width"]
        or metadata["block_height"] != old_metadata["block_height"]
    ):
        raise ValueError(
            "Cannot append records to a table with a different block width/height."
        )
    if metadata["bands"] != old_metadata["bands"]:
        raise ValueError(
            "Cannot append records to a table with different bands."
            f"({metadata['bands']} != {old_metadata['bands']})."
        )


def update_metadata(metadata, old_metadata):
    metadata["bounds"] = (
        min(old_metadata["bounds"][0], metadata["bounds"][0]),
        min(old_metadata["bounds"][1], metadata["bounds"][1]),
        max(old_metadata["bounds"][2], metadata["bounds"][2]),
        max(old_metadata["bounds"][3], metadata["bounds"][3]),
    )
    metadata["center"] = (
        (metadata["bounds"][0] + metadata["bounds"][2]) / 2,
        (metadata["bounds"][1] + metadata["bounds"][3]) / 2,
        metadata["resolution"],
    )
    metadata["num_blocks"] += old_metadata["num_blocks"]
    metadata["num_pixels"] += old_metadata["num_pixels"]
    w, s, _ = quadbin.utils.point_to_tile(
        metadata["bounds"][0], metadata["bounds"][1], metadata["resolution"]
    )
    e, n, _ = quadbin.utils.point_to_tile(
        metadata["bounds"][2], metadata["bounds"][3], metadata["resolution"]
    )
    metadata["height"] = (s - n) * metadata["block_height"]
    metadata["width"] = (e - w) * metadata["block_width"]


def get_metadata(project_id, dataset_id, table_id, client=None):
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


def write_metadata(
    metadata,
    append_records,
    project_id,
    dataset_id,
    table_id,
    client=None,
):
    if append_records:
        """Requires bigquery."""
        if not _has_bigquery:  # pragma: no cover
            import_error_bigquery()

        if client is None:
            client = bigquery.Client(project=project_id)

        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        query = f"""
            UPDATE `{table_ref}`
            SET metadata = (
                SELECT TO_JSON_STRING(
                    PARSE_JSON(
                        {sql_quote(json.dumps(metadata))},
                        wide_number_mode=>'round'
                    )
                )
            ) WHERE block = 0
        """

        job = client.query(query)
        job.result()

        return True
    else:
        return insert_in_bigquery_table(
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
