import json
import sys
import math
from itertools import islice
from typing import Iterable
from typing import Callable
from typing import List
from typing import Tuple

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


def coord_range(start_x, start_y, end_x, end_y, num_subdivisions):
    # Assume all sources to be chuncked into blocks no wider than 180 deg.
    if math.fabs(end_x - start_x) > 180.0:
        end_x = math.fmod(end_x + 360.0, 360.0)
        start_x = math.fmod(start_x + 360.0, 360.0)
    return [
        [
            start_x + (end_x - start_x) * i / num_subdivisions,
            start_y + (end_y - start_y) * i / num_subdivisions,
        ]
        for i in range(0, num_subdivisions + 1)
    ]


def norm_lon(x):
    return x - 360.0 if x > 180.0 else x + 360.0 if x <= -180.0 else x


def norm_coords(coords):
    return [[norm_lon(point[0]), point[1]] for point in coords]


def polygon_geography(coords, format):
    if format == "wkt":
        return polygon_wkt(coords)
    elif format == "geojson":
        return polygon_geojson(coords)
    else:
        raise ValueError(f"Invalid geography format {format}")


def polygon_wkt(coords):
    return (
        "POLYGON(("
        + ",".join([" ".join([str(coord) for coord in point]) for point in coords])
        + "))"
    )


def polygon_geojson(coords):
    return json.dumps({"type": "Polygon", "coordinates": [coords]})


def block_geog(
    lon_NW,
    lat_NW,
    lon_NE,
    lat_NE,
    lon_SE,
    lat_SE,
    lon_SW,
    lat_SW,
    lon_subdivisions,
    lat_subdivisions,
    orientation=1,
    pseudo_planar=False,
    format="wkt",
):
    if orientation < 0:
        coords = (
            coord_range(lon_NW, lat_NW, lon_NE, lat_NE, lon_subdivisions)
            + coord_range(lon_NE, lat_NE, lon_SE, lat_SE, lat_subdivisions)
            + coord_range(lon_SE, lat_SE, lon_SW, lat_SW, lon_subdivisions)
            + coord_range(lon_SW, lat_SW, lon_NW, lat_NW, lat_subdivisions)
        )
    else:
        coords = (
            coord_range(lon_SW, lat_SW, lon_SE, lat_SE, lon_subdivisions)
            + coord_range(lon_SE, lat_SE, lon_NE, lat_NE, lat_subdivisions)
            + coord_range(lon_NE, lat_NE, lon_NW, lat_NW, lon_subdivisions)
            + coord_range(lon_NW, lat_NW, lon_SW, lat_SW, lat_subdivisions)
        )
    if pseudo_planar:
        coords = [pseudoplanar(p[0], p[1]) for p in coords]
    else:
        coords = norm_coords(coords)
    return polygon_geography(coords, format)


def pseudoplanar(x, y):
    return [x / 32768.0, y / 32768.0]


def band_field_name(band: int, band_type: str, base_name: str = "band") -> str:
    return "_".join([base_name, str(band), band_type])


def array_to_record(
    arr: np.ndarray,
    band: int,
    value_field: str,
    dtype_str: str,
    transformer: pyproj.Transformer,
    geotransform: Affine,
    row_off: int = 0,
    col_off: int = 0,
    crs: str = "EPSG:4326",
    orientation=1,
    pseudo_planar: bool = False,
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

    # use 1 subdivision in 64 pixels
    lon_subdivisions = math.ceil(width / 64.0)
    lat_subdivisions = math.ceil(height / 64.0)
    geog = block_geog(
        lon_NW,
        lat_NW,
        lon_NE,
        lat_NE,
        lon_SE,
        lat_SE,
        lon_SW,
        lat_SW,
        lon_subdivisions,
        lat_subdivisions,
        orientation,
        pseudo_planar,
    )

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
        "geog": geog,
        "block_height": height,
        "block_width": width,
        "attrs": json.dumps(attrs),
        value_field: arr_bytes,
    }

    return record


def array_to_quadbin_record(
    arr: np.ndarray,
    band: int,
    value_field: str,
    dtype_str: str,
    transformer: pyproj.Transformer,
    geotransform: Affine,
    resolution: int,
    row_off: int = 0,
    col_off: int = 0,
    crs: str = "EPSG:4326",
) -> dict:
    """Requires quadbin."""
    if not _has_quadbin:  # pragma: no cover
        import_error_quadbin()

    height, width = arr.shape

    x, y = transformer.transform(
        *(geotransform * (col_off + width * 0.5, row_off + height * 0.5))
    )

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


def raster_band_type(raster_dataset: rasterio.io.DatasetReader, band: int) -> str:
    types = {
        i: dtype for i, dtype in zip(raster_dataset.indexes, raster_dataset.dtypes)
    }
    return str(types[band])


def table_columns(quadbin: bool, bands: List[str]) -> List[Tuple[str, str, str]]:
    if quadbin:
        columns = [
            ("quadbin", "INTEGER", "NULLABLE"),
        ]
    else:
        columns = [
            ("lon_NW", "FLOAT", "NULLABLE"),
            ("lat_NW", "FLOAT", "NULLABLE"),
            ("lon_NE", "FLOAT", "NULLABLE"),
            ("lat_NE", "FLOAT", "NULLABLE"),
            ("lon_SE", "FLOAT", "NULLABLE"),
            ("lat_SE", "FLOAT", "NULLABLE"),
            ("lon_SW", "FLOAT", "NULLABLE"),
            ("lat_SW", "FLOAT", "NULLABLE"),
            ("geog", "GEOGRAPHY", "NULLABLE"),
        ]
    columns += [
        ("block_height", "INTEGER", "NULLABLE"),
        ("block_width", "INTEGER", "NULLABLE"),
        ("attrs", "STRING", "REQUIRED"),
        # TODO: upgrade BQ client version and use 'JSON' type for 'attrs'
    ]
    columns += [(band_name, "BYTES", "NULLABLE") for band_name in bands]
    return columns


def raster_bounds(raster_dataset, transformer, pseudo_planar, format):
    # compute whole bounds for metadata
    width = raster_dataset.width
    height = raster_dataset.width
    lon_NW, lat_NW = transformer.transform(*(raster_dataset.transform * (0, 0)))
    lon_NE, lat_NE = transformer.transform(*(raster_dataset.transform * (width, 0)))
    lon_SW, lat_SW = transformer.transform(*(raster_dataset.transform * (0, height)))
    lon_SE, lat_SE = transformer.transform(
        *(raster_dataset.transform * (width, height))
    )
    # use 1 subdivision in 64 pixels
    lon_subdivisions = math.ceil(width / 64.0)
    lat_subdivisions = math.ceil(height / 64.0)
    return block_geog(
        lon_NW,
        lat_NW,
        lon_NE,
        lat_NE,
        lon_SE,
        lat_SE,
        lon_SW,
        lat_SW,
        lon_subdivisions,
        lat_subdivisions,
        raster_orientation(raster_dataset),
        pseudo_planar,
        format,
    )


def rasterio_windows_to_records(
    file_path: str,
    create_table: Callable,
    band: int,
    metadata: dict,
    input_crs: str = None,
    output_quadbin: bool = False,
    pseudo_planar: bool = False,
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
        orientation = raster_orientation(raster_dataset)

        band_type = raster_band_type(raster_dataset, band)
        band_name = band_field_name(band, band_type)
        columns = table_columns(output_quadbin, [band_name])
        clustering = ["quadbin"] if output_quadbin else ["geog"]
        create_table(columns, clustering)

        # compute whole bounds for metadata
        bounds_geog = raster_bounds(
            raster_dataset, transformer, pseudo_planar, "geojson"
        )

        # FIXME: any metadata changes needed for quadbin output?
        # Missing metadata:
        # raster_area (area of bounds_geog) can be computed in BQ as
        #   SELECT ST_AREA(ST_GEOGFROMGEOJSON(JSON_VALUE(attrs, '$.raster_boundary')))
        #   FROM raster_table WHERE geog IS NULL
        # avg_pixel_area (average pixel area) can be computed in BQ as
        #   SELECT AVG(ST_AREA(geog)/(block_height*block_width))
        #   FROM raster_table WHERE geog IS NOT NULL
        metadata["bands"] = [band_name]
        metadata["raster_boundary"] = bounds_geog
        metadata["width_in_pixel"] = raster_dataset.width
        metadata["height_in_pixel"] = raster_dataset.height
        metadata["total_pixels"] = (
            metadata["width_in_pixel"] * metadata["height_in_pixel"]
        )
        metadata["nb_pixel_blocks"] = 0
        metadata["nb_pixel"] = 0
        metadata["max_pixel_block_height_in_pixel"] = 0
        metadata["max_pixel_block_width_in_pixel"] = 0
        metadata["irregular_pixel_block_shape"] = False

        for _, window in raster_dataset.block_windows():

            if output_quadbin:
                rec = array_to_quadbin_record(
                    raster_dataset.read(band, window=window),
                    band,
                    band_name,
                    str(band_type),
                    transformer,
                    raster_dataset.transform,
                    resolution,
                    window.row_off,
                    window.col_off,
                    crs=input_crs,
                )

            else:
                rec = array_to_record(
                    raster_dataset.read(band, window=window),
                    band,
                    band_name,
                    str(band_type),
                    transformer,
                    raster_dataset.transform,
                    window.row_off,
                    window.col_off,
                    crs=input_crs,
                    orientation=orientation,
                    pseudo_planar=pseudo_planar,
                )

            metadata["nb_pixel_blocks"] += 1
            metadata["nb_pixel"] += window.width * window.height
            if metadata["max_pixel_block_height_in_pixel"] < window.height:
                metadata["max_pixel_block_height_in_pixel"] = window.height
            if metadata["max_pixel_block_width_in_pixel"] < window.width:
                metadata["max_pixel_block_width_in_pixel"] = window.width
            if (
                window.height != metadata["max_pixel_block_height_in_pixel"]
                or window.width != metadata["max_pixel_block_width_in_pixel"]
            ):
                metadata["irregular_pixel_block_shape"] = True

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


def create_bigquery_table(
    table_id: str,
    dataset_id: str,
    project_id: str,
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
    rows,
    table_id: str,
    dataset_id: str,
    project_id: str,
    client=None,
) -> bool:
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


def raster_orientation(raster_dataset):
    raster_crs = raster_dataset.crs.to_string()
    transformer = pyproj.Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
    x0, y0 = transformer.transform(*(raster_dataset.transform * (0, 0)))
    x1, y1 = transformer.transform(*(raster_dataset.transform * (0, 1)))
    x2, y2 = transformer.transform(*(raster_dataset.transform * (1, 0)))
    b11 = x1 - x0
    b12 = y1 - y0
    b21 = x2 - x0
    b22 = y2 - y0
    d = b11 * b22 - b12 * b21
    return -1 if d < 0 else 1


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
    pseudo_planar: bool = False,
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
    pseudo_planar : bool, optional
        Use pseudo-planar BigQuery geographies (coordinates are scaled down by 1/32768)

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

    if client is None:  # pragma: no cover
        client = bigquery.Client(project=project_id)

    try:
        create_table = False
        if check_if_bigquery_table_exists(dataset_id, table_id, client):
            if overwrite:
                delete_bigquery_table(table_id, dataset_id, project_id, client)
                create_table = True

            elif not check_if_bigquery_table_is_empty(dataset_id, table_id, client):
                append_records = ask_yes_no_question(
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
                    table_id, dataset_id, project_id, columns, clustering, client
                )

        metadata = {}

        records_gen = rasterio_windows_to_records(
            file_path,
            table_creator,
            # metadata_writer,
            band,
            metadata,
            input_crs,
            output_quadbin,
            pseudo_planar,
        )

        total_blocks = get_number_of_blocks(file_path)
        metadata["total_pixel_blocks"] = total_blocks  # FIXME: for debugging purposes

        if chunk_size is None:
            job = records_to_bigquery(
                records_gen, table_id, dataset_id, project_id, client=client
            )
            # raise error if job went wrong (blocking call)
            job.result()
        else:
            from tqdm.auto import tqdm

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

            # raise error if something went wrong (blocking call)
            while jobs:
                jobs.pop().result()

        # Write metadata
        insert_in_bigquery_table(
            [{"attrs": json.dumps(metadata)}],
            table_id,
            dataset_id,
            project_id,
            client=client,
        )

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
