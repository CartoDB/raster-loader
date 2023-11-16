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
import functools
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
    num_subdivisions = max(num_subdivisions, 1)
    return [
        [
            start_x + (end_x - start_x) * i / num_subdivisions,
            start_y + (end_y - start_y) * i / num_subdivisions,
        ]
        for i in range(0, num_subdivisions + 1)
    ]


MIN_SUBDIVISIONS_PER_DEGREE = 0.3


def line_coords(start_x, start_y, end_x, end_y, num_subdivisions, whole_earth):
    if not whole_earth and math.fabs(end_x - start_x) > 180.0:
        end_x = math.fmod(end_x + 360.0, 360.0)
        start_x = math.fmod(start_x + 360.0, 360.0)

    x_length = math.fabs(end_x - start_x)
    y_length = math.fabs(end_y - start_y)

    near_meridian = x_length <= 1e-13
    near_parallel = y_length < 1.0

    if near_parallel:
        scale = math.cos(0.5 * math.pi / 180.0 * (start_y + end_y))
        x_length *= scale
        num_subdivisions = math.ceil(num_subdivisions * scale)
        num_subdivisions = max(
            num_subdivisions, math.ceil(x_length * MIN_SUBDIVISIONS_PER_DEGREE)
        )

    if near_meridian:
        # No need to subdivide meridians, since they're geodesics
        num_subdivisions = 1

    return coord_range(start_x, start_y, end_x, end_y, num_subdivisions)


def norm_lon(x):
    return x - 360.0 if x > 180.0 else x + 360.0 if x <= -180.0 else x


def norm_coords(coords):
    return [[norm_lon(point[0]), point[1]] for point in coords]


def polygon_geography(rings, format, normalize_coords, pseudo_planar=False):
    if pseudo_planar:
        rings = [[pseudoplanar(p[0], p[1]) for p in coords] for coords in rings]
    elif normalize_coords:
        rings = [norm_coords(coords) for coords in rings]

    if format == "wkt":
        return polygon_wkt(rings)
    elif format == "geojson":
        return polygon_geojson(rings)
    else:
        raise ValueError(f"Invalid geography format {format}")


def polygon_wkt(rings):
    return (
        "POLYGON("
        + ",".join(
            [
                "("
                + ",".join(
                    [" ".join([str(coord) for coord in point]) for point in coords]
                )
                + ")"
                for coords in rings
            ]
        )
        + ")"
    )


def polygon_geojson(rings):
    return json.dumps({"type": "Polygon", "coordinates": rings})


def section_point(x, y):
    if x == 180.0:
        x = -180.0
    return [x, y]


def section_geog(lat_N, lat_S, num_subdivisions, pseudo_planar, format):
    lat_S, lat_N = min(lat_N, lat_S), max(lat_N, lat_S)
    num_subdivisions = max(num_subdivisions, 4)
    ring1 = [
        section_point(*p)
        for p in coord_range(-180.0, lat_S, 180.0, lat_S, num_subdivisions)
    ]
    ring2 = [
        section_point(*p)
        for p in coord_range(180.0, lat_N, -180.0, lat_N, num_subdivisions)
    ]
    if lat_N < 0:
        ring1, ring2 = ring2, ring1
    return polygon_geography([ring1, ring2], format, False, pseudo_planar)


def coords_to_geography(coords, format, whole_earth, pseudo_planar):
    # remove too-close coordinates cause they cause errors
    # in BigQuery's ST_GEOGFROMGEOJSON
    def are_too_close(point1, point2):
        return (
            math.fabs(point1[0] - point2[0]) <= 1e-13
            and math.fabs(point1[1] - point2[1]) <= 1e-13
        )

    def filter_near_points(coords, point):
        previous = None if not coords else coords[-1]
        if not previous or not are_too_close(previous, point):
            coords.append(point)
        return coords

    coords = functools.reduce(filter_near_points, coords, [])

    # now let's make sure the initial and final points are exactly the same
    if coords[0] != coords[-1]:
        # replace the last point; never mind, it must be very close
        coords[-1] = coords[0]
    return polygon_geography([coords], format, not whole_earth, pseudo_planar)


def pseudoplanar(x, y):
    return [x / 32768.0, y / 32768.0]


def band_field_name(band: int, custom_name: str = None) -> str:
    if custom_name and custom_name.lower() != "none":
        return custom_name
    else:
        return "band_" + str(band)


def array_to_quadbin_record(
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

    block_quadbin = quadbin.point_to_cell(x, y, resolution)

    if should_swap[arr.dtype.byteorder]:
        arr_bytes = np.ascontiguousarray(arr.byteswap()).tobytes()
    else:
        arr_bytes = np.ascontiguousarray(arr).tobytes()

    record = {
        "block": block_quadbin,
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


def table_columns(bands: List[str]) -> List[Tuple[str, str, str]]:
    columns = [
        ("block", "INTEGER", "REQUIRED"),
    ]
    columns += [
        ("metadata", "STRING", "NULLABLE"),
        # TODO: upgrade BQ client version and use 'JSON' type for 'attrs'
    ]
    columns += [(band_name, "BYTES", "NULLABLE") for band_name in bands]
    return columns


def raster_bounds(raster_dataset, transformer, pseudo_planar, format):
    raster_dataset.transform,
    min_x = 0
    min_y = 0
    max_x = raster_dataset.width
    max_y = raster_dataset.height

    x_subdivisions = math.ceil((max_x - min_x) / 64.0)
    y_subdivisions = math.ceil((max_y - min_y) / 64.0)
    pixel_coords = (
        # SW -> SE
        coord_range(min_x, max_y, max_x, max_y, x_subdivisions)
        # SE -> NE
        + coord_range(max_x, max_y, max_x, min_y, y_subdivisions)
        # NE -> NW
        + coord_range(max_x, min_y, min_x, min_y, x_subdivisions)
        # NW -> SW
        + coord_range(min_x, min_y, min_x, max_y, y_subdivisions)
    )
    coords = [
        transformer.transform(*(raster_dataset.transform * (x, y)))
        for x, y in pixel_coords
    ]
    lon_NW, _ = transformer.transform(*(raster_dataset.transform * (min_x, min_y)))
    lon_NE, _ = transformer.transform(*(raster_dataset.transform * (max_x, min_y)))
    lon_SW, _ = transformer.transform(*(raster_dataset.transform * (min_x, max_y)))
    lon_SE, _ = transformer.transform(*(raster_dataset.transform * (max_x, max_y)))
    whole_earth = (
        math.fabs(lon_NW - lon_NE) >= 360.0 and math.fabs(lon_SW - lon_SE) >= 360
    )

    return coords_to_geography(coords, format, whole_earth, pseudo_planar)


def rasterio_windows_to_records(
    file_path: str,
    create_table: Callable,
    bands_info: List[Tuple[int, str]],
    metadata: dict,
    input_crs: str = None,
    pseudo_planar: bool = False,
) -> Iterable:
    invalid_names = [
        name for _, name in bands_info if name and name.lower() in ["block", "metadata"]
    ]
    if invalid_names:
        raise ValueError(f"Invalid band column names: {', '.join(invalid_names)}")

    """Open a raster file with rio-cogeo."""
    raster_info = rio_cogeo.cog_info(file_path).dict()

    """Check if raster is quadbin compatible."""
    if "GoogleMapsCompatible" != raster_info.get("Tags", {}).get(
        "Tiling Scheme", {}
    ).get("NAME"):
        msg = (
            "The input raster must be a GoogleMapsCompatible raster.\n"
            "You can make your raster compatible "
            "by converting it using the following command:\n"
            "gdalwarp your_raster.tif -of COG "
            "-co TILING_SCHEME=GoogleMapsCompatible -co COMPRESS=DEFLATE "
            "your_compatible_raster.tif"
        )
        raise ValueError(msg)

    resolution = raster_info["GEO"]["MaxZoom"]
    metadata["resolution"] = resolution

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

        bands_metadata = []
        for band, band_column_name in bands_info:
            band_type = raster_band_type(raster_dataset, band)
            band_name = band_field_name(band, band_column_name)
            meta = {
                "band": band,
                "type": band_type,
                "band_name": band_name,
            }
            bands_metadata.append(meta)

        columns = table_columns([e["band_name"] for e in bands_metadata])
        clustering = ["block"]
        create_table(columns, clustering)

        # compute whole bounds for metadata
        bounds_geog = raster_bounds(raster_dataset, transformer, pseudo_planar, "wkt")
        bounds_polygon = shapely.Polygon(shapely.wkt.loads(bounds_geog))
        center_coords = list(*bounds_polygon.centroid.coords)
        center_coords.append(resolution)

        # assuming all windows have the same dimensions
        a_window = next(raster_dataset.block_windows())

        metadata["bands"] = [
            {"type": e["type"], "band_name": e["band_name"]} for e in bands_metadata
        ]
        metadata["bounds"] = list(bounds_polygon.bounds)
        metadata["center"] = center_coords
        metadata["width"] = raster_info["Profile"]["Width"]
        metadata["height"] = raster_info["Profile"]["Height"]
        metadata["block_width"] = a_window[1].width
        metadata["block_height"] = a_window[1].height
        metadata["num_blocks"] = 0
        metadata["num_pixels"] = 0

        for _, window in raster_dataset.block_windows():
            record = {}
            for band_metadata in bands_metadata:
                band = band_metadata["band"]
                band_name = band_metadata["band_name"]
                band_type = band_metadata["type"]

                newrecord = array_to_quadbin_record(
                    raster_dataset.read(band, window=window),
                    band_name,
                    transformer,
                    raster_dataset.transform,
                    resolution,
                    window.row_off,
                    window.col_off,
                )

                # add the new columns generated by array_to_quadbin_record
                # but leaving unchanged the index e.g. the block column
                record.update(newrecord)

            metadata["num_blocks"] += 1
            metadata["num_pixels"] += window.width * window.height

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
        data_df, f"{project_id}.{dataset_id}.{table_id}"
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


def run_bigquery_query(
    query: str,
    project_id: str,
    client=None,
) -> bool:
    """Run a BigQuery query

    Parameters
    ----------
    query : str
        Query to be run (standard SQL)
    project_id : str
        Project where the query is ran.
    client : google.cloud.bigquery.client.Client, optional
        BigQuery client, by default None

    Returns
    -------
    bool
        True if the query ran successfully.

    """

    """Requires bigquery."""
    if not _has_bigquery:  # pragma: no cover
        import_error_bigquery()

    if client is None:
        client = bigquery.Client(project=project_id)

    job = client.query(query)
    return job.result()


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
    bands_info: List[Tuple[int, str]] = [(1, None)],
    chunk_size: int = None,
    input_crs: int = None,
    client=None,
    overwrite: bool = False,
    append: bool = False,
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
    bands_info : [(int, str)], optional
        Band number(s) and column name(s) to read from the
        raster file, by default [(1, None)].
        If column_name is None default column name is band_<band_num>.
    chunk_size : int, optional
        Number of records to write to BigQuery at a time, the default (None)
        writes all records in a single batch.
    input_crs : int, optional
        Input CRS, by default None
    client : [bigquery.Client()], optional
        BigQuery client, by default None
    overwrite : bool, optional
        Overwrite the table if it already contains data, by default False
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

        metadata = {}

        records_gen = rasterio_windows_to_records(
            file_path,
            table_creator,
            bands_info,
            metadata,
            input_crs,
            pseudo_planar,
        )

        total_blocks = get_number_of_blocks(file_path)
        # metadata["total_pixel_blocks"] = total_blocks  # FIXME: for debugging purposes

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

        write_metadata(
            metadata,
            append_records,
            project_id,
            dataset_id,
            table_id,
            client=client,
        )

        # !!! outdated due the fact that raster windows is constant for each
        # row in the table and area is that stated by the rasterio directly.
        # Postprocess: compute metadata areas
        # run_bigquery_query(
        #     inject_areas_query(f"{project_id}.{dataset_id}.{table_id}"),
        #     project_id,
        #     client=client,
        # )

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


def write_metadata(
    metadata,
    append_records,
    project_id,
    dataset_id,
    table_id,
    client=None,
):
    if append_records:
        table_ref = f"{project_id}.{dataset_id}.{table_id}"
        location = "block"
        query = f"""
            UPDATE `{table_ref}`
            SET metadata = (
                SELECT TO_JSON_STRING(
                    PARSE_JSON(
                        {sql_quote(json.dumps(metadata))}
                    )
                )
            ) WHERE {location} = 0
        """

        """Requires bigquery."""
        if not _has_bigquery:  # pragma: no cover
            import_error_bigquery()

        if client is None:
            client = bigquery.Client(project=project_id)

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


def inject_areas_query(raster_table: str) -> str:
    location_column = "block"
    from_metadata_source = f"FROM `{raster_table}` WHERE {location_column} = 0"
    from_blocks_source = f"FROM `{raster_table}` WHERE {location_column} != 0"

    block_y = "'$.y'"
    block_x = "'$.x'"
    avg_pixel_area_query = f"""
        SELECT
            AVG(
            CAST(JSON_VALUE(metadata, '$.block_area') AS FLOAT64)
            / (block_height*block_width)
            )
        {from_blocks_source}
    """
    area_query = f"""
        SELECT SUM(CAST(JSON_VALUE(metadata, '$.block_area') AS FLOAT64))
        {from_blocks_source}
    """

    width_in_pixel_query = f"""
        SELECT MAX(row_width) FROM (
          SELECT SUM(block_width) OVER (
            PARTITION BY INT64(JSON_QUERY(
                PARSE_JSON(metadata, wide_number_mode=>'round'), {block_y}))
          ) AS row_width
          {from_blocks_source}
        )
    """

    height_in_pixel_query = f"""
        SELECT MAX(col_height) FROM (
          SELECT SUM(block_height) OVER (
            PARTITION BY INT64(JSON_QUERY(
                PARSE_JSON(metadata, wide_number_mode=>'round'), {block_x}))
          ) AS col_height
          {from_blocks_source}
        )
    """

    width_in_pixel_block_query = f"""
        SELECT COUNT(DISTINCT INT64(JSON_QUERY(
            PARSE_JSON(metadata, wide_number_mode=>'round'), {block_x})))
        {from_blocks_source}
    """

    height_in_pixel_block_query = f"""
        SELECT COUNT(DISTINCT INT64(JSON_QUERY(
            PARSE_JSON(metadata, wide_number_mode=>'round'), {block_y})))
        {from_blocks_source}
    """

    sparse_pixel_block_query = f"""
      (SELECT INT64(JSON_QUERY(
        PARSE_JSON(metadata, wide_number_mode=>'round'), '$.nb_pixel_blocks'))
       {from_metadata_source})
      < ({width_in_pixel_block_query}) * ({height_in_pixel_block_query})
    """

    def only_if_unique_crs_query(query):
        return f"""IF(
            (SELECT
              JSON_VALUE(metadata, '$.crs') IS NOT NULL
              AND
              JSON_QUERY_ARRAY(metadata, '$.gdal_transform') IS NOT NULL,
             {from_metadata_source}),
            ({query}),
            NULL)
        """

    return f"""
        CREATE TEMP FUNCTION _mergeJSONs(a JSON, b JSON)
          RETURNS JSON
          LANGUAGE js
           AS r'return {{...a, ...b}};';
        UPDATE `{raster_table}`
        SET metadata = TO_JSON_STRING(_mergeJSONs(
          PARSE_JSON(metadata, wide_number_mode=>'round'),
          TO_JSON(STRUCT(
            ({area_query}) AS raster_area,
            ({avg_pixel_area_query}) AS avg_pixel_area,
            ({width_in_pixel_query}) AS width_in_pixel,
            ({height_in_pixel_query}) AS height_in_pixel,
            ({width_in_pixel_block_query}) AS width_in_pixel_block,
            ({height_in_pixel_block_query}) AS height_in_pixel_block,
            ({sparse_pixel_block_query}) AS sparse_pixel_block
          ))
        ))
        WHERE {location_column} = 0;
    """
