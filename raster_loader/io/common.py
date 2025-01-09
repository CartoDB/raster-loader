import math
import numpy as np
import pyproj
import shapely
import sys
import zlib

from raster_loader._version import __version__
from collections import Counter
from typing import Dict, Callable, Iterable, List, Tuple, Union
from affine import Affine
from shapely import wkt  # Can not use directly from shapely.wkt

import rio_cogeo
import rasterio
import quadbin

from raster_loader.geo import raster_bounds
from raster_loader.errors import (
    error_not_google_compatible,
)
from raster_loader.utils import warnings

DEFAULT_COG_BLOCK_SIZE = 256

DEFAULT_TYPES_NODATA_VALUES = {
    "int8": -128,
    "int16": -32768,
    "int32": -2147483648,
    "int64": -9223372036854775808,
    "uint8": 0,
    "uint16": 65535,
    "uint32": 4294967295,
    "uint64": 18446744073709551615,
    "float16": np.nan,
    "float32": np.nan,
    "float64": np.nan,
}

DEFAULT_MAX_MOST_COMMON = 10
DEFAULT_SAMPLING_MAX_ITERATIONS = 10
DEFAULT_SAMPLING_MAX_SAMPLES = 1000
DEFAULT_OVERVIEWS = range(3, 20)

should_swap = {"=": sys.byteorder != "little", "<": False, ">": True, "|": False}

Samples = Dict[int, List[Union[int, float]]]


def band_field_name(custom_name: str, band: int, band_rename_function: Callable) -> str:
    return band_rename_function(custom_name or "band_" + str(band))


def get_nodata_value(raster_dataset: rasterio.io.DatasetReader) -> float:
    value = band_nodata_value(raster_dataset, 1)
    # So far we only support one nodata value for all bands
    if raster_dataset.nodata is None:
        for band in raster_dataset.indexes:
            band_value = band_nodata_value(raster_dataset, band)
            # Note (np.nan != np.nan) == True
            both_nan = np.isnan(band_value) and np.isnan(value)
            if (not both_nan) and (band_value != value):
                raise ValueError("Invalid no data value")
    return value


def band_original_nodata_value(
    raster_dataset: rasterio.io.DatasetReader, band: int
) -> float:
    nodata_value = raster_dataset.nodata
    if nodata_value is None:
        nodata_value = raster_dataset.get_nodatavals()[band - 1]
    return nodata_value


def band_value_as_string(
    raster_dataset: rasterio.io.DatasetReader, band: int, value: float
) -> str:
    return (
        str(np.array(value).astype(raster_dataset.dtypes[band - 1]))
        if value is not None
        else None
    )


def band_nodata_value(raster_dataset: rasterio.io.DatasetReader, band: int) -> float:
    nodata_value = band_original_nodata_value(raster_dataset, band)
    if nodata_value is None:
        nodata_value = get_default_nodata_value(raster_dataset.dtypes[band - 1])
    return nodata_value


def get_default_nodata_value(dtype: str) -> float:
    if dtype in DEFAULT_TYPES_NODATA_VALUES:
        return DEFAULT_TYPES_NODATA_VALUES[dtype]
    else:
        raise ValueError(f"Unsupported data type: {dtype}")


# For Python <=3.10 compatibility (handling wbits parameter)
# TODO: Remove this once we drop support for Python < 3.11
if sys.version_info < (3, 11):

    def compress_bytes(arr_bytes):
        compressed = zlib.compress(arr_bytes, level=6)
        # Add gzip header corresponding to wbits=31
        return b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + compressed

else:

    def compress_bytes(arr_bytes):
        return zlib.compress(arr_bytes, level=6, wbits=31)


def array_to_record(
    arr: np.ndarray,
    value_field: str,
    band_rename_function: Callable,
    transformer: pyproj.Transformer,
    geotransform: Affine,
    resolution: int,
    window: rasterio.windows.Window,
    no_data_value: float = None,
    compress: bool = False,
) -> dict:
    row_off = window.row_off
    col_off = window.col_off
    width = window.width
    height = window.height

    # Skip blocks without any data to relieve loading burden
    if no_data_value is not None and np.all(arr == no_data_value):
        return None

    x, y = transformer.transform(
        *(geotransform * (col_off + width * 0.5, row_off + height * 0.5))
    )

    block = quadbin.point_to_cell(x, y, resolution)

    if should_swap[arr.dtype.byteorder]:
        arr_bytes = np.ascontiguousarray(arr.byteswap()).tobytes()
    else:
        arr_bytes = np.ascontiguousarray(arr).tobytes()

    # Apply compression if requested
    arr_bytes = compress_bytes(arr_bytes) if compress else arr_bytes

    record = {
        band_rename_function("block"): block,
        band_rename_function("metadata"): None,
        value_field: arr_bytes,
    }

    return record


def raster_band_type(raster_dataset: rasterio.io.DatasetReader, band: int) -> str:
    types = {
        i: dtype for i, dtype in zip(raster_dataset.indexes, raster_dataset.dtypes)
    }
    return str(types[band])


def get_resolution_and_block_sizes(
    raster_dataset: rasterio.io.DatasetReader, raster_info: dict
):
    # assuming all windows have the same dimensions
    a_window = next(raster_dataset.block_windows())
    block_width = a_window[1].width
    block_height = a_window[1].height
    resolution = int(
        raster_info["GEO"]["MaxZoom"]
        - math.log(
            block_width
            / DEFAULT_COG_BLOCK_SIZE
            * block_height
            / DEFAULT_COG_BLOCK_SIZE,
            4,
        )
    )
    return block_width, block_height, resolution


def get_color_name(raster_dataset: rasterio.io.DatasetReader, band: int) -> str:
    try:
        # There is [an issue](https://github.com/OSGeo/gdal/issues/1928)
        # in gdal with the same error message that we see in this line:
        #   "Failed to compute statistics, no valid pixels found in sampling."
        #
        # It seems to be an error with cropped rasters.
        band_colorinterp = raster_dataset.colorinterp[band - 1].name
    except Exception:
        band_colorinterp = None

    return band_colorinterp


def get_color_table(raster_dataset: rasterio.io.DatasetReader, band: int):
    try:
        if get_color_name(raster_dataset, band) == "palette":
            return raster_dataset.colormap(band)
        return None
    except ValueError:
        return None


def rasterio_metadata(
    file_path: str,
    bands_info: List[Tuple[int, str]],
    band_rename_function: Callable,
    exact_stats: bool = False,
    basic_stats: bool = False,
    compress: bool = False,
):
    """Open a raster file with rasterio."""
    raster_info = rio_cogeo.cog_info(file_path).dict()

    """Check if raster is compatible."""
    if "GoogleMapsCompatible" != raster_info.get("Tags", {}).get(
        "Tiling Scheme", {}
    ).get("NAME"):
        error_not_google_compatible()

    metadata = {}
    width = raster_info["Profile"]["Width"]
    height = raster_info["Profile"]["Height"]

    # Add compression info to metadata
    metadata["compression"] = "gzip" if compress else None

    with rasterio.open(file_path) as raster_dataset:
        raster_crs = raster_dataset.crs.to_string()

        transformer = pyproj.Transformer.from_crs(
            raster_crs, "EPSG:4326", always_xy=True
        )

        block_width, block_height, resolution = get_resolution_and_block_sizes(
            raster_dataset, raster_info
        )

        metadata["block_resolution"] = resolution
        metadata["minresolution"] = resolution - len(raster_dataset.overviews(1))
        metadata["maxresolution"] = resolution
        metadata["nodata"] = get_nodata_value(raster_dataset)
        # NaN cannot be represented in JSON so we'll use `null`.
        # Note that json.dumps produces invalid JSON representaion of `float('nan')`.
        if metadata["nodata"] is not None and math.isnan(metadata["nodata"]):
            metadata["nodata"] = None
        bands_metadata = []

        # We only need to sample if we are not computing exact stats
        # and we need to sample from the raster dataset just once!
        if not exact_stats:
            samples = sample_not_masked_values(
                raster_dataset, DEFAULT_SAMPLING_MAX_SAMPLES
            )

        for band, band_name in bands_info:

            if exact_stats:
                print("Computing exact stats...")
                warnings.warn(
                    "Exact statistics can be quite resources demanding. "
                    "User is encourage to compute approximate statistics instead.",
                    UserWarning,
                )
                stats = raster_band_stats(raster_dataset, band, basic_stats)
            else:
                print("Computing approximate stats...")
                stats = raster_band_approx_stats(
                    raster_dataset, samples, band, basic_stats
                )

            band_colorinterp = get_color_name(raster_dataset, band)
            if band_colorinterp == "alpha":
                band_nodata = "0"
            else:
                band_nodata = band_value_as_string(
                    raster_dataset, band, band_nodata_value(raster_dataset, band)
                )
            meta = {
                "band": band,
                "type": raster_band_type(raster_dataset, band),
                "name": band_field_name(band_name, band, band_rename_function),
                "colorinterp": band_colorinterp,
                "colortable": get_color_table(raster_dataset, band),
                "stats": stats,
                "nodata": band_nodata,
            }
            bands_metadata.append(meta)

        # compute whole bounds for metadata
        bounds_geog = raster_bounds(raster_dataset, transformer, "wkt")
        bounds_polygon = shapely.Polygon(wkt.loads(bounds_geog))
        bounds_coords = list(bounds_polygon.bounds)
        center_coords = list(*bounds_polygon.centroid.coords)
        center_coords.append(resolution)
        pixel_resolution = int(resolution + math.log(block_width * block_height, 4))
        if pixel_resolution > 26:
            raise ValueError(
                "Input raster pixel resolution exceeds "
                "the max supported resolution of 26.\n"
                "Please resample the raster to a lower resolution."
            )

        metadata["bands"] = [
            {
                "type": e["type"],
                "name": e["name"],
                "stats": e["stats"],
                "colorinterp": e["colorinterp"],
                "nodata": e["nodata"],
                "colortable": e["colortable"],
            }
            for e in bands_metadata
        ]
        metadata["bounds"] = bounds_coords
        metadata["center"] = center_coords
        metadata["width"] = width
        metadata["height"] = height
        metadata["block_width"] = block_width
        metadata["block_height"] = block_height
        metadata["num_blocks"] = int(width * height / block_width / block_height)
        metadata["num_pixels"] = width * height
        metadata["pixel_resolution"] = pixel_resolution

    return metadata


def get_alpha_band(raster_dataset: rasterio.io.DatasetReader):
    for band in raster_dataset.indexes:
        if get_color_name(raster_dataset, band) == "alpha":
            return band
    return None


def read_masked(
    raster_dataset: rasterio.io.DatasetReader, band: int, alpha_band: int, **args
) -> np.ma.masked_array:
    if alpha_band and raster_dataset.dtypes[alpha_band - 1] != "uint8":
        unmasked_data = raster_dataset.read(band, **args)
        return np.ma.masked_array(
            data=unmasked_data, mask=raster_dataset.read(alpha_band, **args) == 0
        )
    else:
        # Note that if no alpha band exists, then for RGB bands this mask will
        # incorrectly mask pixels that have only some of the RGB components equal
        # to their nodata value.
        # But since this case is only used for filling with nodata values, this is OK,
        # since we'll replace the masked values for this individual band with the
        # nodata value.
        return raster_dataset.read(band, masked=True, **args)


def read_filled(
    raster_dataset: rasterio.io.DatasetReader, band: int, no_data_value, **args
) -> np.array:
    alpha_band = get_alpha_band(raster_dataset)
    if band == alpha_band:
        return raster_dataset.read(band, fill_value=0, **args)
    else:
        masked_array = read_masked(raster_dataset, band, alpha_band, **args)
        return masked_array.filled(fill_value=no_data_value)


def get_compound_bands(
    raster_dataset: rasterio.io.DatasetReader, band: int
) -> List[int]:
    """Get all the bands whose nodata values should be compounded with the given one"""
    rgb = ["red", "green", "blue"]
    if raster_dataset.colorinterp[band - 1].name in rgb:
        rgb_bands = [
            b
            for b in raster_dataset.indexes
            if raster_dataset.colorinterp[b - 1].name in rgb
        ]
        if len(rgb_bands) == 3:
            return rgb_bands
    return [band]


def band_is_float(raster_dataset: rasterio.io.DatasetReader, band: int) -> bool:
    return np.dtype(raster_dataset.dtypes[band - 1]) in [
        "float16",
        "float32",
        "float64",
    ]


def band_with_nodata_mask(
    raster_dataset: rasterio.io.DatasetReader, band: int
) -> np.array:
    raw_data = raster_dataset.read(band)
    nodata_value = band_nodata_value(raster_dataset, band)
    if math.isnan(nodata_value):
        return (raw_data, np.isnan(raw_data))
    else:
        return (raw_data, raw_data == nodata_value)


def quantile_ranges() -> List[List[float]]:
    """Return a list of ranges to compute quantiles."""
    return [[j / i for j in range(1, i)] for i in DEFAULT_OVERVIEWS]


def sample_not_masked_values(
    raster_dataset: rasterio.io.DatasetReader, n_samples: int
) -> Samples:
    """Compute quantiles for a raster dataset band."""

    def not_enough_samples():
        return (
            len(not_masked_samples[1]) < n_samples
            and iterations < DEFAULT_SAMPLING_MAX_ITERATIONS
        )

    west = raster_dataset.bounds.left
    south = raster_dataset.bounds.bottom
    east = raster_dataset.bounds.right
    north = raster_dataset.bounds.top

    bands = range(1, raster_dataset.count + 1)

    not_masked_samples = {b: [] for b in bands}

    iterations = 0

    print("Sampling raster...")
    rng = np.random.default_rng()
    while not_enough_samples():
        x = rng.uniform(west, east, n_samples)
        y = rng.uniform(south, north, n_samples)

        try:
            from rasterio.sample import sort_xy
        except ImportError:
            coords = zip(x, y)
        else:
            coords = sort_xy(zip(x, y))

        samples = raster_dataset.sample(coords, indexes=bands, masked=True)
        for sample in samples:
            raster_is_masked = (
                sample.mask if isinstance(sample.mask, np.bool_) else sample.mask.any()
            )
            if not raster_is_masked:
                for band in bands:
                    not_masked_samples[band].append(sample[band - 1])

        iterations += 1

    if len(not_masked_samples[1]) < n_samples:
        if len(not_masked_samples[1]) == 0:
            raise ValueError(
                "The data is very sparse and no non-masked samples were collected.\n"
                "Please, consider to use the --exact_stats option to compute exact "
                "stats"
            )

        warnings.warn(
            "The data is very sparse and there are not enough non-masked samples.\n"
            f"Only {len(not_masked_samples[1])} samples were collected and "
            "quantiles and most common values may be inaccurate.",
            UserWarning,
        )

    for b in bands:
        not_masked_samples[b] = not_masked_samples[b][:n_samples]

    return not_masked_samples


def most_common_approx(samples: List[Union[int, float]]) -> Dict[int, int]:
    """Compute the most common values in a list of int samples."""
    counts = np.bincount(samples)
    nth = min(DEFAULT_MAX_MOST_COMMON, len(counts))
    idx = np.argpartition(counts, -nth)[-nth:]
    return dict([(int(i), int(counts[i])) for i in idx if counts[i] > 0])


def compute_quantiles(data: List[Union[int, float]], cast_function: Callable) -> dict:
    """Compute quantiles for a raster dataset band."""
    print("Computing quantiles...")
    quantiles = [
        [cast_function(np.quantile(data, q, method="lower")) for q in r]
        for r in quantile_ranges()
    ]
    return dict(zip(DEFAULT_OVERVIEWS, quantiles))


def get_stats(
    raster_dataset: rasterio.io.DatasetReader, band: int
) -> rasterio.Statistics:
    """Get statistics for a raster band."""
    try:
        # stats method is supported since rasterio 1.4.0 and statistics
        # method will be deprecated in future versions of rasterio
        return raster_dataset.stats(indexes=[band], approx=True)[0]
    except AttributeError:
        return raster_dataset.statistics(band, approx=True)


def raster_band_approx_stats(
    raster_dataset: rasterio.io.DatasetReader,
    samples: Samples,
    band: int,
    basic_stats: bool,
) -> dict:
    """Get approximate statistics for a raster band."""

    stats = get_stats(raster_dataset, band)

    samples_band = samples[band]

    count = len(samples_band)
    _sum = 0
    sum_squares = 0
    if count > 0:
        _sum = int(np.sum(samples_band))
        sum_squares = int(np.sum(np.array(samples_band) ** 2))

    if basic_stats:
        quantiles = None
        most_common = None
    else:
        quantiles = compute_quantiles(samples_band, int)

        most_common = dict()
        if not band_is_float(raster_dataset, band):
            most_common = most_common_approx(samples_band)

    return {
        "min": stats.min,
        "max": stats.max,
        "mean": stats.mean,
        "stddev": stats.std,
        "sum": _sum,
        "sum_squares": sum_squares,
        "count": count,
        "quantiles": quantiles,
        "top_values": most_common,
        "version": ".".join(__version__.split(".")[:3]),
        "approximated_stats": True,
    }


def is_masked_band(raster_dataset: rasterio.io.DatasetReader, band: int) -> bool:
    """Check if a band is masked."""
    alpha_band = get_alpha_band(raster_dataset)
    original_nodata_value = band_original_nodata_value(raster_dataset, band)
    return band == alpha_band or (
        alpha_band is None
        and original_nodata_value is None
        and not band_is_float(raster_dataset, band)
    )


def read_raster_band(raster_dataset: rasterio.io.DatasetReader, band: int) -> np.array:
    band_is_masked = is_masked_band(raster_dataset, band)
    if band_is_masked:
        unmasked_data = raster_dataset.read(band)
        return np.ma.masked_array(data=unmasked_data, mask=False)

    alpha_band = get_alpha_band(raster_dataset)
    if alpha_band:
        # mask data with alpha band to exclude from stats
        return read_masked(raster_dataset, band, alpha_band)

    # mask nodata values to exclude from stats
    compound_bands = get_compound_bands(raster_dataset, band)
    if len(compound_bands) > 1:
        # if band is part of a RGB triplet,
        # we need to use the three bands for masking
        bands_and_masks = [
            band_with_nodata_mask(raster_dataset, b) for b in compound_bands
        ]
        mask = np.logical_and.reduce(
            [band_and_mask[1] for band_and_mask in bands_and_masks]
        )
        raw_data = bands_and_masks[compound_bands.index(band)][0]
    else:
        (raw_data, mask) = band_with_nodata_mask(raster_dataset, band)

    return np.ma.masked_array(data=raw_data, mask=mask)


def raster_band_stats(
    raster_dataset: rasterio.io.DatasetReader, band: int, basic_stats: bool
) -> dict:
    """Get statistics for a raster band."""

    print("Computing stats for band {0}...".format(band))

    _stats = get_stats(raster_dataset, band)
    _min = _stats.min
    _max = _stats.max
    _mean = _stats.mean
    _std = _stats.std

    raster_band = read_raster_band(raster_dataset=raster_dataset, band=band)

    count = math.prod(raster_band.shape)
    if is_masked_band(raster_dataset, band):
        count = np.count_nonzero(raster_band.mask is False)

    _sum = _mean * count
    sum_squares = count * _std**2 + _mean**2

    print("Removing masked data...")
    qdata = raster_band.compressed()

    if basic_stats:
        quantiles = None
        most_common = None
    else:
        casting_function = (
            int if np.issubdtype(raster_band.dtype, np.integer) else float
        )

        quantiles = compute_quantiles(qdata, casting_function)

        if casting_function == int:
            print("Computing most commons values...")
            most_common = Counter(qdata).most_common(100)
            most_common.sort(key=lambda x: x[1], reverse=True)
            most_common = dict([(casting_function(x[0]), x[1]) for x in most_common])

    version = ".".join(__version__.split(".")[:3])

    return {
        "min": float(_min),
        "max": float(_max),
        "mean": float(_mean),
        "stddev": float(_std),
        "sum": _sum,
        "sum_squares": sum_squares,
        "count": count,
        "quantiles": quantiles,
        "top_values": most_common,
        "version": version,
        "approximated_stats": False,
    }


def get_number_of_overviews_blocks(file_path: str) -> int:

    raster_info = rio_cogeo.cog_info(file_path).dict()
    with rasterio.open(file_path) as raster_dataset:
        overview_factors = raster_dataset.overviews(1)
        block_width, block_height, resolution = get_resolution_and_block_sizes(
            raster_dataset, raster_info
        )
        raster_crs = raster_dataset.crs.to_string()
        raster_to_4326_transformer = pyproj.Transformer.from_crs(
            raster_crs, "EPSG:4326", always_xy=True
        )
        pixels_to_raster_transform = raster_dataset.transform

        # results are crs 4326, so x = long, y = lat
        min_base_tile_lng, min_base_tile_lat = raster_to_4326_transformer.transform(
            *(pixels_to_raster_transform * (block_width * 0.5, block_height * 0.5))
        )
        max_base_tile_lng, max_base_tile_lat = raster_to_4326_transformer.transform(
            *(
                pixels_to_raster_transform
                * (
                    raster_dataset.width - block_width * 0.5,
                    raster_dataset.height - block_height * 0.5,
                )
            )
        )

        # quadbin cell at base resolution
        min_base_tile = quadbin.point_to_cell(
            min_base_tile_lng, min_base_tile_lat, resolution
        )
        min_base_x, min_base_y, _z = quadbin.cell_to_tile(min_base_tile)

        n_records = 0
        for overview_index in range(0, len(overview_factors)):
            # quadbin cell at overview resolution (quadbin_tile -> quadbin_cell)
            min_tile = quadbin.point_to_cell(
                min_base_tile_lng, min_base_tile_lat, resolution - overview_index - 1
            )
            max_tile = quadbin.point_to_cell(
                max_base_tile_lng, max_base_tile_lat, resolution - overview_index - 1
            )
            min_x, min_y, min_z = quadbin.cell_to_tile(min_tile)
            max_x, max_y, _z = quadbin.cell_to_tile(max_tile)

            n_records += (max_x - min_x + 1) * (max_y - min_y + 1)

    return n_records


def rasterio_overview_to_records(
    #  raster_dataset: rasterio.io.DatasetReader,
    file_path: str,
    band_rename_function: Callable,
    bands_info: List[Tuple[int, str]],
    compress: bool = False,
) -> Iterable:
    raster_info = rio_cogeo.cog_info(file_path).dict()
    with rasterio.open(file_path) as raster_dataset:
        block_width, block_height, resolution = get_resolution_and_block_sizes(
            raster_dataset, raster_info
        )
        raster_crs = raster_dataset.crs.to_string()

        raster_to_4326_transformer = pyproj.Transformer.from_crs(
            raster_crs, "EPSG:4326", always_xy=True
        )
        # raster_crs must be 3857
        pixels_to_raster_transform = raster_dataset.transform
        is_valid_raster_dataset(raster_dataset)

        block_width, block_height, resolution = get_resolution_and_block_sizes(
            raster_dataset, raster_info
        )
        raster_crs = raster_dataset.crs.to_string()

        raster_to_4326_transformer = pyproj.Transformer.from_crs(
            raster_crs, "EPSG:4326", always_xy=True
        )
        # raster_crs must be 3857
        pixels_to_raster_transform = raster_dataset.transform

        overview_factors = raster_dataset.overviews(1)
        (block_width, block_height) = raster_dataset.block_shapes[0]

        for overview_index in range(0, len(overview_factors)):
            # results are crs 4326, so x = long, y = lat
            min_base_tile_lng, min_base_tile_lat = raster_to_4326_transformer.transform(
                *(pixels_to_raster_transform * (block_width * 0.5, block_height * 0.5))
            )
            max_base_tile_lng, max_base_tile_lat = raster_to_4326_transformer.transform(
                *(
                    pixels_to_raster_transform
                    * (
                        raster_dataset.width - block_width * 0.5,
                        raster_dataset.height - block_height * 0.5,
                    )
                )
            )
            # quadbin cell at base resolution
            min_base_tile = quadbin.point_to_cell(
                min_base_tile_lng, min_base_tile_lat, resolution
            )
            min_base_x, min_base_y, _z = quadbin.cell_to_tile(min_base_tile)

            # quadbin cell at overview resolution (quadbin_tile -> quadbin_cell)
            min_tile = quadbin.point_to_cell(
                min_base_tile_lng, min_base_tile_lat, resolution - overview_index - 1
            )
            max_tile = quadbin.point_to_cell(
                max_base_tile_lng, max_base_tile_lat, resolution - overview_index - 1
            )
            min_x, min_y, min_z = quadbin.cell_to_tile(min_tile)
            max_x, max_y, _z = quadbin.cell_to_tile(max_tile)

            for tile_x in range(min_x, max_x + 1):
                for tile_y in range(min_y, max_y + 1):
                    children = quadbin.cell_to_children(
                        quadbin.tile_to_cell((tile_x, tile_y, min_z)), resolution
                    )
                    # children x,y,z tuples (tiles)
                    children_tiles = [quadbin.cell_to_tile(child) for child in children]
                    child_xs = [child[0] for child in children_tiles]
                    child_ys = [child[1] for child in children_tiles]
                    min_child_x, max_child_x = min(child_xs), max(child_xs)
                    min_child_y, max_child_y = min(child_ys), max(child_ys)
                    factor = overview_factors[overview_index]

                    # tile_window for current overview
                    tile_window = rasterio.windows.Window(
                        col_off=block_width * (min_child_x - min_base_x),
                        row_off=block_height * (min_child_y - min_base_y),
                        width=(max_child_x - min_child_x + 1)
                        * block_width,  # should equal block_width * factor
                        height=(max_child_y - min_child_y + 1)
                        * block_height,  # should equal block_width * factor
                    )

                    # So far we only support one nodata value for all bands
                    no_data_value = get_nodata_value(raster_dataset)
                    record = {}
                    for band, band_name in bands_info:
                        tile_data = read_filled(
                            raster_dataset,
                            band,
                            no_data_value,
                            window=tile_window,
                            out_shape=(
                                tile_window.width // factor,
                                tile_window.height // factor,
                            ),
                            boundless=True,
                        )
                        newrecord = array_to_record(
                            tile_data,
                            band_field_name(band_name, band, band_rename_function),
                            band_rename_function,
                            raster_to_4326_transformer,
                            pixels_to_raster_transform,
                            resolution - overview_index - 1,
                            tile_window,
                            no_data_value,
                            compress=compress,
                        )
                        if newrecord:
                            record.update(newrecord)

                    if record:
                        yield record


def rasterio_windows_to_records(
    file_path: str,
    band_rename_function: Callable,
    bands_info: List[Tuple[int, str]],
    compress: bool = False,
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
        error_not_google_compatible()

    """Open a raster file with rasterio."""
    with rasterio.open(file_path) as raster_dataset:
        block_width, block_height, resolution = get_resolution_and_block_sizes(
            raster_dataset, raster_info
        )
        raster_crs = raster_dataset.crs.to_string()

        raster_to_4326_transformer = pyproj.Transformer.from_crs(
            raster_crs, "EPSG:4326", always_xy=True
        )
        # raster_crs must be 3857
        pixels_to_raster_transform = raster_dataset.transform

        # Base raster
        for _, window in raster_dataset.block_windows():
            record = {}
            no_data_value = get_nodata_value(raster_dataset)
            for band, band_name in bands_info:
                tile_data = read_filled(
                    raster_dataset, band, no_data_value, window=window, boundless=True
                )
                newrecord = array_to_record(
                    tile_data,
                    band_field_name(band_name, band, band_rename_function),
                    band_rename_function,
                    raster_to_4326_transformer,
                    pixels_to_raster_transform,
                    resolution,
                    window,
                    no_data_value,
                    compress=compress,
                )

                # add the new columns generated by array_t
                # o_record but leaving unchanged the index e.g. the block column
                if newrecord:
                    record.update(newrecord)

            if record:
                yield record


def is_valid_overview_indexes(overview_factors) -> bool:
    for overview_index in range(0, len(overview_factors)):
        if overview_factors[overview_index] != pow(2, overview_index + 1):
            return False
    return True


def is_valid_block_shapes(block_shapes) -> bool:
    # Block size must be equal for all bands;
    # We avoid looping here over bands because we need
    # to loop internally to accumulate, for each block
    # the data for all bands.
    (block_width, block_height) = block_shapes[0]
    for block_shape_index in range(0, len(block_shapes)):
        (index_block_width, index_block_height) = block_shapes[block_shape_index]
        if (block_width != index_block_width) or (block_height != index_block_height):
            return False
    return True


def is_valid_raster_dataset(raster_dataset: rasterio.io.DatasetReader) -> bool:

    if not is_valid_block_shapes(raster_dataset.block_shapes):
        raise ValueError("Invalid block shapes: must be equal for all bands")

    if not is_valid_overview_indexes(raster_dataset.overviews(1)):
        raise ValueError("Invalid overview factors: must be consecutive powers of 2")

    return True


def band_without_stats(band):
    return {
        k: band[k]
        for k in set(list(band.keys())) - set(["stats", "colorinterp", "colortable"])
    }


def bands_without_stats(metadata):
    return [band_without_stats(band) for band in metadata["bands"]]


def check_metadata_is_compatible(metadata, old_metadata):
    """Check that the metadata of a raster file is compatible with the metadata of a
    table.

    Parameters
    ----------
    metadata : dict
        Metadata of the raster file.
    old_metadata : dict
        Metadata of the table.

    Raises
    ------
    ValueError
        If the metadata is not compatible.
    """
    if metadata["block_resolution"] != old_metadata["block_resolution"]:
        raise ValueError(
            "Cannot append records to a table with a different block_resolution "
            f"({metadata['block_resolution']} != {old_metadata['block_resolution']})."
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

    if bands_without_stats(metadata) != bands_without_stats(old_metadata):
        raise ValueError(
            "Cannot append records to a table with different bands."
            f"({metadata['bands']} != {old_metadata['bands']})."
        )


def update_metadata(metadata, old_metadata):
    """Update a metadata object, combining it with another existing metadata object

    Parameters
    ----------
    metadata : dict
        Metadata to update (taken from a raster file to import).
        This dictionary will be modified in place.
    old_metadata : dict
        Metadata to combine with (taken from an existing BigQuery table).
    """

    metadata["bounds"] = (
        min(old_metadata["bounds"][0], metadata["bounds"][0]),
        min(old_metadata["bounds"][1], metadata["bounds"][1]),
        max(old_metadata["bounds"][2], metadata["bounds"][2]),
        max(old_metadata["bounds"][3], metadata["bounds"][3]),
    )
    metadata["center"] = (
        (metadata["bounds"][0] + metadata["bounds"][2]) / 2,
        (metadata["bounds"][1] + metadata["bounds"][3]) / 2,
        metadata["block_resolution"],
    )
    metadata["num_blocks"] += old_metadata["num_blocks"]
    metadata["num_pixels"] += old_metadata["num_pixels"]
    w, s, _ = quadbin.utils.point_to_tile(
        metadata["bounds"][0], metadata["bounds"][1], metadata["block_resolution"]
    )
    e, n, _ = quadbin.utils.point_to_tile(
        metadata["bounds"][2], metadata["bounds"][3], metadata["block_resolution"]
    )
    metadata["height"] = (s - n) * metadata["block_height"]
    metadata["width"] = (e - w) * metadata["block_width"]

    for band in metadata["bands"]:
        old_band = next(
            (
                old_band
                for old_band in old_metadata["bands"]
                if old_band["name"] == band["name"]
            ),
            None,
        )

        if old_band is None:
            # Extra precaution as this should never happen
            raise ValueError(
                "Cannot append records to a table with different bands"
                f"(band {old_band['name']} not found)."
            )

        new_stats = band["stats"]
        old_stats = old_band["stats"]

        _min = min(old_stats["min"], new_stats["min"])
        _max = max(old_stats["max"], new_stats["max"])
        _sum = old_stats["sum"] + new_stats["sum"]
        sum_squares = old_stats["sum_squares"] + new_stats["sum_squares"]
        count = old_stats["count"] + new_stats["count"]

        if old_stats["count"] == 0 or new_stats["count"] == 0:
            mean = (old_stats["mean"] + new_stats["mean"]) / 2
            stdev = math.sqrt(old_stats["stddev"] ** 2 + new_stats["stddev"] ** 2)
        else:
            mean = _sum / count
            try:
                stdev = math.sqrt(sum_squares / count - mean * mean)
            except ValueError:
                stdev = math.sqrt(old_stats["stddev"] ** 2 + new_stats["stddev"] ** 2)

        approximated_stats = (
            old_stats["approximated_stats"] or new_stats["approximated_stats"]
        )

        try:
            top_values = set(new_stats["top_values"] + old_stats["top_values"])
            top_values = list(top_values).sort(reverse=True)[:DEFAULT_MAX_MOST_COMMON]
        except TypeError:
            # If top values in any of either metadata is None,
            # the above will raise a TypeError
            top_values = None

        version = max(old_stats["version"], new_stats["version"])

        band["stats"] = {
            "min": _min,
            "max": _max,
            "sum": _sum,
            "sum_squares": sum_squares,
            "count": count,
            "mean": mean,
            "stddev": stdev,
            "quantiles": None,
            "top_values": top_values,
            "version": version,
            "approximated_stats": approximated_stats,
        }


def get_number_of_blocks(file_path: str) -> int:
    """Get the number of blocks in a raster file."""

    with rasterio.open(file_path) as raster_dataset:
        return len(list(raster_dataset.block_windows()))


def size_mb_of_rasterio_band(file_path: str, band: int = 1) -> int:
    """Get the size in MB of a rasterio band."""

    with rasterio.open(file_path) as raster_dataset:
        W = raster_dataset.width
        H = raster_dataset.height
        S = np.dtype(raster_dataset.dtypes[band - 1]).itemsize
        return (W * H * S) / 1024 / 1024


def print_band_information(file_path: str):
    """Print out information about the bands in a raster file."""

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

    with rasterio.open(file_path) as raster_dataset:
        return raster_dataset.block_shapes[0]
