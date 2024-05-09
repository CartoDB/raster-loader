import sys
import math
import pyproj
import shapely
import numpy as np

from typing import Iterable
from typing import Callable
from typing import List
from typing import Tuple
from affine import Affine
from shapely import wkt  # Can not use directly from shapely.wkt

import rio_cogeo
import rasterio
import quadbin

from raster_loader.geo import raster_bounds
from raster_loader.errors import (
    error_not_google_compatible,
)

DEFAULT_COG_BLOCK_SIZE = 256

DEFAULT_TYPES_NODATA_VALUES = {
    "int8": -128,
    "int16": -32768,
    "int32": -2147483648,
    "int64": -9223372036854775808,
    "uint8": 255,
    "uint16": 65535,
    "uint32": 4294967295,
    "uint64": 18446744073709551615,
    "float16": np.NaN,
    "float32": np.NaN,
    "float64": np.NaN,
}

should_swap = {"=": sys.byteorder != "little", "<": False, ">": True, "|": False}


def band_field_name(custom_name: str, band: int, band_rename_function: Callable) -> str:
    return band_rename_function(custom_name or "band_" + str(band))


def get_nodata_value(raster_dataset: rasterio.io.DatasetReader, bands_info: List[Tuple[int, str]]) -> float:
    value = band_nodata_value(raster_dataset, 1)
    # So far we only support one nodata value for all bands
    if raster_dataset.nodata is None:
        for band, band_name in bands_info:
            band_value = band_nodata_value(raster_dataset, band)
            # Note (np.NaN != np.NaN) == True
            both_nan = np.isnan(band_value) and np.isnan(value)
            if (not both_nan) and (band_value != value):
                raise ValueError("Invalid no data value")
    return value


def band_nodata_value(raster_dataset: rasterio.io.DatasetReader, band: int) -> float:
    nodata_value = raster_dataset.nodata
    if nodata_value is None:
        nodata_value = raster_dataset.get_nodatavals()[band - 1]
    if nodata_value is None:
        nodata_value = get_default_nodata_value(raster_dataset.dtypes[band - 1])
    return nodata_value


def get_default_nodata_value(dtype: str) -> float:
    if dtype in DEFAULT_TYPES_NODATA_VALUES:
        return DEFAULT_TYPES_NODATA_VALUES[dtype]
    else:
        raise ValueError(f"Unsupported data type: {dtype}")


def array_to_record(
    arr: np.ndarray,
    value_field: str,
    band_rename_function: Callable,
    transformer: pyproj.Transformer,
    geotransform: Affine,
    resolution: int,
    window: rasterio.windows.Window,
) -> dict:
    row_off = window.row_off
    col_off = window.col_off
    width = window.width
    height = window.height

    x, y = transformer.transform(*(geotransform * (col_off + width * 0.5, row_off + height * 0.5)))

    block = quadbin.point_to_cell(x, y, resolution)

    if should_swap[arr.dtype.byteorder]:
        arr_bytes = np.ascontiguousarray(arr.byteswap()).tobytes()
    else:
        arr_bytes = np.ascontiguousarray(arr).tobytes()

    record = {
        band_rename_function("block"): block,
        band_rename_function("metadata"): None,
        value_field: arr_bytes,
    }

    return record


def raster_band_type(raster_dataset: rasterio.io.DatasetReader, band: int) -> str:
    types = {i: dtype for i, dtype in zip(raster_dataset.indexes, raster_dataset.dtypes)}
    return str(types[band])


def get_resolution_and_block_sizes(raster_dataset: rasterio.io.DatasetReader, raster_info: dict):
    # assuming all windows have the same dimensions
    a_window = next(raster_dataset.block_windows())
    block_width = a_window[1].width
    block_height = a_window[1].height
    resolution = int(
        raster_info["GEO"]["MaxZoom"]
        - math.log(
            block_width / DEFAULT_COG_BLOCK_SIZE * block_height / DEFAULT_COG_BLOCK_SIZE,
            4,
        )
    )
    return block_width, block_height, resolution


def rasterio_metadata(
    file_path: str,
    bands_info: List[Tuple[int, str]],
    band_rename_function: Callable,
):
    """Open a raster file with rasterio."""
    raster_info = rio_cogeo.cog_info(file_path).dict()

    """Check if raster is compatible."""
    if "GoogleMapsCompatible" != raster_info.get("Tags", {}).get("Tiling Scheme", {}).get("NAME"):
        error_not_google_compatible()

    metadata = {}
    width = raster_info["Profile"]["Width"]
    height = raster_info["Profile"]["Height"]

    with rasterio.open(file_path) as raster_dataset:
        raster_crs = raster_dataset.crs.to_string()

        transformer = pyproj.Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)

        block_width, block_height, resolution = get_resolution_and_block_sizes(raster_dataset, raster_info)

        metadata["block_resolution"] = resolution
        metadata["minresolution"] = resolution - len(raster_dataset.overviews(1))
        metadata["maxresolution"] = resolution
        metadata["nodata"] = raster_info["Profile"]["Nodata"]
        if metadata["nodata"] is not None and math.isnan(metadata["nodata"]):
            metadata["nodata"] = None
        bands_metadata = []
        for band, band_name in bands_info:
            meta = {
                "band": band,
                "type": raster_band_type(raster_dataset, band),
                "band_name": band_field_name(band_name, band, band_rename_function),
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

        metadata["bands"] = [{"type": e["type"], "name": e["band_name"]} for e in bands_metadata]
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


def rasterio_windows_to_records(
    file_path: str,
    band_rename_function: Callable,
    bands_info: List[Tuple[int, str]],
) -> Iterable:
    invalid_names = [name for _, name in bands_info if name and name.lower() in ["block", "metadata"]]

    if invalid_names:
        raise ValueError(f"Invalid band names: {', '.join(invalid_names)}")

    """Open a raster file with rio-cogeo."""
    raster_info = rio_cogeo.cog_info(file_path).dict()

    """Check if raster is compatible."""
    if "GoogleMapsCompatible" != raster_info.get("Tags", {}).get("Tiling Scheme", {}).get("NAME"):
        error_not_google_compatible()

    """Open a raster file with rasterio."""
    with rasterio.open(file_path) as raster_dataset:
        block_width, block_height, resolution = get_resolution_and_block_sizes(raster_dataset, raster_info)
        raster_crs = raster_dataset.crs.to_string()

        raster_to_4326_transformer = pyproj.Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
        # raster_crs must be 3857
        pixels_to_raster_transform = raster_dataset.transform

        # Base raster
        for _, window in raster_dataset.block_windows():
            record = {}
            no_data_value = get_nodata_value(raster_dataset, bands_info)
            for band, band_name in bands_info:
                tile_data = raster_dataset.read(
                    band, window=window, boundless=True, masked=True, fill_value=no_data_value
                )
                newrecord = array_to_record(
                    tile_data.filled(fill_value=no_data_value),
                    band_field_name(band_name, band, band_rename_function),
                    band_rename_function,
                    raster_to_4326_transformer,
                    pixels_to_raster_transform,
                    resolution,
                    window,
                )

                # add the new columns generated by array_t
                # o_record
                # but leaving unchanged the index e.g. the block column
                record.update(newrecord)

            yield record

        # Overviews

        # Block size must be equal for all bands;
        # We avoid looping here over bands because we need
        # to loop internally to accumulate, for each block
        # the data for all bands.
        if not is_valid_block_shapes(raster_dataset.block_shapes):
            raise ValueError("Invalid block shapes: must be equal for all bands")

        (block_width, block_height) = raster_dataset.block_shapes[0]
        overview_factors = raster_dataset.overviews(1)

        if not is_valid_overview_indexes(overview_factors):
            raise ValueError("Invalid overview factors: must be consecutive powers of 2")

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
            min_base_tile = quadbin.point_to_cell(min_base_tile_lng, min_base_tile_lat, resolution)
            min_base_x, min_base_y, _z = quadbin.cell_to_tile(min_base_tile)

            # quadbin cell at overview resolution (quadbin_tile -> quadbin_cell)
            min_tile = quadbin.point_to_cell(min_base_tile_lng, min_base_tile_lat, resolution - overview_index - 1)
            max_tile = quadbin.point_to_cell(max_base_tile_lng, max_base_tile_lat, resolution - overview_index - 1)
            min_x, min_y, min_z = quadbin.cell_to_tile(min_tile)
            max_x, max_y, _z = quadbin.cell_to_tile(max_tile)
            for tile_x in range(min_x, max_x + 1):
                for tile_y in range(min_y, max_y + 1):
                    children = quadbin.cell_to_children(quadbin.tile_to_cell((tile_x, tile_y, min_z)), resolution)
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
                        width=(max_child_x - min_child_x + 1) * block_width,  # should equal block_width * factor
                        height=(max_child_y - min_child_y + 1) * block_height,  # should equal block_width * factor
                    )

                    # So far we only support one nodata value for all bands
                    no_data_value = get_nodata_value(raster_dataset, bands_info)
                    record = {}
                    for band, band_name in bands_info:
                        tile_data = raster_dataset.read(
                            band,
                            window=tile_window,
                            out_shape=(tile_window.width // factor, tile_window.height // factor),
                            boundless=True,
                            masked=True,
                            fill_value=no_data_value,
                        )
                        newrecord = array_to_record(
                            tile_data.filled(fill_value=no_data_value),
                            band_field_name(band_name, band, band_rename_function),
                            band_rename_function,
                            raster_to_4326_transformer,
                            pixels_to_raster_transform,
                            resolution - overview_index - 1,
                            tile_window,
                        )
                        record.update(newrecord)

                    yield record


def is_valid_overview_indexes(overview_factors) -> bool:
    for overview_index in range(0, len(overview_factors)):
        if overview_factors[overview_index] != pow(2, overview_index + 1):
            return False
    return True


def is_valid_block_shapes(block_shapes) -> bool:
    (block_width, block_height) = block_shapes[0]
    for block_shape_index in range(0, len(block_shapes)):
        (index_block_width, index_block_height) = block_shapes[block_shape_index]
        if (block_width != index_block_width) or (block_height != index_block_height):
            return False
    return True


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
        raise ValueError("Cannot append records to a table with a different block width/height.")
    if metadata["bands"] != old_metadata["bands"]:
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
    w, s, _ = quadbin.utils.point_to_tile(metadata["bounds"][0], metadata["bounds"][1], metadata["block_resolution"])
    e, n, _ = quadbin.utils.point_to_tile(metadata["bounds"][2], metadata["bounds"][3], metadata["block_resolution"])
    metadata["height"] = (s - n) * metadata["block_height"]
    metadata["width"] = (e - w) * metadata["block_width"]


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
                [size_mb_of_rasterio_band(file_path, band + 1) for band in range(raster_dataset.count)]
            )
        )


def get_block_dims(file_path: str) -> tuple:
    """Get the dimensions of a raster file's blocks."""

    with rasterio.open(file_path) as raster_dataset:
        return raster_dataset.block_shapes[0]
