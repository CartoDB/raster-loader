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

try:
    import rio_cogeo
except ImportError:  # pragma: no cover
    _has_rio_cogeo = False
else:
    _has_rio_cogeo = True

import rasterio
import quadbin

from raster_loader.geo import raster_bounds
from raster_loader.errors import (
    import_error_rio_cogeo,
    error_not_google_compatible,
)

should_swap = {"=": sys.byteorder != "little", "<": False, ">": True, "|": False}


def band_field_name(custom_name: str, band: int, band_rename_function: Callable) -> str:
    return band_rename_function(custom_name or "band_" + str(band))


def array_to_record(
    arr: np.ndarray,
    value_field: str,
    band_rename_function: Callable,
    transformer: pyproj.Transformer,
    geotransform: Affine,
    resolution: int,
    row_off: int = 0,
    col_off: int = 0,
) -> dict:
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


def rasterio_metadata(
    file_path: str,
    bands_info: List[Tuple[int, str]],
    band_rename_function: Callable,
):
    """Requires rio_cogeo."""
    if not _has_rio_cogeo:  # pragma: no cover
        import_error_rio_cogeo()

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
    resolution = raster_info["GEO"]["MaxZoom"]
    metadata["block_resolution"] = resolution
    metadata["minresolution"] = resolution
    metadata["maxresolution"] = resolution
    metadata["nodata"] = raster_info["Profile"]["Nodata"]
    if metadata["nodata"] is not None and math.isnan(metadata["nodata"]):
        metadata["nodata"] = None

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
                "band_name": band_field_name(band_name, band, band_rename_function),
            }
            bands_metadata.append(meta)

        # compute whole bounds for metadata
        bounds_geog = raster_bounds(raster_dataset, transformer, "wkt")
        bounds_polygon = shapely.Polygon(shapely.wkt.loads(bounds_geog))
        bounds_coords = list(bounds_polygon.bounds)
        center_coords = list(*bounds_polygon.centroid.coords)
        center_coords.append(resolution)
        # assuming all windows have the same dimensions
        a_window = next(raster_dataset.block_windows())
        block_width = a_window[1].width
        block_height = a_window[1].height

        pixel_resolution = int(resolution + math.log(block_width * block_height, 4))
        if pixel_resolution > 26:
            raise ValueError(
                "Input raster pixel resolution exceeds "
                "the max supported resolution of 26.\n"
                "Please resample the raster to a lower resolution."
            )

        metadata["bands"] = [
            {"type": e["type"], "name": e["band_name"]} for e in bands_metadata
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


def rasterio_windows_to_records(
    file_path: str,
    band_rename_function: Callable,
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
        error_not_google_compatible()

    resolution = raster_info["GEO"]["MaxZoom"]

    """Open a raster file with rasterio."""
    with rasterio.open(file_path) as raster_dataset:
        raster_crs = raster_dataset.crs.to_string()

        transformer = pyproj.Transformer.from_crs(
            raster_crs, "EPSG:4326", always_xy=True
        )

        for _, window in raster_dataset.block_windows():
            record = {}
            for band, band_name in bands_info:
                newrecord = array_to_record(
                    raster_dataset.read(band, window=window),
                    band_field_name(band_name, band, band_rename_function),
                    band_rename_function,
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
    w, s, _ = quadbin.utils.point_to_tile(
        metadata["bounds"][0], metadata["bounds"][1], metadata["block_resolution"]
    )
    e, n, _ = quadbin.utils.point_to_tile(
        metadata["bounds"][2], metadata["bounds"][3], metadata["block_resolution"]
    )
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
