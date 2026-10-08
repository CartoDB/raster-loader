import os
from itertools import chain
from unittest.mock import patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

import raster_loader.io.common as common

from raster_loader.io.common import (
    is_window_inside,
    rasterio_overview_to_records,
    rasterio_windows_to_records,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(HERE, "fixtures")
MOSAIC_COG = os.path.join(FIXTURES_DIR, "mosaic_cog.tif")


class FakeDataset:
    width = 512
    height = 256


def test_is_window_inside():
    dataset = FakeDataset()

    assert is_window_inside(Window(0, 0, 256, 256), dataset)
    assert is_window_inside(Window(256, 0, 256, 256), dataset)
    assert not is_window_inside(Window(384, 0, 256, 256), dataset)
    assert not is_window_inside(Window(0, 128, 256, 256), dataset)
    assert not is_window_inside(Window(-1, 0, 256, 256), dataset)


def _records(file_path):
    def band_rename_function(x):
        return x.upper()

    bands_info = [(1, "band_1")]

    return list(
        chain(
            rasterio_overview_to_records(
                file_path, band_rename_function, bands_info, compress=True
            ),
            rasterio_windows_to_records(
                file_path, band_rename_function, bands_info, compress=True
            ),
        )
    )


@pytest.fixture(scope="module", params=["nearest", "average"])
def web_optimized_cog(request, tmp_path_factory):
    """A GoogleMapsCompatible COG large enough to have overview tile windows
    inside the raster.

    In ``mosaic_cog.tif`` every overview tile window overruns the raster, so
    only a larger raster exercises the direct overview read with ``out_shape``.
    """
    cog_path = str(tmp_path_factory.mktemp(f"cog_{request.param}") / "cog.tif")

    size = 1000
    data = np.random.default_rng(0).integers(0, 1000, (size, size), dtype="int16")
    # rasterio.shutil.copy upper-cases option values, and raster-loader expects
    # the tiling scheme spelled "GoogleMapsCompatible".
    with rasterio.open(
        cog_path,
        "w",
        driver="COG",
        width=size,
        height=size,
        count=1,
        dtype="int16",
        crs="EPSG:3857",
        transform=from_origin(-1_000_000, 5_000_000, 100, 100),
        nodata=-1,
        TILING_SCHEME="GoogleMapsCompatible",
        BLOCKSIZE=256,
        COMPRESS="DEFLATE",
        RESAMPLING=request.param,
        ADD_ALPHA="NO",
    ) as dataset:
        dataset.write(data, 1)

    return cog_path


def _read_filled_calls(file_path):
    calls = []
    read_filled = common.read_filled

    def spy(raster_dataset, band, no_data_value, **args):
        calls.append(
            {
                "overview": "out_shape" in args,
                "boundless": args["boundless"],
                "inside": is_window_inside(args["window"], raster_dataset),
            }
        )
        return read_filled(raster_dataset, band, no_data_value, **args)

    with patch("raster_loader.io.common.read_filled", spy):
        _records(file_path)

    return calls


def _assert_same_records_as_boundless_reads(file_path):
    records = _records(file_path)
    with patch("raster_loader.io.common.is_window_inside", return_value=False):
        boundless_records = _records(file_path)

    assert records
    assert records == boundless_records


def test_direct_reads_produce_the_same_records_as_boundless_reads():
    _assert_same_records_as_boundless_reads(MOSAIC_COG)


def test_direct_overview_reads_produce_the_same_records_as_boundless_reads(
    web_optimized_cog,
):
    calls = _read_filled_calls(web_optimized_cog)
    overview_calls = [call for call in calls if call["overview"]]

    assert any(not call["boundless"] for call in overview_calls)
    assert any(call["boundless"] for call in overview_calls)
    assert all(call["boundless"] == (not call["inside"]) for call in calls)

    _assert_same_records_as_boundless_reads(web_optimized_cog)


def test_boundless_is_used_only_for_windows_overrunning_the_raster():
    calls = _read_filled_calls(MOSAIC_COG)

    assert any(call["boundless"] for call in calls)
    assert any(not call["boundless"] for call in calls)
    assert all(call["boundless"] == (not call["inside"]) for call in calls)
