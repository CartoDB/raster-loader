import os
from itertools import chain
from unittest.mock import patch

from rasterio.windows import Window

import raster_loader.io.common as common

from raster_loader.io.common import (
    is_window_inside,
    rasterio_overview_to_records,
    rasterio_windows_to_records,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES_DIR = os.path.join(HERE, "fixtures")


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


def test_direct_reads_produce_the_same_records_as_boundless_reads():
    file_path = os.path.join(FIXTURES_DIR, "mosaic_cog.tif")

    records = _records(file_path)
    with patch("raster_loader.io.common.is_window_inside", return_value=False):
        boundless_records = _records(file_path)

    assert records
    assert records == boundless_records


def test_boundless_is_used_only_for_windows_overrunning_the_raster():
    file_path = os.path.join(FIXTURES_DIR, "mosaic_cog.tif")
    calls = []
    read_filled = common.read_filled

    def spy(raster_dataset, band, no_data_value, **args):
        calls.append(
            (
                args["boundless"],
                is_window_inside(args["window"], raster_dataset),
            )
        )
        return read_filled(raster_dataset, band, no_data_value, **args)

    with patch("raster_loader.io.common.read_filled", spy):
        _records(file_path)

    assert any(boundless for boundless, _ in calls)
    assert any(not boundless for boundless, _ in calls)
    assert all(boundless == (not inside) for boundless, inside in calls)
