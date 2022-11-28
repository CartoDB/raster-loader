import numpy as np
from affine import Affine

from raster_loader.io import array_to_record


def test_array_to_record():
    arr = np.linspace(0, 100, 180 * 360).reshape(180, 360)
    geotransform = Affine.from_gdal(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)
    record = array_to_record(arr, geotransform)

    assert record["lat_NW"] == 90.0
    assert record["lon_NW"] == -180.0
    assert record["lat_NE"] == 90.0
    assert record["lon_NE"] == 180.0
    assert record["lat_SE"] == -90.0
    assert record["lon_SE"] == 180.0
    assert record["lat_SW"] == -90.0
    assert record["lon_SW"] == -180.0
    assert record["block_height"] == 180
    assert record["block_width"] == 360
    assert record["band_1_float64"] == arr.tobytes()


def test_array_to_record_offset():
    arr = np.linspace(0, 100, 160 * 340).reshape(160, 340)
    geotransform = Affine.from_gdal(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)
    record = array_to_record(arr, geotransform, row_off=20, col_off=20)

    assert record["lat_NW"] == 70.0
    assert record["lon_NW"] == -160.0
    assert record["lat_NE"] == 70.0
    assert record["lon_NE"] == 180.0
    assert record["lat_SE"] == -90.0
    assert record["lon_SE"] == 180.0
    assert record["lat_SW"] == -90.0
    assert record["lon_SW"] == -160.0
    assert record["block_height"] == 160
    assert record["block_width"] == 340
    assert record["band_1_float64"] == arr.tobytes()
