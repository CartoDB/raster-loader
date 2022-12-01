import json
import os
import numpy as np
import pytest

from affine import Affine

from raster_loader.io import array_to_record
from raster_loader.io import record_to_array


HERE = os.path.dirname(os.path.abspath(__file__))
fixtures_dir = os.path.join(HERE, "fixtures")


def test_array_to_record():
    arr = np.linspace(0, 100, 180 * 360).reshape(180, 360)
    geotransform = Affine.from_gdal(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)
    crs = "EPSG:4326"
    band = 1
    record = array_to_record(arr, geotransform, crs=crs, band=band)

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

    expected_attrs = {
        "band": band,
        "value_field": "band_1_float64",
        "dtype": "float64",
        "crs": crs,
        "gdal_transform": list(geotransform.to_gdal()),
        "row_off": 0,
        "col_off": 0,
    }

    for key, value in expected_attrs.items():
        assert json.loads(record["attrs"])[key] == value


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

    expected_attrs = {
        "band": 1,
        "value_field": "band_1_float64",
        "dtype": "float64",
        "crs": "EPSG:4326",
        "gdal_transform": list(geotransform.to_gdal()),
        "row_off": 20,
        "col_off": 20,
    }

    for key, value in expected_attrs.items():
        assert json.loads(record["attrs"])[key] == value


def test_record_to_array():
    arr = np.linspace(0, 100, 180 * 360).reshape(180, 360)
    geotransform = Affine.from_gdal(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)
    crs = "EPSG:4326"
    band = 1
    record = array_to_record(arr, geotransform, crs=crs, band=band)
    arr2 = record_to_array(record)
    assert np.allclose(arr, arr2)
    assert arr.dtype == arr2.dtype
    assert arr.shape == arr2.shape


def test_rasterio_to_record():
    import rasterio
    import os

    test_file = os.path.join(fixtures_dir, "mosaic.tif")
    band = 1

    with rasterio.open(test_file) as src:
        record = array_to_record(
            src.read(band), src.transform, crs=src.crs.to_string(), band=band
        )

    assert isinstance(record, dict)

    expected_attrs = {
        "band": 1,
        "value_field": "band_1_uint8",
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "gdal_transform": list(src.transform.to_gdal()),
        "row_off": 0,
        "col_off": 0,
    }

    for key, value in expected_attrs.items():
        assert json.loads(record["attrs"])[key] == value


@pytest.mark.integration_test
def test_bigquery_to_records():

    from raster_loader.io import bigquery_to_records

    test_cols = [
        "lat_NW",
        "lon_NW",
        "lat_NE",
        "lon_NE",
        "lat_SE",
        "lon_SE",
        "lat_SW",
        "lon_SW",
        "block_height",
        "block_width",
    ]

    records1 = bigquery_to_records(
        table_id="first_upload",
        project_id="carto-raster-loader",
        dataset_id="brendan_dev",
    )

    records2 = bigquery_to_records(
        table_id="first_upload2",
        project_id="carto-raster-loader",
        dataset_id="brendan_dev",
    )

    for c in test_cols:
        print("Testing column: {}".format(c))
        assert records1[c].equals(records2[c])


# OLDER TESTS
"""
@patch.object(RasterLoader, "_bigquery_client", return_value=mocks.bigquery_client())
def test_upload_to_bigquery_successful(*args, **kwargs):
    raster_loader = RasterLoader(file_path="raster_loader/tests/fixtures/mosaic.tif")

    raster_loader.to_bigquery(
        project="mock_project",
        dataset="mock_dataset",
        table="raster_data",
    )


@patch.object(
    RasterLoader,
    "_bigquery_client",
    side_effect=errors.ClientError,
)
def test_upload_to_bigquery_unsuccessful_client_error(*args, **kwargs):
    raster_loader = RasterLoader(file_path="raster_loader/tests/fixtures/mosaic.tif")

    with pytest.raises(errors.ClientError):
        raster_loader.to_bigquery(
            project="mock_project",
            dataset="mock_dataset",
            table="raster_data",
        )


@patch.object(
    RasterLoader,
    "_bigquery_client",
    return_value=mocks.bigquery_client(load_error=True),
)
def test_upload_to_bigquery_unsuccessful_load_error(*args, **kwargs):
    raster_loader = RasterLoader(file_path="raster_loader/tests/fixtures/mosaic.tif")

    with pytest.raises(errors.UploadError):
        raster_loader.to_bigquery(
            project="mock_project",
            dataset="mock_dataset",
            table="raster_data",
        )
"""
