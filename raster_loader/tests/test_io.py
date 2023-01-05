import json
import os
import sys
from unittest.mock import patch

from affine import Affine
import numpy as np
import pytest

from raster_loader import io
from raster_loader.tests import mocks


HERE = os.path.dirname(os.path.abspath(__file__))
fixtures_dir = os.path.join(HERE, "fixtures")


should_swap = {"=": sys.byteorder == "little", "<": True, ">": False, "|": False}


def test_array_to_record():
    arr = np.linspace(0, 100, 180 * 360).reshape(180, 360)
    geotransform = Affine.from_gdal(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)
    crs = "EPSG:4326"
    band = 1
    record = io.array_to_record(arr, geotransform, crs=crs, band=band)

    if should_swap[arr.dtype.byteorder]:
        arr_bytes = np.ascontiguousarray(arr.byteswap()).tobytes()
    else:
        arr_bytes = np.ascontiguousarray(arr).tobytes()

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
    assert record["band_1_float64"] == arr_bytes

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
    record = io.array_to_record(arr, geotransform, row_off=20, col_off=20)

    if should_swap[arr.dtype.byteorder]:
        arr_bytes = np.ascontiguousarray(arr.byteswap()).tobytes()
    else:
        arr_bytes = np.ascontiguousarray(arr).tobytes()

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
    assert record["band_1_float64"] == arr_bytes

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
    record = io.array_to_record(arr, geotransform, crs=crs, band=band)
    arr2 = io.record_to_array(record)
    assert np.allclose(arr, arr2)
    assert arr.dtype.name == arr2.dtype.name
    assert arr.shape == arr2.shape


def test_record_to_array_invalid_dtype():
    arr = np.linspace(0, 100, 180 * 360).reshape(180, 360)
    geotransform = Affine.from_gdal(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)
    crs = "EPSG:4326"
    band = 1
    record = io.array_to_record(arr, geotransform, crs=crs, band=band)

    with pytest.raises(TypeError):
        io.record_to_array(record, "band_1_dtype")


def test_rasterio_to_record():
    import rasterio
    import os

    test_file = os.path.join(fixtures_dir, "mosaic.tif")
    band = 1

    with rasterio.open(test_file) as src:
        record = io.array_to_record(
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


@patch("raster_loader.io.check_if_bigquery_table_exists", return_value=False)
def test_rasterio_to_bigquery(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    success = io.rasterio_to_bigquery(
        test_file, project_id="test", dataset_id="test", table_id="test", client=client
    )
    assert success


@patch("raster_loader.io.check_if_bigquery_table_exists", return_value=True)
@patch("raster_loader.io.delete_bigquery_table", return_value=None)
def test_rasterio_to_bigquery_overwrite(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    success = io.rasterio_to_bigquery(
        test_file,
        project_id="test",
        dataset_id="test",
        table_id="test",
        client=client,
        overwrite=True,
    )
    assert success


@patch("raster_loader.io.check_if_bigquery_table_exists", return_value=True)
@patch("raster_loader.io.delete_bigquery_table", return_value=None)
@patch("raster_loader.io.check_if_bigquery_table_is_empty", return_value=False)
@patch("raster_loader.io.ask_yes_no_question", return_value=True)
def test_rasterio_to_bigquery_table_is_not_empty_append(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    success = io.rasterio_to_bigquery(
        test_file,
        project_id="test",
        dataset_id="test",
        table_id="test",
        client=client,
    )
    assert success


@patch("raster_loader.io.check_if_bigquery_table_exists", return_value=True)
@patch("raster_loader.io.delete_bigquery_table", return_value=None)
@patch("raster_loader.io.check_if_bigquery_table_is_empty", return_value=False)
@patch("raster_loader.io.ask_yes_no_question", return_value=False)
def test_rasterio_to_bigquery_table_is_not_empty_dont_append(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    with pytest.raises(SystemExit):
        io.rasterio_to_bigquery(
            test_file,
            project_id="test",
            dataset_id="test",
            table_id="test",
            client=client,
        )


@patch("raster_loader.io.records_to_bigquery", side_effect=Exception())
@patch("raster_loader.io.delete_bigquery_table", return_value=True)
@patch("raster_loader.io.ask_yes_no_question", return_value=True)
def test_rasterio_to_bigquery_uploading_error(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    with pytest.raises(IOError):
        io.rasterio_to_bigquery(
            test_file,
            project_id="test",
            dataset_id="test",
            table_id="test",
            client=client,
            chunk_size=100,
        )


@patch("raster_loader.io.records_to_bigquery", side_effect=KeyboardInterrupt())
@patch("raster_loader.io.delete_bigquery_table", return_value=True)
@patch("raster_loader.io.ask_yes_no_question", return_value=True)
@patch("raster_loader.io.check_if_bigquery_table_exists", return_value=False)
def test_rasterio_to_bigquery_keyboard_interrupt(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    with pytest.raises(KeyboardInterrupt):
        io.rasterio_to_bigquery(
            test_file,
            project_id="test",
            dataset_id="test",
            table_id="test",
            client=client,
            chunk_size=100,
        )


@patch("raster_loader.io.check_if_bigquery_table_exists", return_value=False)
def test_rasterio_to_bigquery_with_chunk_size(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    success = io.rasterio_to_bigquery(
        test_file,
        project_id="test",
        dataset_id="test",
        table_id="test",
        client=client,
        chunk_size=100,
    )
    assert success


@patch("raster_loader.io.check_if_bigquery_table_exists", return_value=False)
def test_rasterio_to_bigquery_with_one_chunk_size(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    success = io.rasterio_to_bigquery(
        test_file,
        project_id="test",
        dataset_id="test",
        table_id="test",
        client=client,
        chunk_size=1,
    )
    assert success


@patch("raster_loader.io.check_if_bigquery_table_exists", return_value=False)
def test_rasterio_to_bigquery_invalid_input_crs(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    success = io.rasterio_to_bigquery(
        test_file,
        project_id="test",
        dataset_id="test",
        table_id="test",
        client=client,
        input_crs=3232,
    )
    assert success
