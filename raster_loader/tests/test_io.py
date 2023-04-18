import json
import os
import sys
from unittest.mock import patch
import pyproj

from affine import Affine
import numpy as np
import pytest

from raster_loader import io
from raster_loader.tests import mocks


HERE = os.path.dirname(os.path.abspath(__file__))
fixtures_dir = os.path.join(HERE, "fixtures")


should_swap = {"=": sys.byteorder != "little", "<": False, ">": True, "|": False}

env_filename = os.path.join(HERE, ".env")
if os.path.isfile(env_filename):
    with open(env_filename) as env_file:
        for line in env_file:
            line = line.strip()
            if line:
                var, value = line.split("=")
                os.environ[var] = value


BQ_PROJECT_ID = os.environ.get("BQ_PROJECT_ID")
BQ_DATASET_ID = os.environ.get("BQ_DATASET_ID")


def check_integration_config():
    if not BQ_PROJECT_ID or not BQ_DATASET_ID:
        raise Exception(
            "You need to copy tests/.env.sample to test/.env and set your configuration"
            "before running the tests"
        )


def test_array_to_record():
    arr = np.linspace(0, 100, 180 * 360).reshape(180, 360)
    crs = "EPSG:4326"
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    geotransform = Affine.from_gdal(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)
    band = 1
    nodata = -1
    value_field = "band_1_float64"
    dtype_str = "float64"
    record = io.array_to_record(
        arr, nodata, band, value_field, dtype_str, transformer, geotransform, crs=crs
    )

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
        "nodata": nodata,
        "crs": crs,
        "gdal_transform": list(geotransform.to_gdal()),
        "row_off": 0,
        "col_off": 0,
    }

    for key, value in expected_attrs.items():
        assert json.loads(record["attrs"])[key] == value


def test_array_to_record_offset():
    arr = np.linspace(0, 100, 160 * 340).reshape(160, 340)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
    geotransform = Affine.from_gdal(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)

    record = io.array_to_record(
        arr,
        -1,
        1,
        "band_1_float64",
        "float64",
        transformer,
        geotransform,
        row_off=20,
        col_off=20,
    )

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


def test_array_to_quadbin_record():
    arr = np.linspace(0, 100, 160 * 340).reshape(160, 340)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
    geotransform = Affine.from_gdal(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)
    record = io.array_to_quadbin_record(
        arr,
        -1,
        1,
        "band_1_float64",
        "float64",
        transformer,
        geotransform,
        resolution=4,
        row_off=20,
        col_off=20,
    )

    if should_swap[arr.dtype.byteorder]:
        arr_bytes = np.ascontiguousarray(arr.byteswap()).tobytes()
    else:
        arr_bytes = np.ascontiguousarray(arr).tobytes()

    assert record["quadbin"] == 5209556461146865663
    assert record["block_height"] == 160
    assert record["block_width"] == 340
    assert record["band_1_float64"] == arr_bytes

    expected_attrs = {
        "band": 1,
        "value_field": "band_1_float64",
        "dtype": "float64",
        "nodata": -1,
        "crs": "EPSG:4326",
        "gdal_transform": list(geotransform.to_gdal()),
        "row_off": 20,
        "col_off": 20,
    }

    for key, value in expected_attrs.items():
        assert json.loads(record["attrs"])[key] == value


def test_record_to_array():
    arr = np.linspace(0, 100, 180 * 360).reshape(180, 360)
    crs = "EPSG:4326"
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    geotransform = Affine.from_gdal(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)
    crs = "EPSG:4326"
    band = 1
    nodata = 1
    value_field = "band_1_float64"
    dtype_str = "float64"
    record = io.array_to_record(
        arr, nodata, band, value_field, dtype_str, transformer, geotransform, crs=crs
    )
    arr2 = io.record_to_array(record)
    assert np.allclose(arr, arr2)
    assert arr.dtype.name == arr2.dtype.name
    assert arr.shape == arr2.shape


def test_record_to_array_invalid_dtype():
    arr = np.linspace(0, 100, 180 * 360).reshape(180, 360)
    crs = "EPSG:4326"
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    geotransform = Affine.from_gdal(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)
    band = 1
    nodata = 1
    value_field = "band_1_dtype"
    dtype_str = "dtype"
    record = io.array_to_record(
        arr, nodata, band, value_field, dtype_str, transformer, geotransform, crs=crs
    )

    with pytest.raises(TypeError):
        io.record_to_array(record, "band_1_dtype")


def test_rasterio_to_record():
    import rasterio
    import os

    test_file = os.path.join(fixtures_dir, "mosaic.tif")
    band = 1
    value_field = "band_1_uint8"
    dtype_str = "uint8"
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)

    with rasterio.open(test_file) as src:
        record = io.array_to_record(
            src.read(band),
            src.nodata,
            band,
            value_field,
            dtype_str,
            transformer,
            src.transform,
            crs=src.crs.to_string(),
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


def test_rasterio_to_record_with_nodata():
    import rasterio
    import os

    test_file = os.path.join(fixtures_dir, "fuji.tif")
    band = 1
    value_field = "band_1_uint8"
    dtype_str = "uint8"
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)

    with rasterio.open(test_file) as src:
        record = io.array_to_record(
            src.read(band),
            src.nodata,
            band,
            value_field,
            dtype_str,
            transformer,
            src.transform,
            crs=src.crs.to_string(),
        )

    assert isinstance(record, dict)

    expected_attrs = {
        "band": 1,
        "value_field": "band_1_uint8",
        "dtype": "uint8",
        "nodata": -32767,
        "crs": "EPSG:8692",
        "gdal_transform": list(src.transform.to_gdal()),
        "row_off": 0,
        "col_off": 0,
    }

    for key, value in expected_attrs.items():
        assert json.loads(record["attrs"])[key] == value


@pytest.mark.integration_test
def test_rasterio_to_bigquery_with_generic_raster():
    from raster_loader.io import rasterio_to_bigquery
    from raster_loader.io import bigquery_to_records

    check_integration_config()

    table_name = "test_mosaic_1"

    rasterio_to_bigquery(
        os.path.join(fixtures_dir, "mosaic.tif"),
        table_name,
        BQ_DATASET_ID,
        BQ_PROJECT_ID,
        overwrite=True,
        output_quadbin=False,
    )

    result = bigquery_to_records(
        table_id=table_name,
        project_id=BQ_PROJECT_ID,
        dataset_id=BQ_DATASET_ID,
    )

    expected_columns = [
        "lon_NW",
        "lat_NW",
        "lon_NE",
        "lat_NE",
        "lon_SE",
        "lat_SE",
        "lon_SW",
        "lat_SW",
        "geog",
        "block_height",
        "block_width",
        "attrs",
        "band_1_uint8",
    ]

    assert sorted(list(result.columns)) == sorted(expected_columns)

    # TODO: select metadata row and check metadata contents
    # TODO: select some block row and check contents


@pytest.mark.integration_test
def test_rasterio_to_bigquery_with_quadbin_raster():
    from raster_loader.io import rasterio_to_bigquery
    from raster_loader.io import bigquery_to_records

    check_integration_config()

    table_name = "test_mosaic_quadbin_1"

    rasterio_to_bigquery(
        os.path.join(fixtures_dir, "quadbin_raster.tif"),
        table_name,
        BQ_DATASET_ID,
        BQ_PROJECT_ID,
        overwrite=True,
        output_quadbin=True,
    )

    result = bigquery_to_records(
        table_id=table_name,
        project_id=BQ_PROJECT_ID,
        dataset_id=BQ_DATASET_ID,
    )

    expected_columns = [
        "quadbin",
        "block_height",
        "block_width",
        "attrs",
        "band_1_uint8",
    ]

    assert sorted(list(result.columns)) == sorted(expected_columns)

    # TODO: select metadata row and check metadata contents
    # TODO: select some block row and check contents


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

    import rasterio

    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    # test that invalid input crs raises an error
    invalid_crs = 8675309
    with pytest.raises(rasterio.errors.CRSError):
        io.rasterio_to_bigquery(
            test_file,
            project_id="test",
            dataset_id="test",
            table_id="test",
            client=client,
            input_crs=invalid_crs,
        )


@patch("raster_loader.io.check_if_bigquery_table_exists", return_value=True)
@patch("raster_loader.io.delete_bigquery_table", return_value=None)
@patch("raster_loader.io.check_if_bigquery_table_is_empty", return_value=False)
@patch("raster_loader.io.ask_yes_no_question", return_value=True)
def test_rasterio_to_bigquery_invalid_quadbin_raster(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    with pytest.raises(OSError):
        io.rasterio_to_bigquery(
            test_file,
            project_id="test",
            dataset_id="test",
            table_id="test",
            client=client,
            output_quadbin=True,
        )


@patch("raster_loader.io.check_if_bigquery_table_exists", return_value=True)
@patch("raster_loader.io.delete_bigquery_table", return_value=None)
@patch("raster_loader.io.check_if_bigquery_table_is_empty", return_value=False)
@patch("raster_loader.io.ask_yes_no_question", return_value=True)
def test_rasterio_to_bigquery_valid_quadbbin_raster(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "quadbin_raster.tif")

    success = io.rasterio_to_bigquery(
        test_file,
        project_id="test",
        dataset_id="test",
        table_id="test",
        client=client,
        output_quadbin=True,
    )
    assert success
