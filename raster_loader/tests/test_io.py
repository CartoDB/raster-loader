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

    assert record["block"] == 5209556461146865663
    # assert record["block_height"] == 160
    # assert record["block_width"] == 340
    assert record["band_1_float64"] == arr_bytes
    assert record["metadata"] is None

    # expected_attrs = {
    #     "band": 1,
    #     "value_field": "band_1_float64",
    #     "dtype": "float64",
    #     "nodata": -1,
    #     "crs": "EPSG:4326",
    #     "gdal_transform": list(geotransform.to_gdal()),
    #     "row_off": 20,
    #     "col_off": 20,
    # }
    # for key, value in expected_attrs.items():
    #     assert json.loads(record["metadata"])[key] == value


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
    )

    result = bigquery_to_records(
        table_id=table_name,
        project_id=BQ_PROJECT_ID,
        dataset_id=BQ_DATASET_ID,
    )

    expected_columns = [
        "block",
        "metadata",
        "band1_uint8",
    ]

    assert sorted(list(result.columns)) == sorted(expected_columns)

    # TODO: select metadata row and check metadata contents
    # TODO: select some block row and check contents


@pytest.mark.integration_test
def test_rasterio_to_bigquery_with_quadbin_raster_custom_band_column():
    from raster_loader.io import rasterio_to_bigquery
    from raster_loader.io import bigquery_to_records

    check_integration_config()

    table_name = "test_mosaic_quadbin_custom_band_column_1"

    rasterio_to_bigquery(
        os.path.join(fixtures_dir, "quadbin_raster.tif"),
        table_name,
        BQ_DATASET_ID,
        BQ_PROJECT_ID,
        overwrite=True,
        band_name_prefix="customband",
    )

    result = bigquery_to_records(
        table_id=table_name,
        project_id=BQ_PROJECT_ID,
        dataset_id=BQ_DATASET_ID,
    )

    expected_columns = [
        "block",
        "metadata",
        "customband_uint8",
    ]

    assert sorted(list(result.columns)) == sorted(expected_columns)

    # TODO: select metadata row and check metadata contents
    # TODO: select some block row and check contents


@patch("raster_loader.io.check_if_bigquery_table_exists", return_value=False)
@patch("raster_loader.io.ask_yes_no_question", return_value=False)
def test_rasterio_to_bigquery(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

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
@patch("raster_loader.io.rasterio_windows_to_records", return_value={})
@patch("raster_loader.io.get_number_of_blocks", return_value=1)
@patch("raster_loader.io.write_metadata", return_value=None)
@patch("raster_loader.io.run_bigquery_query", return_value=None)
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
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

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
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

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
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

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
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

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
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

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
    )
    assert success
