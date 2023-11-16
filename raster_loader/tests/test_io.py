import math
import os
import sys
from unittest.mock import patch
import pyproj

from affine import Affine
import numpy as np
import pandas as pd
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
        "band_1_float64",
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
    assert record["band_1_float64"] == arr_bytes
    assert record["metadata"] is None


@pytest.mark.integration_test
def test_rasterio_to_bigquery_with_quadbin_raster_default_band_name():
    from raster_loader.io import rasterio_to_bigquery
    from raster_loader.io import bigquery_to_records

    check_integration_config()

    table_name = "test_mosaic_quadbin_1"

    rasterio_to_bigquery(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
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

    # get expected data from fixture numpy array
    expected_ndarray = np.load(
        os.path.join(fixtures_dir, "expected_default_column.npy"), allow_pickle=True
    )
    expected_dataframe = pd.DataFrame(
        expected_ndarray, columns=["block", "metadata", "band_1"]
    )
    expected_dataframe = expected_dataframe.sort_values("block")

    assert sorted(result.columns) == sorted(expected_dataframe.columns)
    assert sorted(
        list(result.block), key=lambda x: x if x is not None else -math.inf
    ) == sorted(
        list(expected_dataframe.block), key=lambda x: x if x is not None else -math.inf
    )
    assert sorted(
        list(result.metadata), key=lambda x: x if x is not None else ""
    ) == sorted(
        list(expected_dataframe.metadata), key=lambda x: x if x is not None else ""
    )
    assert sorted(
        list(result.band_1), key=lambda x: x if x is not None else b""
    ) == sorted(
        list(expected_dataframe.band_1), key=lambda x: x if x is not None else b""
    )


@pytest.mark.integration_test
def test_rasterio_to_bigquery_with_quadbin_raster_custom_band_column():
    from raster_loader.io import rasterio_to_bigquery
    from raster_loader.io import bigquery_to_records

    check_integration_config()

    table_name = "test_mosaic_quadbin_custom_band_column_1"

    rasterio_to_bigquery(
        # os.path.join(fixtures_dir, "quadbin_raster.tif"),
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        table_name,
        BQ_DATASET_ID,
        BQ_PROJECT_ID,
        overwrite=True,
        bands_info=[(1, "customband")],
    )

    result = bigquery_to_records(
        table_id=table_name,
        project_id=BQ_PROJECT_ID,
        dataset_id=BQ_DATASET_ID,
    )
    # sort value because return query can vary the order of rows
    result = result.sort_values("block")

    # get expected data from fixture numpy array
    expected_ndarray = np.load(
        os.path.join(fixtures_dir, "expected_custom_column.npy"), allow_pickle=True
    )
    expected_dataframe = pd.DataFrame(
        expected_ndarray, columns=["block", "metadata", "customband"]
    )
    expected_dataframe = expected_dataframe.sort_values("block")

    assert sorted(result.columns) == sorted(expected_dataframe.columns)
    assert sorted(
        list(result.block), key=lambda x: x if x is not None else -math.inf
    ) == sorted(
        list(expected_dataframe.block), key=lambda x: x if x is not None else -math.inf
    )
    assert sorted(
        list(result.metadata), key=lambda x: x if x is not None else ""
    ) == sorted(
        list(expected_dataframe.metadata), key=lambda x: x if x is not None else ""
    )
    assert sorted(
        list(result.customband), key=lambda x: x if x is not None else b""
    ) == sorted(
        list(expected_dataframe.customband), key=lambda x: x if x is not None else b""
    )


@pytest.mark.integration_test
def test_rasterio_to_bigquery_with_quadbin_raster_multiple_default():
    from raster_loader.io import rasterio_to_bigquery
    from raster_loader.io import bigquery_to_records

    check_integration_config()

    table_name = "test_mosaic_quadbin_default_band_column_1"

    rasterio_to_bigquery(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        table_name,
        BQ_DATASET_ID,
        BQ_PROJECT_ID,
        overwrite=True,
        bands_info=[(1, None), (2, None)],
    )

    result = bigquery_to_records(
        table_id=table_name,
        project_id=BQ_PROJECT_ID,
        dataset_id=BQ_DATASET_ID,
    )

    # sort value because return query can vary the order of rows
    result = result.sort_values("block")

    # get expected data from fixture numpy array
    expected_ndarray = np.load(
        os.path.join(fixtures_dir, "expected_default_multiple_column.npy"),
        allow_pickle=True,
    )
    expected_dataframe = pd.DataFrame(
        expected_ndarray, columns=["block", "metadata", "band_1", "band_2"]
    )
    expected_dataframe = expected_dataframe.sort_values("block")

    assert sorted(result.columns) == sorted(expected_dataframe.columns)
    assert sorted(
        list(result.block), key=lambda x: x if x is not None else -math.inf
    ) == sorted(
        list(expected_dataframe.block), key=lambda x: x if x is not None else -math.inf
    )
    assert sorted(
        list(result.metadata), key=lambda x: x if x is not None else ""
    ) == sorted(
        list(expected_dataframe.metadata), key=lambda x: x if x is not None else ""
    )
    assert sorted(
        list(result.band_1), key=lambda x: x if x is not None else b""
    ) == sorted(
        list(expected_dataframe.band_1), key=lambda x: x if x is not None else b""
    )
    assert sorted(
        list(result.band_2), key=lambda x: x if x is not None else b""
    ) == sorted(
        list(expected_dataframe.band_2), key=lambda x: x if x is not None else b""
    )


@pytest.mark.integration_test
def test_rasterio_to_bigquery_with_quadbin_raster_multiple_custom():
    from raster_loader.io import rasterio_to_bigquery
    from raster_loader.io import bigquery_to_records

    check_integration_config()

    table_name = "test_mosaic_quadbin_custom_band_column_1"

    rasterio_to_bigquery(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        table_name,
        BQ_DATASET_ID,
        BQ_PROJECT_ID,
        overwrite=True,
        bands_info=[(1, "custom_band_1"), (2, "custom_band_2")],
    )

    result = bigquery_to_records(
        table_id=table_name,
        project_id=BQ_PROJECT_ID,
        dataset_id=BQ_DATASET_ID,
    )

    # sort value because return query can vary the order of rows
    result = result.sort_values("block")

    # get expected data from fixture numpy array
    expected_ndarray = np.load(
        os.path.join(fixtures_dir, "expected_custom_multiple_column.npy"),
        allow_pickle=True,
    )
    expected_dataframe = pd.DataFrame(
        expected_ndarray,
        columns=["block", "metadata", "custom_band_1", "custom_band_2"],
    )
    expected_dataframe = expected_dataframe.sort_values("block")

    assert sorted(result.columns) == sorted(expected_dataframe.columns)
    assert sorted(
        list(result.block), key=lambda x: x if x is not None else -math.inf
    ) == sorted(
        list(expected_dataframe.block), key=lambda x: x if x is not None else -math.inf
    )
    assert sorted(
        list(result.metadata), key=lambda x: x if x is not None else ""
    ) == sorted(
        list(expected_dataframe.metadata), key=lambda x: x if x is not None else ""
    )
    assert sorted(
        list(result.custom_band_1), key=lambda x: x if x is not None else b""
    ) == sorted(
        list(expected_dataframe.custom_band_1),
        key=lambda x: x if x is not None else b"",
    )
    assert sorted(
        list(result.custom_band_2), key=lambda x: x if x is not None else b""
    ) == sorted(
        list(expected_dataframe.custom_band_2),
        key=lambda x: x if x is not None else b"",
    )


@patch("raster_loader.io.ask_yes_no_question", return_value=False)
def test_rasterio_to_bigquery_wrong_band_name_metadata(*args, **kwargs):
    from raster_loader.io import rasterio_to_bigquery

    table_name = "test_mosaic_quadbin_custom_band_column_1"
    client = mocks.bigquery_client()

    with pytest.raises(IOError):
        rasterio_to_bigquery(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            table_name,
            BQ_DATASET_ID,
            BQ_PROJECT_ID,
            overwrite=True,
            client=client,
            bands_info=[(1, "metadata"), (2, "custom_band_2")],
        )


@patch("raster_loader.io.ask_yes_no_question", return_value=False)
def test_rasterio_to_bigquery_wrong_band_name_block(*args, **kwargs):
    from raster_loader.io import rasterio_to_bigquery

    table_name = "test_mosaic_quadbin_custom_band_column_1"
    client = mocks.bigquery_client()

    with pytest.raises(IOError):
        rasterio_to_bigquery(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            table_name,
            BQ_DATASET_ID,
            BQ_PROJECT_ID,
            overwrite=True,
            client=client,
            bands_info=[(1, "block"), (2, "custom_band_2")],
        )


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
