import math
import os
import sys
import json
from unittest.mock import patch
import pyproj

from affine import Affine
import numpy as np
import pandas as pd
import pytest

from raster_loader import io
from raster_loader.tests import mocks


HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    arr = np.linspace(0, 100, 160 * 340).reshape(160, 340)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4326", always_xy=True)
    geotransform = Affine.from_gdal(-180.0, 1.0, 0.0, 90.0, 0.0, -1.0)
    record = io.common.array_to_record(
        arr,
        "band_1",
        lambda x: x,
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
    assert record["band_1"] == arr_bytes
    assert record["metadata"] is None


@pytest.mark.integration_test
def test_rasterio_to_bigquery_with_raster_default_band_name():
    from raster_loader.io.bigquery import rasterio_to_table
    from raster_loader.io.bigquery import table_to_records

    check_integration_config()

    table_name = "test_mosaic_1"

    rasterio_to_table(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        table_name,
        BQ_DATASET_ID,
        BQ_PROJECT_ID,
        overwrite=True,
    )

    result = table_to_records(
        table_id=table_name,
        project_id=BQ_PROJECT_ID,
        dataset_id=BQ_DATASET_ID,
    )

    expected_dataframe = pd.read_pickle(
        os.path.join(fixtures_dir, "expected_default_column.pkl")
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
def test_rasterio_to_bigquery_appending_rows():
    from raster_loader.io.bigquery import rasterio_to_table
    from raster_loader.io.bigquery import table_to_records

    check_integration_config()

    table_name = "test_mosaic_append_rows"

    rasterio_to_table(
        os.path.join(fixtures_dir, "mosaic_cog_1_1.tif"),
        table_name,
        BQ_DATASET_ID,
        BQ_PROJECT_ID,
        overwrite=True,
    )

    result = table_to_records(
        table_id=table_name, project_id=BQ_PROJECT_ID, dataset_id=BQ_DATASET_ID
    )

    metadata = json.loads([x for x in list(result.metadata) if x][0])

    assert metadata == {
        "pixel_resolution": 13,
        "block_resolution": 5,
        "minresolution": 5,
        "maxresolution": 5,
        "nodata": None,
        "bands": [{"type": "uint8", "name": "band_1"}],
        "bounds": [
            11.249999999997055,
            40.979898069622585,
            22.49999999999707,
            48.92249926376037,
        ],
        "center": [16.874999999997062, 44.951198666691475, 5],
        "width": 256,
        "height": 256,
        "block_width": 256,
        "block_height": 256,
        "num_blocks": 1,
        "num_pixels": 65536,
    }

    rasterio_to_table(
        os.path.join(fixtures_dir, "mosaic_cog_1_2.tif"),
        table_name,
        BQ_DATASET_ID,
        BQ_PROJECT_ID,
        append=True,
    )

    result = table_to_records(
        table_id=table_name, project_id=BQ_PROJECT_ID, dataset_id=BQ_DATASET_ID
    )

    metadata = json.loads([x for x in list(result.metadata) if x][0])

    assert metadata == {
        "bands": [{"name": "band_1", "type": "uint8"}],
        "block_height": 256,
        "block_width": 256,
        "bounds": [
            11.249999999997055,
            40.979898069622585,
            33.74999999999708,
            48.92249926376037,
        ],
        "center": [22.499999999997065, 44.95119866669148, 5],
        "height": 256,
        "maxresolution": 5,
        "minresolution": 5,
        "nodata": None,
        "num_blocks": 2,
        "num_pixels": 131072,
        "block_resolution": 5,
        "pixel_resolution": 13,
        "width": 512,
    }

    assert len(result) == 3


@pytest.mark.integration_test
def test_rasterio_to_bigquery_with_raster_custom_band_column():
    from raster_loader.io.bigquery import rasterio_to_table
    from raster_loader.io.bigquery import table_to_records

    check_integration_config()

    table_name = "test_mosaic_custom_band_column_1"

    rasterio_to_table(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        table_name,
        BQ_DATASET_ID,
        BQ_PROJECT_ID,
        overwrite=True,
        bands_info=[(1, "customband")],
    )

    result = table_to_records(
        table_id=table_name,
        project_id=BQ_PROJECT_ID,
        dataset_id=BQ_DATASET_ID,
    )
    # sort value because return query can vary the order of rows
    result = result.sort_values("block")

    expected_dataframe = pd.read_pickle(
        os.path.join(fixtures_dir, "expected_custom_column.pkl")
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
def test_rasterio_to_bigquery_with_raster_multiple_default():
    from raster_loader.io.bigquery import rasterio_to_table
    from raster_loader.io.bigquery import table_to_records

    check_integration_config()

    table_name = "test_mosaic_multiple_default_bands"

    rasterio_to_table(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        table_name,
        BQ_DATASET_ID,
        BQ_PROJECT_ID,
        overwrite=True,
        bands_info=[(1, None), (2, None)],
    )

    result = table_to_records(
        table_id=table_name,
        project_id=BQ_PROJECT_ID,
        dataset_id=BQ_DATASET_ID,
    )

    # sort value because return query can vary the order of rows
    result = result.sort_values("block")

    expected_dataframe = pd.read_pickle(
        os.path.join(fixtures_dir, "expected_multiple_column.pkl")
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
def test_rasterio_to_bigquery_with_raster_multiple_custom():
    from raster_loader.io.bigquery import rasterio_to_table
    from raster_loader.io.bigquery import table_to_records

    check_integration_config()

    table_name = "test_mosaic_multiple_custom_bands"

    rasterio_to_table(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        table_name,
        BQ_DATASET_ID,
        BQ_PROJECT_ID,
        overwrite=True,
        bands_info=[(1, "custom_band_1"), (2, "custom_band_2")],
    )

    result = table_to_records(
        table_id=table_name,
        project_id=BQ_PROJECT_ID,
        dataset_id=BQ_DATASET_ID,
    )

    # sort value because return query can vary the order of rows
    result = result.sort_values("block")

    expected_dataframe = pd.read_pickle(
        os.path.join(fixtures_dir, "expected_custom_multiple_column.pkl")
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


@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=False)
def test_rasterio_to_bigquery_wrong_band_name_metadata(*args, **kwargs):
    from raster_loader.io.bigquery import rasterio_to_table

    table_name = "test_mosaic_custom_band_column_1"
    client = mocks.bigquery_client()

    with pytest.raises(IOError):
        rasterio_to_table(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            table_name,
            BQ_DATASET_ID,
            BQ_PROJECT_ID,
            overwrite=True,
            client=client,
            bands_info=[(1, "metadata"), (2, "custom_band_2")],
        )


@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=False)
def test_rasterio_to_bigquery_wrong_band_name_block(*args, **kwargs):
    from raster_loader.io.bigquery import rasterio_to_table

    table_name = "test_mosaic_custom_band_column_1"
    client = mocks.bigquery_client()

    with pytest.raises(IOError):
        rasterio_to_table(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            table_name,
            BQ_DATASET_ID,
            BQ_PROJECT_ID,
            overwrite=True,
            client=client,
            bands_info=[(1, "block"), (2, "custom_band_2")],
        )


@patch("raster_loader.io.bigquery.check_if_table_exists", return_value=False)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=False)
def test_rasterio_to_table(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

    success = io.bigquery.rasterio_to_table(
        test_file,
        project_id="test",
        dataset_id="test",
        table_id="test",
        client=client,
    )
    assert success


@patch("raster_loader.io.bigquery.check_if_table_exists", return_value=True)
@patch("raster_loader.io.bigquery.delete_table", return_value=None)
@patch("raster_loader.io.common.rasterio_windows_to_records", return_value={})
@patch("raster_loader.io.common.rasterio_metadata", return_value={})
@patch("raster_loader.io.common.get_number_of_blocks", return_value=1)
@patch("raster_loader.io.bigquery.write_metadata", return_value=None)
def test_rasterio_to_bigquery_overwrite(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

    success = io.bigquery.rasterio_to_table(
        test_file,
        project_id="test",
        dataset_id="test",
        table_id="test",
        client=client,
        overwrite=True,
    )
    assert success


@patch("raster_loader.io.bigquery.check_if_table_exists", return_value=True)
@patch("raster_loader.io.bigquery.delete_table", return_value=None)
@patch("raster_loader.io.bigquery.check_if_table_is_empty", return_value=False)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=True)
@patch(
    "raster_loader.io.bigquery.get_metadata",
    return_value={
        "bounds": [0, 0, 0, 0],
        "block_resolution": 5,
        "nodata": None,
        "block_width": 256,
        "block_height": 256,
        "bands": [{"type": "uint8", "name": "band_1"}],
        "num_blocks": 1,
        "num_pixels": 1,
    },
)
def test_rasterio_to_bigquery_table_is_not_empty_append(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

    success = io.bigquery.rasterio_to_table(
        test_file,
        project_id="test",
        dataset_id="test",
        table_id="test",
        client=client,
    )
    assert success


@patch("raster_loader.io.bigquery.check_if_table_exists", return_value=True)
@patch("raster_loader.io.bigquery.delete_table", return_value=None)
@patch("raster_loader.io.bigquery.check_if_table_is_empty", return_value=False)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=False)
def test_rasterio_to_bigquery_table_is_not_empty_dont_append(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    with pytest.raises(SystemExit):
        io.bigquery.rasterio_to_table(
            test_file,
            project_id="test",
            dataset_id="test",
            table_id="test",
            client=client,
        )


@patch("raster_loader.io.bigquery.records_to_bigquery", side_effect=Exception())
@patch("raster_loader.io.bigquery.delete_table", return_value=True)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=True)
def test_rasterio_to_bigquery_uploading_error(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    with pytest.raises(IOError):
        io.bigquery.rasterio_to_table(
            test_file,
            project_id="test",
            dataset_id="test",
            table_id="test",
            client=client,
            chunk_size=100,
        )


@patch("raster_loader.io.bigquery.records_to_bigquery", side_effect=KeyboardInterrupt())
@patch("raster_loader.io.bigquery.delete_table", return_value=True)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=True)
@patch("raster_loader.io.bigquery.check_if_table_exists", return_value=False)
def test_rasterio_to_bigquery_keyboard_interrupt(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

    with pytest.raises(KeyboardInterrupt):
        io.bigquery.rasterio_to_table(
            test_file,
            project_id="test",
            dataset_id="test",
            table_id="test",
            client=client,
            chunk_size=100,
        )


@patch("raster_loader.io.bigquery.check_if_table_exists", return_value=False)
def test_rasterio_to_bigquery_with_chunk_size(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

    success = io.bigquery.rasterio_to_table(
        test_file,
        project_id="test",
        dataset_id="test",
        table_id="test",
        client=client,
        chunk_size=100,
    )
    assert success


@patch("raster_loader.io.bigquery.check_if_table_exists", return_value=False)
def test_rasterio_to_bigquery_with_one_chunk_size(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

    success = io.bigquery.rasterio_to_table(
        test_file,
        project_id="test",
        dataset_id="test",
        table_id="test",
        client=client,
        chunk_size=1,
    )
    assert success


@patch("raster_loader.io.bigquery.check_if_table_exists", return_value=True)
@patch("raster_loader.io.bigquery.delete_table", return_value=None)
@patch("raster_loader.io.bigquery.check_if_table_is_empty", return_value=False)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=True)
def test_rasterio_to_bigquery_invalid_raster(*args, **kwargs):
    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    with pytest.raises(OSError):
        io.bigquery.rasterio_to_table(
            test_file,
            project_id="test",
            dataset_id="test",
            table_id="test",
            client=client,
        )


@patch("raster_loader.io.bigquery.check_if_table_exists", return_value=True)
@patch("raster_loader.io.bigquery.delete_table", return_value=None)
@patch("raster_loader.io.bigquery.check_if_table_is_empty", return_value=False)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=True)
@patch(
    "raster_loader.io.bigquery.get_metadata",
    return_value={
        "bounds": [0, 0, 0, 0],
        "block_resolution": 5,
        "nodata": None,
        "block_width": 256,
        "block_height": 256,
        "bands": [{"type": "uint8", "name": "band_1"}],
        "num_blocks": 1,
        "num_pixels": 1,
    },
)
def test_rasterio_to_bigquery_valid_raster(*args, **kwargs):
    from raster_loader.io.bigquery import rasterio_to_table

    client = mocks.bigquery_client()
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

    success = rasterio_to_table(
        test_file,
        project_id="test",
        dataset_id="test",
        table_id="test",
        client=client,
    )
    assert success


@patch("raster_loader.io.bigquery.check_if_table_exists", return_value=True)
@patch("raster_loader.io.bigquery.delete_table", return_value=None)
@patch("raster_loader.io.bigquery.check_if_table_is_empty", return_value=False)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=True)
@patch(
    "raster_loader.io.bigquery.get_metadata",
    return_value={"bounds": [0, 0, 0, 0], "block_resolution": 1},
)
def test_append_with_different_resolution(*args, **kwargs):
    from raster_loader.io.bigquery import rasterio_to_table

    client = mocks.bigquery_client()

    with pytest.raises(OSError):
        rasterio_to_table(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            project_id="test",
            dataset_id="test",
            table_id="test",
            overwrite=False,
            append=True,
            client=client,
        )
