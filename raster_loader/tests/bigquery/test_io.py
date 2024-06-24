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
import rasterio

from raster_loader import io
from raster_loader.tests import mocks
from raster_loader.io.bigquery import BigQueryConnection


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
        window=rasterio.windows.Window(col_off=20, row_off=20, width=340, height=160),
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
    check_integration_config()

    table_name = "test_mosaic_1"
    fqn = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}"

    connector = BigQueryConnection(BQ_PROJECT_ID)

    connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        fqn,
        overwrite=True,
    )

    result = connector.get_records(fqn, 20)

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

    table = connector.client.get_table(fqn)
    assert table.labels.get("raster_loader") is not None


@pytest.mark.integration_test
def test_rasterio_to_bigquery_with_blocksize_512():
    check_integration_config()

    table_name = "test_mosaic_blocksize_512"
    fqn = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}"

    connector = BigQueryConnection(BQ_PROJECT_ID)

    connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog_512.tif"),
        fqn,
        overwrite=True,
    )

    result = connector.get_records(fqn, 20)

    expected_dataframe = pd.read_pickle(
        os.path.join(fixtures_dir, "expected_blocksize_512.pkl")
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
    check_integration_config()

    table_name = "test_mosaic_append_rows"

    fqn = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}"

    connector = BigQueryConnection(BQ_PROJECT_ID)

    connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog_1_1.tif"),
        fqn,
        overwrite=True,
    )

    result = connector.get_records(fqn, 20)

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

    connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog_1_2.tif"),
        fqn,
        append=True,
    )

    result = connector.get_records(fqn, 20)

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
    check_integration_config()

    table_name = "test_mosaic_custom_band_column_1"

    fqn = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}"

    connector = BigQueryConnection(BQ_PROJECT_ID)

    connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        fqn,
        overwrite=True,
        bands_info=[(1, "customband")],
    )

    result = connector.get_records(fqn, 20)

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
    check_integration_config()

    table_name = "test_mosaic_multiple_default_bands"

    fqn = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}"

    connector = BigQueryConnection(BQ_PROJECT_ID)

    connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        fqn,
        overwrite=True,
        bands_info=[(1, None), (2, None)],
    )

    result = connector.get_records(fqn, 20)

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
    check_integration_config()

    table_name = "test_mosaic_multiple_custom_bands"

    fqn = f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}"

    connector = BigQueryConnection(BQ_PROJECT_ID)

    connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        fqn,
        overwrite=True,
        bands_info=[(1, "custom_band_1"), (2, "custom_band_2")],
    )

    result = connector.get_records(fqn, 20)

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
def test_rasterio_to_table_wrong_band_name_metadata(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockBigQueryConnection()

    with pytest.raises(IOError):
        connector.upload_raster(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}",
            overwrite=True,
            bands_info=[(1, "metadata"), (2, "custom_band_2")],
        )


@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=False)
def test_rasterio_to_table_wrong_band_name_block(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockBigQueryConnection()

    with pytest.raises(IOError):
        connector.upload_raster(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}",
            overwrite=True,
            bands_info=[(1, "block"), (2, "custom_band_2")],
        )


@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_exists",
    return_value=False,
)
@patch("raster_loader.io.bigquery.BigQueryConnection.update_labels", return_value=None)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=False)
def test_rasterio_to_table(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockBigQueryConnection()

    success = connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}",
    )
    assert success


@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_exists",
    return_value=True,
)
@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_is_empty",
    return_value=True,
)
@patch("raster_loader.io.bigquery.BigQueryConnection.delete_table", return_value=None)
@patch("raster_loader.io.common.rasterio_windows_to_records", return_value={})
@patch("raster_loader.io.common.rasterio_metadata", return_value={})
@patch("raster_loader.io.common.get_number_of_blocks", return_value=1)
@patch("raster_loader.io.bigquery.BigQueryConnection.write_metadata", return_value=None)
@patch("raster_loader.io.bigquery.BigQueryConnection.update_labels", return_value=None)
def test_rasterio_to_table_overwrite(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockBigQueryConnection()

    success = connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}",
        overwrite=True,
    )
    assert success


@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_exists",
    return_value=True,
)
@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_is_empty",
    return_value=False,
)
@patch("raster_loader.io.bigquery.BigQueryConnection.delete_table", return_value=None)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=True)
@patch(
    "raster_loader.io.bigquery.BigQueryConnection.get_metadata",
    return_value={
        "bounds": [0, 0, 0, 0],
        "block_resolution": 5,
        "nodata": 255,
        "block_width": 256,
        "block_height": 256,
        "bands": [
            {
                "type": "uint8",
                "name": "band_1",
                "colorinterp": "red",
                "stats": {
                    "min": 0.0,
                    "max": 255.0,
                    "mean": 28.66073989868164,
                    "stddev": 41.5693439511935,
                    "count": 100000,
                    "sum": 2866073.989868164,
                    "sum_squares": 1e15,
                },
                "nodata": "255",
            }
        ],
        "num_blocks": 1,
        "num_pixels": 1,
    },
)
@patch("raster_loader.io.bigquery.BigQueryConnection.update_labels", return_value=None)
def test_rasterio_to_table_is_not_empty_append(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockBigQueryConnection()

    success = connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}",
    )
    assert success


@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_exists",
    return_value=True,
)
@patch("raster_loader.io.bigquery.BigQueryConnection.delete_table", return_value=None)
@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_is_empty",
    return_value=False,
)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=False)
def test_rasterio_to_table_is_not_empty_dont_append(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockBigQueryConnection()

    with pytest.raises(SystemExit):
        connector.upload_raster(
            os.path.join(fixtures_dir, "mosaic.tif"),
            f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}",
        )


@patch(
    "raster_loader.io.bigquery.BigQueryConnection.upload_records",
    side_effect=Exception(),
)
@patch("raster_loader.io.bigquery.BigQueryConnection.delete_table", return_value=True)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=True)
def test_rasterio_to_table_uploading_error(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockBigQueryConnection()

    with pytest.raises(IOError):
        connector.upload_raster(
            os.path.join(fixtures_dir, "mosaic.tif"),
            f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}",
        )


@patch(
    "raster_loader.io.bigquery.BigQueryConnection.upload_records",
    side_effect=KeyboardInterrupt(),
)
@patch("raster_loader.io.bigquery.BigQueryConnection.delete_table", return_value=True)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=True)
@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_exists",
    return_value=False,
)
def test_rasterio_to_table_keyboard_interrupt(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockBigQueryConnection()

    with pytest.raises(KeyboardInterrupt):
        connector.upload_raster(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}",
        )


@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_exists",
    return_value=False,
)
@patch("raster_loader.io.bigquery.BigQueryConnection.update_labels", return_value=None)
def test_rasterio_to_table_with_chunk_size(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockBigQueryConnection()

    success = connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}",
        chunk_size=10000,
    )

    assert success


@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_exists",
    return_value=False,
)
@patch("raster_loader.io.bigquery.BigQueryConnection.update_labels", return_value=None)
def test_rasterio_to_table_with_one_chunk_size(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockBigQueryConnection()

    success = connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}",
        chunk_size=1,
    )

    assert success


@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_exists",
    return_value=False,
)
def test_rasterio_to_table_invalid_raster(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockBigQueryConnection()

    with pytest.raises(OSError):
        connector.upload_raster(
            os.path.join(fixtures_dir, "mosaic.tif"),
            f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}",
            chunk_size=10000,
        )


@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_exists",
    return_value=True,
)
@patch("raster_loader.io.bigquery.BigQueryConnection.delete_table", return_value=None)
@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_is_empty",
    return_value=False,
)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=True)
@patch(
    "raster_loader.io.bigquery.BigQueryConnection.get_metadata",
    return_value={
        "bounds": [0, 0, 0, 0],
        "block_resolution": 5,
        "nodata": 255,
        "block_width": 256,
        "block_height": 256,
        "bands": [
            {
                "type": "uint8",
                "name": "band_1",
                "colorinterp": "red",
                "stats": {
                    "min": 0.0,
                    "max": 255.0,
                    "mean": 28.66073989868164,
                    "stddev": 41.5693439511935,
                    "count": 100000,
                    "sum": 2866073.989868164,
                    "sum_squares": 1e15,
                },
                "nodata": "255",
            }
        ],
        "num_blocks": 1,
        "num_pixels": 1,
    },
)
@patch("raster_loader.io.bigquery.BigQueryConnection.update_labels", return_value=None)
def test_rasterio_to_bigquery_valid_raster(*args, **kwargs):
    table_name = "test_mosaic_valid_raster".upper()
    connector = mocks.MockBigQueryConnection()
    success = connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}",
    )
    assert success


@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_exists",
    return_value=True,
)
@patch("raster_loader.io.bigquery.BigQueryConnection.delete_table", return_value=None)
@patch(
    "raster_loader.io.bigquery.BigQueryConnection.check_if_table_is_empty",
    return_value=False,
)
@patch("raster_loader.io.bigquery.ask_yes_no_question", return_value=True)
@patch(
    "raster_loader.io.bigquery.BigQueryConnection.get_metadata",
    return_value={"bounds": [0, 0, 0, 0], "block_resolution": 1},
)
def test_append_with_different_resolution(*args, **kwargs):
    table_name = "test_different_resolution"
    connector = mocks.MockBigQueryConnection()
    with pytest.raises(OSError):
        connector.upload_raster(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            f"{BQ_PROJECT_ID}.{BQ_DATASET_ID}.{table_name}",
        )


def test_get_labels(*args, **kwargs):
    connector = mocks.MockBigQueryConnection()

    cases = {
        "": {"raster_loader": ""},
        "0.1.0": {"raster_loader": "0_1_0"},
        "0.1.0 something": {"raster_loader": "0_1_0_something"},
        "0.1.0+17$g1d1f3a3H": {"raster_loader": "0_1_0_17_g1d1f3a3h"},
    }
    for version, expected_labels in cases.items():
        assert connector.get_labels(version) == expected_labels
