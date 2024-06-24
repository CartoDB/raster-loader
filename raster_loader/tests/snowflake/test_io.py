import math
import os
import sys
import json
from unittest.mock import patch

import pandas as pd
import pytest

from raster_loader.tests import mocks
from raster_loader.io.snowflake import SnowflakeConnection


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


SF_ACCOUNT = os.environ.get("SF_ACCOUNT")
SF_USERNAME = os.environ.get("SF_USERNAME")
SF_PASSWORD = os.environ.get("SF_PASSWORD")
SF_DATABASE = os.environ.get("SF_DATABASE")
SF_SCHEMA = os.environ.get("SF_SCHEMA")
SF_ROLE = os.environ.get("SF_ROLE")


def check_integration_config():
    if not all([SF_ACCOUNT, SF_USERNAME, SF_PASSWORD, SF_DATABASE, SF_SCHEMA, SF_ROLE]):
        raise Exception(
            "You need to copy tests/.env.sample to test/.env and set your configuration"
            "before running the tests"
        )


@pytest.mark.integration_test
def test_rasterio_to_snowflake_with_raster_default_band_name():
    check_integration_config()

    table_name = "test_mosaic_1".upper()
    fqn = f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}"

    connector = SnowflakeConnection(
        username=SF_USERNAME,
        password=SF_PASSWORD,
        account=SF_ACCOUNT,
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        role=SF_ROLE,
        token=None,
    )

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

    assert sorted(result.columns) == sorted(
        [col.upper() for col in expected_dataframe.columns]
    )
    assert sorted(
        list(result.BLOCK), key=lambda x: x if x is not None else -math.inf
    ) == sorted(
        list(expected_dataframe.block), key=lambda x: x if x is not None else -math.inf
    )
    assert sorted(
        [x.upper() for x in list(result.METADATA) if x is not None]
    ) == sorted(
        [x.upper() for x in list(expected_dataframe.metadata) if x is not None],
    )
    assert sorted(
        list(result.BAND_1), key=lambda x: x if x is not None else b""
    ) == sorted(
        list(expected_dataframe.band_1), key=lambda x: x if x is not None else b""
    )


@pytest.mark.integration_test
def test_rasterio_to_snowflake_appending_rows():
    check_integration_config()

    table_name = "test_mosaic_append_rows".upper()
    fqn = f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}"

    connector = SnowflakeConnection(
        username=SF_USERNAME,
        password=SF_PASSWORD,
        account=SF_ACCOUNT,
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        role=SF_ROLE,
        token=None,
    )

    connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog_1_1.tif"),
        fqn,
        overwrite=True,
    )

    result = connector.get_records(fqn, 20)

    metadata = json.loads([x for x in list(result.METADATA) if x][0])

    assert metadata == {
        "pixel_resolution": 13,
        "block_resolution": 5,
        "minresolution": 5,
        "maxresolution": 5,
        "nodata": None,
        "bands": [{"type": "uint8", "name": "BAND_1"}],
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

    metadata = json.loads([x for x in list(result.METADATA) if x][0])

    assert metadata == {
        "bands": [{"name": "BAND_1", "type": "uint8"}],
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
def test_rasterio_to_snowflake_with_raster_custom_band_column():
    check_integration_config()

    table_name = "test_mosaic_custom_band_column_1".upper()
    fqn = f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}"

    connector = SnowflakeConnection(
        username=SF_USERNAME,
        password=SF_PASSWORD,
        account=SF_ACCOUNT,
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        role=SF_ROLE,
        token=None,
    )

    connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        fqn,
        overwrite=True,
        bands_info=[(1, "customband")],
    )

    result = connector.get_records(fqn, 20)

    # sort value because return query can vary the order of rows
    result = result.sort_values("BLOCK")

    expected_dataframe = pd.read_pickle(
        os.path.join(fixtures_dir, "expected_custom_column.pkl")
    )
    expected_dataframe = expected_dataframe.sort_values("block")

    assert sorted(result.columns) == sorted(
        [col.upper() for col in expected_dataframe.columns]
    )
    assert sorted(
        list(result.BLOCK), key=lambda x: x if x is not None else -math.inf
    ) == sorted(
        list(expected_dataframe.block), key=lambda x: x if x is not None else -math.inf
    )
    assert sorted(
        [x.upper() for x in list(result.METADATA) if x is not None]
    ) == sorted(
        [x.upper() for x in list(expected_dataframe.metadata) if x is not None],
    )
    assert sorted(
        list(result.CUSTOMBAND), key=lambda x: x if x is not None else b""
    ) == sorted(
        list(expected_dataframe.customband), key=lambda x: x if x is not None else b""
    )


@pytest.mark.integration_test
def test_rasterio_to_snowflake_with_raster_multiple_default():
    check_integration_config()

    table_name = "test_mosaic_multiple_default_bands".upper()
    fqn = f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}"

    connector = SnowflakeConnection(
        username=SF_USERNAME,
        password=SF_PASSWORD,
        account=SF_ACCOUNT,
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        role=SF_ROLE,
        token=None,
    )

    connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        fqn,
        overwrite=True,
        bands_info=[(1, None), (2, None)],
    )

    result = connector.get_records(fqn, 20)

    # sort value because return query can vary the order of rows
    result = result.sort_values("BLOCK")

    expected_dataframe = pd.read_pickle(
        os.path.join(fixtures_dir, "expected_multiple_column.pkl")
    )
    expected_dataframe = expected_dataframe.sort_values("block")

    assert sorted(result.columns) == sorted(
        [col.upper() for col in expected_dataframe.columns]
    )
    assert sorted(
        list(result.BLOCK), key=lambda x: x if x is not None else -math.inf
    ) == sorted(
        list(expected_dataframe.block), key=lambda x: x if x is not None else -math.inf
    )
    assert sorted(
        [x.upper() for x in list(result.METADATA) if x is not None]
    ) == sorted(
        [x.upper() for x in list(expected_dataframe.metadata) if x is not None],
    )
    assert sorted(
        list(result.BAND_1), key=lambda x: x if x is not None else b""
    ) == sorted(
        list(expected_dataframe.band_1), key=lambda x: x if x is not None else b""
    )
    assert sorted(
        list(result.BAND_2), key=lambda x: x if x is not None else b""
    ) == sorted(
        list(expected_dataframe.band_2), key=lambda x: x if x is not None else b""
    )


@pytest.mark.integration_test
def test_rasterio_to_snowflake_with_raster_multiple_custom():
    check_integration_config()

    table_name = "test_mosaic_multiple_custom_bands".upper()
    fqn = f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}"

    connector = SnowflakeConnection(
        username=SF_USERNAME,
        password=SF_PASSWORD,
        account=SF_ACCOUNT,
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        role=SF_ROLE,
        token=None,
    )

    connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        fqn,
        overwrite=True,
        bands_info=[(1, "custom_band_1"), (2, "custom_band_2")],
    )

    result = connector.get_records(fqn, 20)

    # sort value because return query can vary the order of rows
    result = result.sort_values("BLOCK")

    expected_dataframe = pd.read_pickle(
        os.path.join(fixtures_dir, "expected_custom_multiple_column.pkl")
    )
    expected_dataframe = expected_dataframe.sort_values("block")

    assert sorted(result.columns) == sorted(
        [col.upper() for col in expected_dataframe.columns]
    )
    assert sorted(
        list(result.BLOCK), key=lambda x: x if x is not None else -math.inf
    ) == sorted(
        list(expected_dataframe.block), key=lambda x: x if x is not None else -math.inf
    )
    assert sorted(
        [x.upper() for x in list(result.METADATA) if x is not None]
    ) == sorted(
        [x.upper() for x in list(expected_dataframe.metadata) if x is not None],
    )
    assert sorted(
        list(result.CUSTOM_BAND_1), key=lambda x: x if x is not None else b""
    ) == sorted(
        list(expected_dataframe.custom_band_1),
        key=lambda x: x if x is not None else b"",
    )
    assert sorted(
        list(result.CUSTOM_BAND_2), key=lambda x: x if x is not None else b""
    ) == sorted(
        list(expected_dataframe.custom_band_2),
        key=lambda x: x if x is not None else b"",
    )


@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=False)
def test_rasterio_to_table_wrong_band_name_metadata(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockSnowflakeConnection()

    with pytest.raises(IOError):
        connector.upload_raster(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}",
            overwrite=True,
            bands_info=[(1, "metadata"), (2, "custom_band_2")],
        )


@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=False)
def test_rasterio_to_table_wrong_band_name_block(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockSnowflakeConnection()

    with pytest.raises(IOError):
        connector.upload_raster(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}",
            overwrite=True,
            bands_info=[(1, "block"), (2, "custom_band_2")],
        )


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_exists",
    return_value=False,
)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=False)
@patch("raster_loader.io.snowflake.write_pandas", return_value=[True])
def test_rasterio_to_table(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockSnowflakeConnection()

    success = connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}",
    )
    assert success


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_exists",
    return_value=True,
)
@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_is_empty",
    return_value=True,
)
@patch("raster_loader.io.snowflake.SnowflakeConnection.delete_table", return_value=None)
@patch("raster_loader.io.common.rasterio_windows_to_records", return_value={})
@patch("raster_loader.io.common.rasterio_metadata", return_value={})
@patch("raster_loader.io.common.get_number_of_blocks", return_value=1)
@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.write_metadata", return_value=None
)
@patch("raster_loader.io.snowflake.write_pandas", return_value=[True])
def test_rasterio_to_table_overwrite(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockSnowflakeConnection()

    success = connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}",
        overwrite=True,
    )
    assert success


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_exists",
    return_value=True,
)
@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_is_empty",
    return_value=False,
)
@patch("raster_loader.io.snowflake.SnowflakeConnection.delete_table", return_value=None)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=True)
@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.get_metadata",
    return_value={
        "bounds": [0, 0, 0, 0],
        "block_resolution": 5,
        "nodata": 255,
        "block_width": 256,
        "block_height": 256,
        "bands": [
            {
                "type": "uint8",
                "name": "BAND_1",
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
@patch("raster_loader.io.snowflake.write_pandas", return_value=[True])
def test_rasterio_to_table_is_not_empty_append(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockSnowflakeConnection()

    success = connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}",
    )
    assert success


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_exists",
    return_value=True,
)
@patch("raster_loader.io.snowflake.SnowflakeConnection.delete_table", return_value=None)
@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_is_empty",
    return_value=False,
)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=False)
def test_rasterio_to_table_is_not_empty_dont_append(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockSnowflakeConnection()

    with pytest.raises(SystemExit):
        connector.upload_raster(
            os.path.join(fixtures_dir, "mosaic.tif"),
            f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}",
        )


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.upload_records",
    side_effect=Exception(),
)
@patch("raster_loader.io.snowflake.SnowflakeConnection.delete_table", return_value=True)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=True)
def test_rasterio_to_table_uploading_error(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockSnowflakeConnection()

    with pytest.raises(IOError):
        connector.upload_raster(
            os.path.join(fixtures_dir, "mosaic.tif"),
            f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}",
        )


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.upload_records",
    side_effect=KeyboardInterrupt(),
)
@patch("raster_loader.io.snowflake.SnowflakeConnection.delete_table", return_value=True)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=True)
@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_exists",
    return_value=False,
)
def test_rasterio_to_table_keyboard_interrupt(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockSnowflakeConnection()

    with pytest.raises(KeyboardInterrupt):
        connector.upload_raster(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}",
        )


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_exists",
    return_value=False,
)
@patch("raster_loader.io.snowflake.write_pandas", return_value=[True])
def test_rasterio_to_table_with_chunk_size(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockSnowflakeConnection()

    success = connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}",
        chunk_size=10000,
    )

    assert success


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_exists",
    return_value=False,
)
@patch("raster_loader.io.snowflake.write_pandas", return_value=[True])
def test_rasterio_to_table_with_one_chunk_size(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockSnowflakeConnection()

    success = connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}",
        chunk_size=1,
    )

    assert success


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_exists",
    return_value=False,
)
def test_rasterio_to_table_invalid_raster(*args, **kwargs):
    table_name = "test_mosaic_custom_band_column_1"
    connector = mocks.MockSnowflakeConnection()

    with pytest.raises(OSError):
        connector.upload_raster(
            os.path.join(fixtures_dir, "mosaic.tif"),
            f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}",
            chunk_size=10000,
        )


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_exists",
    return_value=True,
)
@patch("raster_loader.io.snowflake.SnowflakeConnection.delete_table", return_value=None)
@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_is_empty",
    return_value=False,
)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=True)
@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.get_metadata",
    return_value={
        "bounds": [0, 0, 0, 0],
        "block_resolution": 5,
        "nodata": 255,
        "block_width": 256,
        "block_height": 256,
        "bands": [
            {
                "type": "uint8",
                "name": "BAND_1",
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
@patch("raster_loader.io.snowflake.write_pandas", return_value=[True])
def test_rasterio_to_snowflake_valid_raster(*args, **kwargs):
    table_name = "test_mosaic_valid_raster".upper()
    connector = mocks.MockSnowflakeConnection()
    success = connector.upload_raster(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}",
    )
    assert success


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_exists",
    return_value=True,
)
@patch("raster_loader.io.snowflake.SnowflakeConnection.delete_table", return_value=None)
@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.check_if_table_is_empty",
    return_value=False,
)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=True)
@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.get_metadata",
    return_value={"bounds": [0, 0, 0, 0], "block_resolution": 1},
)
def test_append_with_different_resolution(*args, **kwargs):
    table_name = "test_different_resolution".upper()
    connector = mocks.MockSnowflakeConnection()
    with pytest.raises(OSError):
        connector.upload_raster(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            f"{SF_DATABASE}.{SF_SCHEMA}.{table_name}",
        )
