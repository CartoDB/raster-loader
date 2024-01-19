import math
import os
import sys
import json
from unittest.mock import patch

import pandas as pd
import pytest

from raster_loader import io
from raster_loader.tests import mocks

import snowflake.connector as connector

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


def check_integration_config():
    if not all([SF_ACCOUNT, SF_USERNAME, SF_PASSWORD, SF_DATABASE, SF_SCHEMA]):
        raise Exception(
            "You need to copy tests/.env.sample to test/.env and set your configuration"
            "before running the tests"
        )


@pytest.mark.integration_test
def test_rasterio_to_snowflake_with_raster_default_band_name():
    from raster_loader.io.snowflake import rasterio_to_table
    from raster_loader.io.snowflake import table_to_records

    check_integration_config()

    client = connector.connect(
        user=SF_USERNAME,
        password=SF_PASSWORD,
        account=SF_ACCOUNT,
        database=SF_DATABASE,
        schema=SF_SCHEMA,
    )

    table_name = "test_mosaic_1".upper()

    rasterio_to_table(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        table=table_name,
        client=client,
        overwrite=True,
    )

    result = table_to_records(
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        table=table_name,
        client=client,
    )

    expected_dataframe = pd.read_pickle(
        os.path.join(fixtures_dir, "expected_default_column.pkl")
    )

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
    from raster_loader.io.snowflake import (
        rasterio_to_table,
        delete_table,
        table_to_records,
    )

    check_integration_config()

    client = connector.connect(
        user=SF_USERNAME,
        password=SF_PASSWORD,
        account=SF_ACCOUNT,
        database=SF_DATABASE,
        schema=SF_SCHEMA,
    )

    table_name = "test_mosaic_append_rows".upper()

    delete_table(
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        table=table_name,
        client=client,
    )

    rasterio_to_table(
        os.path.join(fixtures_dir, "mosaic_cog_1_1.tif"),
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        table=table_name,
        client=client,
        overwrite=True,
    )

    result = table_to_records(
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        table=table_name,
        client=client,
    )

    metadata = json.loads([x for x in list(result.METADATA) if x][0])

    assert metadata == {
        "block_resolution": 5,
        "pixel_resolution": 13,
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

    rasterio_to_table(
        os.path.join(fixtures_dir, "mosaic_cog_1_2.tif"),
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        table=table_name,
        client=client,
        append=True,
    )

    result = table_to_records(
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        table=table_name,
        client=client,
    )

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
    from raster_loader.io.snowflake import rasterio_to_table
    from raster_loader.io.snowflake import table_to_records

    check_integration_config()

    client = connector.connect(
        user=SF_USERNAME,
        password=SF_PASSWORD,
        account=SF_ACCOUNT,
        database=SF_DATABASE,
        schema=SF_SCHEMA,
    )

    table_name = "test_mosaic_custom_band_column_1".upper()

    rasterio_to_table(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        table=table_name,
        client=client,
        overwrite=True,
        bands_info=[(1, "customband")],
    )

    result = table_to_records(
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        table=table_name,
        client=client,
    )

    expected_dataframe = pd.read_pickle(
        os.path.join(fixtures_dir, "expected_custom_column.pkl")
    )

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
    from raster_loader.io.snowflake import rasterio_to_table
    from raster_loader.io.snowflake import table_to_records

    check_integration_config()

    table_name = "test_mosaic_multiple_default_bands".upper()

    client = connector.connect(
        user=SF_USERNAME,
        password=SF_PASSWORD,
        account=SF_ACCOUNT,
        database=SF_DATABASE,
        schema=SF_SCHEMA,
    )

    rasterio_to_table(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        table=table_name,
        client=client,
        overwrite=True,
        bands_info=[(1, None), (2, None)],
    )

    result = table_to_records(
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        table=table_name,
        client=client,
    )

    expected_dataframe = pd.read_pickle(
        os.path.join(fixtures_dir, "expected_multiple_column.pkl")
    )

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
    from raster_loader.io.snowflake import rasterio_to_table
    from raster_loader.io.snowflake import table_to_records

    check_integration_config()

    table_name = "test_mosaic_multiple_custom_bands".upper()

    client = connector.connect(
        user=SF_USERNAME,
        password=SF_PASSWORD,
        account=SF_ACCOUNT,
        database=SF_DATABASE,
        schema=SF_SCHEMA,
    )

    rasterio_to_table(
        os.path.join(fixtures_dir, "mosaic_cog.tif"),
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        table=table_name,
        client=client,
        overwrite=True,
        bands_info=[(1, "custom_band_1"), (2, "custom_band_2")],
    )

    result = table_to_records(
        database=SF_DATABASE,
        schema=SF_SCHEMA,
        table=table_name,
        client=client,
    )

    expected_dataframe = pd.read_pickle(
        os.path.join(fixtures_dir, "expected_custom_multiple_column.pkl")
    )

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
def test_rasterio_to_snowflake_wrong_band_name_metadata(*args, **kwargs):
    from raster_loader.io.snowflake import rasterio_to_table

    table_name = "test_mosaic_custom_band_column_1"
    client = mocks.snowflake_client()

    with pytest.raises(IOError):
        rasterio_to_table(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            SF_DATABASE,
            SF_SCHEMA,
            table_name,
            overwrite=True,
            client=client,
            bands_info=[(1, "METADATA"), (2, "CUSTOM_BAND_2")],
        )


@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=False)
def test_rasterio_to_snowflake_wrong_band_name_block(*args, **kwargs):
    from raster_loader.io.snowflake import rasterio_to_table

    table_name = "test_mosaic_custom_band_column_1"
    client = mocks.snowflake_client()

    with pytest.raises(IOError):
        rasterio_to_table(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            SF_DATABASE,
            SF_SCHEMA,
            table_name,
            overwrite=True,
            client=client,
            bands_info=[(1, "BLOCK"), (2, "CUSTOM_BAND_2")],
        )


@patch("raster_loader.io.snowflake.check_if_table_exists", return_value=False)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=False)
@patch("raster_loader.io.snowflake.write_pandas", return_value=[True])
def test_rasterio_to_table(*args, **kwargs):
    client = mocks.snowflake_client()
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

    success = io.snowflake.rasterio_to_table(
        test_file,
        database="test",
        schema="test",
        table="test",
        client=client,
    )
    assert success


@patch("raster_loader.io.snowflake.check_if_table_exists", return_value=True)
@patch("raster_loader.io.snowflake.delete_table", return_value=None)
@patch("raster_loader.io.common.rasterio_windows_to_records", return_value={})
@patch("raster_loader.io.common.rasterio_metadata", return_value={})
@patch("raster_loader.io.common.get_number_of_blocks", return_value=1)
@patch("raster_loader.io.snowflake.write_metadata", return_value=None)
@patch("raster_loader.io.snowflake.write_pandas", return_value=[True])
def test_rasterio_to_snowflake_overwrite(*args, **kwargs):
    client = mocks.snowflake_client()
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

    success = io.snowflake.rasterio_to_table(
        test_file,
        database="test",
        schema="test",
        table="test",
        client=client,
        overwrite=True,
    )
    assert success


@patch("raster_loader.io.snowflake.check_if_table_exists", return_value=True)
@patch("raster_loader.io.snowflake.delete_table", return_value=None)
@patch("raster_loader.io.snowflake.check_if_table_is_empty", return_value=False)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=True)
@patch(
    "raster_loader.io.snowflake.get_metadata",
    return_value={
        "bounds": [0, 0, 0, 0],
        "block_resolution": 5,
        "nodata": None,
        "block_width": 256,
        "block_height": 256,
        "bands": [{"type": "uint8", "name": "BAND_1"}],
        "num_blocks": 1,
        "num_pixels": 1,
    },
)
@patch("raster_loader.io.snowflake.write_pandas", return_value=[True])
def test_rasterio_to_snowflake_table_is_not_empty_append(*args, **kwargs):
    client = mocks.snowflake_client()
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

    success = io.snowflake.rasterio_to_table(
        test_file,
        database="test",
        schema="test",
        table="test",
        client=client,
    )
    assert success


@patch("raster_loader.io.snowflake.check_if_table_exists", return_value=True)
@patch("raster_loader.io.snowflake.delete_table", return_value=None)
@patch("raster_loader.io.snowflake.check_if_table_is_empty", return_value=False)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=False)
def test_rasterio_to_snowflake_table_is_not_empty_dont_append(*args, **kwargs):
    client = mocks.snowflake_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    with pytest.raises(SystemExit):
        io.snowflake.rasterio_to_table(
            test_file,
            database="test",
            schema="test",
            table="test",
            client=client,
        )


@patch("raster_loader.io.snowflake.records_to_table", side_effect=Exception())
@patch("raster_loader.io.snowflake.delete_table", return_value=True)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=True)
def test_rasterio_to_snowflake_uploading_error(*args, **kwargs):
    client = mocks.snowflake_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    with pytest.raises(IOError):
        io.snowflake.rasterio_to_table(
            test_file,
            database="test",
            schema="test",
            table="test",
            client=client,
            chunk_size=100,
        )


@patch("raster_loader.io.snowflake.records_to_table", side_effect=KeyboardInterrupt())
@patch("raster_loader.io.snowflake.delete_table", return_value=True)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=True)
@patch("raster_loader.io.snowflake.check_if_table_exists", return_value=False)
def test_rasterio_to_snowflake_keyboard_interrupt(*args, **kwargs):
    client = mocks.snowflake_client()
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

    with pytest.raises(KeyboardInterrupt):
        io.snowflake.rasterio_to_table(
            test_file,
            database="test",
            schema="test",
            table="test",
            client=client,
            chunk_size=100,
        )


@patch("raster_loader.io.snowflake.check_if_table_exists", return_value=False)
@patch("raster_loader.io.snowflake.write_pandas", return_value=[True])
def test_rasterio_to_snowflake_with_chunk_size(*args, **kwargs):
    client = mocks.snowflake_client()
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

    success = io.snowflake.rasterio_to_table(
        test_file,
        database="test",
        schema="test",
        table="test",
        client=client,
        chunk_size=100,
    )
    assert success


@patch("raster_loader.io.snowflake.check_if_table_exists", return_value=True)
@patch("raster_loader.io.snowflake.delete_table", return_value=None)
@patch("raster_loader.io.snowflake.check_if_table_is_empty", return_value=False)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=True)
def test_rasterio_to_snowflake_invalid_raster(*args, **kwargs):
    client = mocks.snowflake_client()
    test_file = os.path.join(fixtures_dir, "mosaic.tif")

    with pytest.raises(OSError):
        io.snowflake.rasterio_to_table(
            test_file,
            database="test",
            schema="test",
            table="test",
            client=client,
        )


@patch("raster_loader.io.snowflake.check_if_table_exists", return_value=True)
@patch("raster_loader.io.snowflake.delete_table", return_value=None)
@patch("raster_loader.io.snowflake.check_if_table_is_empty", return_value=False)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=True)
@patch(
    "raster_loader.io.snowflake.get_metadata",
    return_value={
        "bounds": [0, 0, 0, 0],
        "block_resolution": 5,
        "nodata": None,
        "block_width": 256,
        "block_height": 256,
        "bands": [{"type": "uint8", "name": "BAND_1"}],
        "num_blocks": 1,
        "num_pixels": 1,
    },
)
@patch("raster_loader.io.snowflake.write_pandas", return_value=[True])
def test_rasterio_to_snowflake_valid_raster(*args, **kwargs):
    from raster_loader.io.snowflake import rasterio_to_table

    client = mocks.snowflake_client()
    test_file = os.path.join(fixtures_dir, "mosaic_cog.tif")

    success = rasterio_to_table(
        test_file,
        database="test",
        schema="test",
        table="test",
        client=client,
    )
    assert success


@patch("raster_loader.io.snowflake.check_if_table_exists", return_value=True)
@patch("raster_loader.io.snowflake.delete_table", return_value=None)
@patch("raster_loader.io.snowflake.check_if_table_is_empty", return_value=False)
@patch("raster_loader.io.snowflake.ask_yes_no_question", return_value=True)
@patch(
    "raster_loader.io.snowflake.get_metadata",
    return_value={"bounds": [0, 0, 0, 0], "block_resolution": 1},
)
def test_append_with_different_resolution(*args, **kwargs):
    from raster_loader.io.snowflake import rasterio_to_table

    client = mocks.snowflake_client()

    with pytest.raises(OSError):
        rasterio_to_table(
            os.path.join(fixtures_dir, "mosaic_cog.tif"),
            database="test",
            schema="test",
            table="test",
            overwrite=False,
            append=True,
            client=client,
        )
