import os
from unittest.mock import patch

from click.testing import CliRunner
import pandas as pd

from raster_loader.cli import main


here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fixtures = os.path.join(here, "fixtures")
tiff = os.path.join(fixtures, "mosaic_cog.tif")


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.upload_raster", return_value=None
)
@patch("raster_loader.io.snowflake.SnowflakeConnection.__init__", return_value=None)
def test_snowflake_upload(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "snowflake",
            "upload",
            "--file_path",
            f"{tiff}",
            "--database",
            "database",
            "--schema",
            "schema",
            "--table",
            "table",
            "--account",
            "account",
            "--username",
            "username",
            "--password",
            "password",
            "--chunk_size",
            1,
            "--band",
            1,
        ],
    )
    assert result.exit_code == 0


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.upload_raster", return_value=None
)
@patch("raster_loader.io.snowflake.SnowflakeConnection.__init__", return_value=None)
def test_bigquery_upload_multiple_bands(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "snowflake",
            "upload",
            "--file_path",
            f"{tiff}",
            "--database",
            "database",
            "--schema",
            "schema",
            "--table",
            "table",
            "--account",
            "account",
            "--username",
            "username",
            "--password",
            "password",
            "--chunk_size",
            1,
            "--band",
            1,
            "--band",
            2,
        ],
    )
    assert result.exit_code == 0


def test_snowflake_fail_upload_multiple_bands_misaligned_with_band_names(
    *args, **kwargs
):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "snowflake",
            "upload",
            "--file_path",
            f"{tiff}",
            "--database",
            "database",
            "--schema",
            "schema",
            "--table",
            "table",
            "--account",
            "account",
            "--username",
            "username",
            "--password",
            "password",
            "--chunk_size",
            1,
            "--band",
            1,
            "--band_name",
            "band_1",
            "--band",
            2,
        ],
    )
    assert result.exit_code == 1

    assert "The number of bands must equal the number of band names." in result.output


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.upload_raster", return_value=None
)
@patch("raster_loader.io.snowflake.SnowflakeConnection.__init__", return_value=None)
def test_snowflake_upload_multiple_bands_aligned_with_band_names(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "snowflake",
            "upload",
            "--file_path",
            f"{tiff}",
            "--database",
            "database",
            "--schema",
            "schema",
            "--table",
            "table",
            "--account",
            "account",
            "--username",
            "username",
            "--password",
            "password",
            "--chunk_size",
            1,
            "--band",
            1,
            "--band_name",
            "band_1",
            "--band_name",
            "band_2",
            "--band",
            2,
        ],
    )
    assert result.exit_code == 0


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.upload_raster", return_value=None
)
@patch("raster_loader.io.snowflake.SnowflakeConnection.__init__", return_value=None)
def test_snowflake_upload_no_table_name(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "snowflake",
            "upload",
            "--file_path",
            f"{tiff}",
            "--database",
            "database",
            "--schema",
            "schema",
            "--table",
            "table",
            "--account",
            "account",
            "--username",
            "username",
            "--password",
            "password",
            "--chunk_size",
            1,
            "--band",
            1,
        ],
    )
    assert result.exit_code == 0


@patch(
    "raster_loader.io.snowflake.SnowflakeConnection.get_records",
    return_value=pd.DataFrame.from_dict({"col_1": [1, 2], "col_2": ["a", "b"]}),
)
@patch("raster_loader.io.snowflake.SnowflakeConnection.__init__", return_value=None)
def test_snowflake_describe(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "snowflake",
            "describe",
            "--database",
            "database",
            "--schema",
            "schema",
            "--table",
            "table",
            "--account",
            "account",
            "--username",
            "username",
            "--password",
            "password",
        ],
    )
    assert result.exit_code == 0
