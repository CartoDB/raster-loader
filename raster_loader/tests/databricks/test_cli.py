import os
from unittest.mock import patch

from click.testing import CliRunner
import pandas as pd

from raster_loader.cli import main


here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fixtures = os.path.join(here, "fixtures")
tiff = os.path.join(fixtures, "mosaic_cog.tif")


@patch(
    "raster_loader.io.databricks.DatabricksConnection.upload_raster", return_value=None
)
@patch("raster_loader.io.databricks.DatabricksConnection.__init__", return_value=None)
def test_databricks_upload(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "databricks",
            "upload",
            "--file_path",
            f"{tiff}",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--table",
            "table",
            "--host",
            "https://databricks-host",
            "--token",
            "token",
            "--cluster-id",
            "cluster-1234",
            "--chunk_size",
            1,
            "--band",
            1,
        ],
    )
    print(result.output)
    assert result.exit_code == 0


@patch(
    "raster_loader.io.databricks.DatabricksConnection.upload_raster", return_value=None
)
@patch("raster_loader.io.databricks.DatabricksConnection.__init__", return_value=None)
def test_databricks_file_path_or_url_check(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "databricks",
            "upload",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--host",
            "https://databricks-host",
            "--token",
            "token",
            "--cluster-id",
            "cluster-1234",
            "--chunk_size",
            1,
            "--band",
            1,
        ],
    )
    assert result.exit_code == 1
    assert "Error: Need either a --file_path or --file_url" in result.output

    result = runner.invoke(
        main,
        [
            "databricks",
            "upload",
            "--file_path",
            f"{tiff}",
            "--file_url",
            "http://example.com/raster.tif",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--host",
            "https://databricks-host",
            "--token",
            "token",
            "--cluster-id",
            "cluster-1234",
            "--chunk_size",
            1,
            "--band",
            1,
        ],
    )
    assert result.exit_code == 1
    assert "Only one of --file_path or --file_url must be provided" in result.output


@patch(
    "raster_loader.io.databricks.DatabricksConnection.upload_raster", return_value=None
)
@patch("raster_loader.io.databricks.DatabricksConnection.__init__", return_value=None)
def test_databricks_upload_multiple_bands(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "databricks",
            "upload",
            "--file_path",
            f"{tiff}",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--host",
            "https://databricks-host",
            "--token",
            "token",
            "--cluster-id",
            "cluster-1234",
            "--chunk_size",
            1,
            "--band",
            1,
            "--band",
            2,
        ],
    )
    assert result.exit_code == 0


def test_databricks_fail_upload_multiple_bands_misaligned_with_band_names(
    *args, **kwargs
):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "databricks",
            "upload",
            "--file_path",
            f"{tiff}",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--host",
            "https://databricks-host",
            "--token",
            "token",
            "--cluster-id",
            "cluster-1234",
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
    assert "Error: Must supply the same number of band_names as bands" in result.output


@patch(
    "raster_loader.io.databricks.DatabricksConnection.upload_raster", return_value=None
)
@patch("raster_loader.io.databricks.DatabricksConnection.__init__", return_value=None)
def test_databricks_upload_multiple_bands_aligned_with_band_names(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "databricks",
            "upload",
            "--file_path",
            f"{tiff}",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--host",
            "https://databricks-host",
            "--token",
            "token",
            "--cluster-id",
            "cluster-1234",
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
    "raster_loader.io.databricks.DatabricksConnection.upload_raster", return_value=None
)
@patch("raster_loader.io.databricks.DatabricksConnection.__init__", return_value=None)
def test_databricks_upload_no_table_name(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "databricks",
            "upload",
            "--file_path",
            f"{tiff}",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--host",
            "https://databricks-host",
            "--token",
            "token",
            "--cluster-id",
            "cluster-1234",
        ],
    )
    assert result.exit_code == 0
    assert "Table: mosaic_cog_band__1___" in result.output
