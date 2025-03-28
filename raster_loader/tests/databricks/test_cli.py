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
            "--server-hostname",
            "test.cloud.databricks.com",
            "--token",
            "test-token",
            "--cluster-id",
            "test-cluster-id",
            "--file_path",
            f"{tiff}",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--table",
            "table",
            "--chunk_size",
            1,
            "--band",
            1,
        ],
    )
    assert result.exit_code == 0


@patch(
    "raster_loader.io.databricks.DatabricksConnection.upload_raster", return_value=None
)
@patch("raster_loader.io.databricks.DatabricksConnection.__init__", return_value=None)
def test_databricks_upload_with_basic_stats(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "databricks",
            "upload",
            "--server-hostname",
            "test.cloud.databricks.com",
            "--token",
            "test-token",
            "--cluster-id",
            "test-cluster-id",
            "--file_path",
            f"{tiff}",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--table",
            "table",
            "--chunk_size",
            1,
            "--band",
            1,
            "--basic_stats",
        ],
    )
    assert result.exit_code == 0


@patch(
    "raster_loader.io.databricks.DatabricksConnection.upload_raster", return_value=None
)
@patch("raster_loader.io.databricks.DatabricksConnection.__init__", return_value=None)
def test_databricks_upload_with_all_stats(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "databricks",
            "upload",
            "--server-hostname",
            "test.cloud.databricks.com",
            "--token",
            "test-token",
            "--cluster-id",
            "test-cluster-id",
            "--file_path",
            f"{tiff}",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--table",
            "table",
            "--chunk_size",
            1,
            "--band",
            1,
        ],
    )
    assert result.exit_code == 0


@patch(
    "raster_loader.io.databricks.DatabricksConnection.upload_raster", return_value=None
)
@patch("raster_loader.io.databricks.DatabricksConnection.__init__", return_value=None)
def test_databricks_upload_with_exact_stats(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "databricks",
            "upload",
            "--server-hostname",
            "test.cloud.databricks.com",
            "--token",
            "test-token",
            "--cluster-id",
            "test-cluster-id",
            "--file_path",
            f"{tiff}",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--table",
            "table",
            "--chunk_size",
            1,
            "--band",
            1,
            "--exact_stats",
        ],
    )
    assert result.exit_code == 0


@patch("raster_loader.io.databricks.DatabricksConnection.__init__", return_value=None)
def test_databricks_credentials_validation(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "databricks",
            "upload",
            "--server-hostname",
            "test.cloud.databricks.com",
            "--token",
            "test-token",
            "--file_path",
            f"{tiff}",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--table",
            "table",
            "--chunk_size",
            1,
            "--band",
            1,
        ],
    )
    assert result.exit_code == 2
    assert "Missing option '--cluster-id'" in result.output


@patch("raster_loader.io.databricks.DatabricksConnection.__init__", return_value=None)
def test_databricks_file_path_or_url_check(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "databricks",
            "upload",
            "--server-hostname",
            "test.cloud.databricks.com",
            "--token",
            "test-token",
            "--cluster-id",
            "test-cluster-id",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--table",
            "table",
            "--chunk_size",
            1,
            "--band",
            1,
        ],
    )
    assert result.exit_code == 1
    assert "Either --file_path or --file_url must be provided" in result.output

    result = runner.invoke(
        main,
        [
            "databricks",
            "upload",
            "--file_path",
            f"{tiff}",
            "--file_url",
            "http://example.com/raster.tif",
            "--server-hostname",
            "test.cloud.databricks.com",
            "--token",
            "test-token",
            "--cluster-id",
            "test-cluster-id",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--table",
            "table",
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
            "--server-hostname",
            "test.cloud.databricks.com",
            "--token",
            "test-token",
            "--cluster-id",
            "test-cluster-id",
            "--file_path",
            f"{tiff}",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--table",
            "table",
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
            "--server-hostname",
            "test.cloud.databricks.com",
            "--token",
            "test-token",
            "--cluster-id",
            "test-cluster-id",
            "--file_path",
            f"{tiff}",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--table",
            "table",
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
            "--server-hostname",
            "test.cloud.databricks.com",
            "--token",
            "test-token",
            "--cluster-id",
            "test-cluster-id",
            "--file_path",
            f"{tiff}",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--table",
            "table",
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
            "--server-hostname",
            "test.cloud.databricks.com",
            "--token",
            "test-token",
            "--cluster-id",
            "test-cluster-id",
            "--file_path",
            f"{tiff}",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--chunk_size",
            1,
            "--band",
            1,
        ],
    )
    assert result.exit_code == 0
    assert "Table: mosaic_cog_band__1___" in result.output


@patch(
    "raster_loader.io.databricks.DatabricksConnection.execute_to_dataframe",
    return_value=pd.DataFrame.from_dict({"col_1": [1, 2], "col_2": ["a", "b"]}),
)
@patch("raster_loader.io.databricks.DatabricksConnection.__init__", return_value=None)
def test_databricks_describe(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "databricks",
            "describe",
            "--server-hostname",
            "test.cloud.databricks.com",
            "--token",
            "test-token",
            "--cluster-id",
            "test-cluster-id",
            "--catalog",
            "catalog",
            "--schema",
            "schema",
            "--table",
            "table",
        ],
    )
    assert result.exit_code == 0
