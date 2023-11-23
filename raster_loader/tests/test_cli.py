import os
from unittest.mock import patch

from click.testing import CliRunner
import pandas as pd

from raster_loader.cli import main


here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "fixtures")
tiff = os.path.join(fixtures, "mosaic_cog.tif")


@patch("raster_loader.io.rasterio_to_bigquery", return_value=None)
def test_bigquery_upload(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "bigquery",
            "upload",
            "--file_path",
            f"{tiff}",
            "--project",
            "project",
            "--dataset",
            "dataset",
            "--table",
            "table",
            "--chunk_size",
            1,
            "--band",
            1,
            "--test",
        ],
    )
    assert result.exit_code == 0


@patch("raster_loader.io.rasterio_to_bigquery", return_value=None)
def test_bigquery_upload_multiple_bands(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "bigquery",
            "upload",
            "--file_path",
            f"{tiff}",
            "--project",
            "project",
            "--dataset",
            "dataset",
            "--table",
            "table",
            "--chunk_size",
            1,
            "--band",
            1,
            "--band",
            2,
            "--test",
        ],
    )
    assert result.exit_code == 0


def test_bigquery_fail_upload_multiple_bands_misaligned_with_band_names(
    *args, **kwargs
):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "bigquery",
            "upload",
            "--file_path",
            f"{tiff}",
            "--project",
            "project",
            "--dataset",
            "dataset",
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
            "--test",
        ],
    )
    assert result.exit_code == 1

    assert "The number of bands must equal the number of band names." in result.output


@patch("raster_loader.io.rasterio_to_bigquery", return_value=None)
def test_bigquery_upload_multiple_bands_aligned_with_band_names(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "bigquery",
            "upload",
            "--file_path",
            f"{tiff}",
            "--project",
            "project",
            "--dataset",
            "dataset",
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
            "--test",
        ],
    )
    assert result.exit_code == 0


@patch("raster_loader.io.rasterio_to_bigquery", return_value=None)
def test_bigquery_upload_no_table_name(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "bigquery",
            "upload",
            "--file_path",
            f"{tiff}",
            "--project",
            "project",
            "--dataset",
            "dataset",
            "--chunk_size",
            1,
            "--band",
            1,
            "--test",
        ],
    )
    assert result.exit_code == 0


@patch(
    "raster_loader.io.bigquery_to_records",
    return_value=pd.DataFrame.from_dict({"col_1": [1, 2], "col_2": ["a", "b"]}),
)
def test_bigquery_describe(mocker):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "bigquery",
            "describe",
            "--project",
            "project",
            "--dataset",
            "dataset",
            "--table",
            "table",
        ],
    )
    assert result.exit_code == 0


def test_info(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(main, ["info"])

    assert result.exit_code == 0
    assert "Raster Loader version" in result.output
    assert "Python version" in result.output
    assert "Platform" in result.output
    assert "System version" in result.output
    assert "Machine" in result.output
    assert "Processor" in result.output
    assert "Architecture" in result.output
