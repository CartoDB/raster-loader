import os
from unittest.mock import patch

from click.testing import CliRunner
import pandas as pd

from raster_loader.cli import main


here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "fixtures")
tiff = os.path.join(fixtures, "mosaic.tif")


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
            "--input_crs",
            "4326",
            "--band",
            1,
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
            "--input_crs",
            "4326",
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
