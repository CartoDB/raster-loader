import os

from click.testing import CliRunner
from unittest.mock import patch

from raster_loader import RasterLoader
from raster_loader.raster_loader_cli import raster_loader_cli
from tests import mocks

FIXTURE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../tests/fixtures/mosaic.tif")
)


def test_info():
    """Test the info command."""
    runner = CliRunner()
    result = runner.invoke(raster_loader_cli, ["info"])
    print(result.output)
    assert result.exit_code == 0
    assert "Python version" in result.output
    assert "Operating system" in result.output


## Test TBD!
# @patch.object(RasterLoader, "_bigquery_client", return_value=mocks.bigquery_client())
# def test_raster(*args, **kwargs):
#     """Test the raster command."""
#     runner = CliRunner()
#     result = runner.invoke(
#         raster_loader_cli,
#         ["raster", "load", FIXTURE_FILE, "test_project.test_dataset.test_table"],
#     )
#     assert result.exit_code == 0
#     assert FIXTURE_FILE in result.output

def test_raster_invalid_filename():
    """Test whether the raster command checks for invalid filename."""
    runner = CliRunner()
    result = runner.invoke(
        raster_loader_cli,
        ["raster", "load", "invalid file name", "test_project.test_dataset.test_table"],
    )
    assert result.exit_code == 2
    assert "Error" in result.output


def test_raster_invalid_table():
    """Test whether the raster command checks for invalid tables."""
    runner = CliRunner()
    result = runner.invoke(
        raster_loader_cli, ["raster", "load", FIXTURE_FILE, "invalid_table"]
    )
    assert result.exit_code == 2
    assert "Error" in result.output


def test_raster_missing_input():
    """Test whether the raster command always requires a file and table input."""
    runner = CliRunner()
    result = runner.invoke(raster_loader_cli, ["raster", "load", FIXTURE_FILE])
    print(result.output)
    assert result.exit_code == 2
    assert "Error: Missing argument" in result.output
