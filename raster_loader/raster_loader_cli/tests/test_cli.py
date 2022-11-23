from click.testing import CliRunner
from raster_loader.raster_loader_cli import raster_loader_cli


def test_info():
    """Test the info command."""
    runner = CliRunner()
    result = runner.invoke(raster_loader_cli, ["info"])
    print(result.output)
    assert result.exit_code == 0
    assert "Python version" in result.output
    assert "Operating system" in result.output


# TBD: We need to mock the raster_loader.upload.RasterLoader.to_bigquery method (the Google API)
# for the following test to make sense
# def test_raster():
#     """Test the raster command."""
#     runner = CliRunner()
#     result = runner.invoke(
#         raster_loader_cli, ["raster", "load", __file__, "test_project.test_dataset.test_table"]
#     )
#     assert result.exit_code == 0
#     assert __file__ in result.output


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
        raster_loader_cli, ["raster", "load", __file__, "invalid_table"]
    )
    assert result.exit_code == 2
    assert "Error" in result.output


def test_raster_missing_input():
    """Test whether the raster command always requires a file and table input."""
    runner = CliRunner()
    result = runner.invoke(raster_loader_cli, ["raster", "load", __file__])
    print(result.output)
    assert result.exit_code == 2
    assert "Error: Missing argument" in result.output
