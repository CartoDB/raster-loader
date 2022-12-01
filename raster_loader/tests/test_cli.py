from unittest.mock import patch

from click.testing import CliRunner

from raster_loader import RasterLoader
from raster_loader.cli import main
from raster_loader.tests import mocks


@patch.object(RasterLoader, "_bigquery_client", return_value=mocks.bigquery_client())
def test_bigquery_upload(*args, **kwargs):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "bigquery",
            "upload",
            "--file_path",
            "raster_loader/tests/fixtures/mosaic.tif",
            "--project",
            "carto-ml-dev",
            "--dataset",
            "raster_loader",
            "--table",
            "mosaic",
        ],
    )
    assert result.exit_code == 0
    assert "Uploading raster file to Google BigQuery" in result.output
    assert "Raster file uploaded to Google BigQuery" in result.output