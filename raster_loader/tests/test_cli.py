from raster_loader.cli.bigquery import upload

from click.testing import CliRunner

import os

here = os.path.dirname(os.path.abspath(__file__))
fixtures = os.path.join(here, "fixtures")
tiff = os.path.join(fixtures, "mosaic.tif")


def test_bigquery_upload_func():

    table = "table"
    dataset = "dataset"
    project = "project"
    chunk_size = 100
    input_crs = "4326"
    band = 1

    result = upload(
        tiff, table, dataset, project, chunk_size, input_crs, band, test=True
    )

    assert result is not None


def test_bigquery_upload_with_client():
    runner = CliRunner()
    result = runner.invoke(
        upload,
        [
            tiff,
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
