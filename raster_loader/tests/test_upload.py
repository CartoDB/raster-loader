from unittest.mock import patch

import pytest

from raster_loader import errors, RasterLoader
from raster_loader.tests import mocks


@patch.object(RasterLoader, "_bigquery_client", return_value=mocks.bigquery_client())
def test_upload_to_bigquery_successful(*args, **kwargs):
    raster_loader = RasterLoader(file_path="raster_loader/tests/fixtures/mosaic.tif")

    raster_loader.to_bigquery(
        project="mock_project",
        dataset="mock_dataset",
        table="raster_data",
    )


@patch.object(
    RasterLoader,
    "_bigquery_client",
    side_effect=errors.ClientError,
)
def test_upload_to_bigquery_unsuccessful_client_error(*args, **kwargs):
    raster_loader = RasterLoader(file_path="raster_loader/tests/fixtures/mosaic.tif")

    with pytest.raises(errors.ClientError):
        raster_loader.to_bigquery(
            project="mock_project",
            dataset="mock_dataset",
            table="raster_data",
        )


@patch.object(
    RasterLoader,
    "_bigquery_client",
    return_value=mocks.bigquery_client(load_error=True),
)
def test_upload_to_bigquery_unsuccessful_load_error(*args, **kwargs):
    raster_loader = RasterLoader(file_path="raster_loader/tests/fixtures/mosaic.tif")

    with pytest.raises(errors.UploadError):
        raster_loader.to_bigquery(
            project="mock_project",
            dataset="mock_dataset",
            table="raster_data",
        )