from unittest.mock import patch

from google.cloud import bigquery
import pytest
from raster_loader import errors, RasterLoader


@patch('google.cloud.bigquery.Client', autospec=True)
def test_upload_to_bigquery_successful(*args, **kwargs):
    raster_loader = RasterLoader(file_path="tests/fixtures/mosaic.tif", dst_crs=4326)

    raster_loader.to_bigquery(
        project="mock_project",
        dataset="mock_dataset",
        table="raster_data",
    )


@patch('google.cloud.client.ClientWithProject', autospec=True)
@patch.object(
    bigquery.Client,
    "load_table_from_dataframe",
    side_effect=errors.UploadError,
)
def test_upload_to_bigquery_unsuccessful(*args, **kwargs):
    raster_loader = RasterLoader(file_path="tests/fixtures/mosaic.tif", dst_crs=4326)

    with pytest.raises(errors.UploadError):
        raster_loader.to_bigquery(
            project="mock_project",
            dataset="mock_dataset",
            table="raster_data",
        )
