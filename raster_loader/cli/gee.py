import os
import uuid
import geojson


import click

# INTERMEDIATE_BUCKET = 'gs://bq_vdelacruz/raster_test/'
INTERMEDIATE_BUCKET = """
https://console.cloud.google.com/storage/browser/carto-hackathon-2023-bqcarto;tab=objects?forceOnBucketsSortingFiltering=true&project=bqcarto/
"""

try:
    import ee

    # service_account = "<account-mail>@<project>.iam.gserviceaccount.com"
    # credentials = ee.ServiceAccountCredentials(
    #     service_account, "/path/to/gee_service_account.json"
    # )
    # ee.Initialize(credentials)
    ee.Initialize()
except ImportError:  # pragma: no cover
    _has_gee = False
except Exception as e:
    raise e
else:
    _has_gee = True

try:
    import google.cloud.bigquery
except ImportError:  # pragma: no cover
    _has_bigquery = False
else:
    _has_bigquery = True


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def gee(args=None):
    """
    Manage Google Earth Engine resources.
    """
    pass


@gee.command(help="Upload a GEE raster file to Google BigQuery.")
@click.option("--image", help="The GEE asset of the raster.", required=True)
@click.option("--roi", help="Region of Interest json file.", required=True)
@click.option("--project", help="The name of the Google Cloud project.", required=True)
@click.option("--dataset", help="The name of the dataset.", required=True)
@click.option("--table", help="The name of the table.", default=None)
@click.option("--bands", help="Bands within raster to upload.", default="1")
@click.option(
    "--chunk_size", help="The number of blocks to upload in each chunk.", default=100
)
@click.option(
    "--input_crs", help="The EPSG code of the input raster's CRS.", default=None
)
@click.option(
    "--overwrite",
    help="Overwrite existing data in the table if it already exists.",
    default=False,
    is_flag=True,
)
@click.option(
    "--output_quadbin",
    help=(
        "Upload the raster to the BigQuery table in a quadbin format "
        "(input raster must be a GoogleMapsCompatible raster)."
    ),
    default=False,
    is_flag=True,
)
@click.option("--test", help="Use Mock BigQuery Client", default=False, is_flag=True)
def upload(
    image,
    roi,
    project,
    dataset,
    table,
    bands,
    chunk_size,
    input_crs,
    overwrite=False,
    output_quadbin=False,
    test=False,
):

    from raster_loader.tests.mocks import bigquery_client
    from raster_loader.io import import_error_bigquery
    from raster_loader.io import rasterio_to_bigquery
    from raster_loader.io import gee_to_bucket

    # from raster_loader.io import get_number_of_blocks
    from raster_loader.io import gee_print_band_information

    # from raster_loader.io import get_block_dims

    # split bands
    requiredBands = []
    if bands is not None:
        requiredBands = bands.replace(" ", "").split(",")

    # create default table name if not provided
    tables = []
    if table is None:
        table = os.path.basename(image)
        for band in bands:
            tables.append("_".join([table, "band", str(band), str(uuid.uuid4())]))

    if roi is not None:
        with open(roi) as f:
            roi_geojson = geojson.load(f)
            # geometry = ee.Geometry.Polygon(roi_geojson)

    # swap out BigQuery client for testing purposes
    if test:
        client = bigquery_client()
    else:  # pragma: no cover
        """Requires bigquery."""
        if not _has_bigquery:  # pragma: no cover
            import_error_bigquery()
        client = google.cloud.bigquery.Client(project=project)

    # GEE open it to get image metadata
    click.echo("Preparing to get raster file from GEE...")
    for band in requiredBands:
        gee_image = ee.Image(image).select(band)

        destination_table = "_".join([table, "band", str(band), str(uuid.uuid4())])
        gcloud_path = f"{INTERMEDIATE_BUCKET}{destination_table}"

        # introspect raster file
        # num_blocks = get_number_of_blocks(file_path)
        file_size_mb = gee_image.size() / 1024 / 1024

        # center = geometry.centroid().getInfo()['coordinates']
        # center.reverse()

        click.echo("GEE Image: {}".format(image))
        click.echo("Source Band: {}".format(band))
        click.echo("File Size: {} MB".format(file_size_mb))
        click.echo("Bucket: {}".format(INTERMEDIATE_BUCKET))
        # print_band_information(file_path)
        gee_print_band_information(gee_image)
        # click.echo("Number of Blocks: {}".format(num_blocks))
        # click.echo("Block Dims: {}".format(get_block_dims(file_path)))
        click.echo("Project: {}".format(project))
        click.echo("Dataset: {}".format(dataset))
        click.echo("Table: {}".format(table))
        click.echo("Number of Records Per BigQuery Append: {}".format(chunk_size))
        click.echo("Input CRS: {}".format(input_crs))

        # step1 download GEE image in intermediate bucket
        click.echo("Downloading band {band} to bucket")
        gee_to_bucket(gee_image, roi_geojson, INTERMEDIATE_BUCKET, destination_table)

        # step2 import image from  intermediate bucket to BQ
        click.echo("Uploading Raster to BigQuery")
        rasterio_to_bigquery(
            gcloud_path,
            destination_table,
            dataset,
            project,
            band,
            chunk_size,
            input_crs,
            client=client,
            overwrite=overwrite,
            output_quadbin=output_quadbin,
        )

        click.echo("Raster file uploaded to Google BigQuery")
        return 0
