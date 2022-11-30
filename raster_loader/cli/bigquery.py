import click

from raster_loader import RasterLoader


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def bigquery(args=None):
    """
    Manage Google BigQuery resources.
    """
    pass


@bigquery.command(help="Upload a raster file to Google BigQuery.")
@click.option("--file_path", help="The path to the raster file.", required=True)
@click.option("--project", help="The name of the Google Cloud project.", required=True)
@click.option("--dataset", help="The name of the dataset.", required=True)
@click.option("--table", help="The name of the table.", required=True)
@click.option("--dst_crs", help="The EPSG code of the destination CRS.", default=4326)
@click.option(
    "--chunk_size", help="The number of blocks to upload in each chunk.", default=100
)
def upload(file_path, project, dataset, table, dst_crs, chunk_size):
    click.echo("Uploading raster file to Google BigQuery")
    raster_loader = RasterLoader(
        file_path=file_path,
        dst_crs=dst_crs,
    )

    raster_loader.to_bigquery(
        project=project,
        dataset=dataset,
        table=table,
        chunk_size=chunk_size,
    )
    click.echo("Raster file uploaded to Google BigQuery")


@bigquery.command(help="Upload a raster file to Google BigQuery.")
@click.option("--file_path", help="The path to the raster file.", required=True)
@click.option("--project", help="The name of the Google Cloud project.", required=True)
@click.option("--dataset", help="The name of the dataset.", required=True)
@click.option("--table", help="The name of the table.", required=True)
@click.option("--dst_crs", help="The EPSG code of the destination CRS.", default=4326)
@click.option(
    "--chunk_size", help="The number of blocks to upload in each chunk.", default=100
)
def upload2(file_path, project, dataset, table, dst_crs, chunk_size):

    from raster_loader.io import rasterio_to_bigquery

    click.echo("Uploading raster file to Google BigQuery")
    rasterio_to_bigquery(file_path, table, dataset, project, chunk_size)
    click.echo("Raster file uploaded to Google BigQuery")
