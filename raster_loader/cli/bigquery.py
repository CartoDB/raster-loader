import os
import uuid

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
@click.option("--input_crs", help="The EPSG code of the input dataset.", default=4326)
@click.option(
    "--chunk_size", help="The number of blocks to upload in each chunk.", default=100
)
def upload(file_path, project, dataset, table, input_crs, chunk_size):
    click.echo("Uploading raster file to Google BigQuery")
    raster_loader = RasterLoader(
        file_path=file_path,
        input_crs=input_crs,
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
@click.option("--table", help="The name of the table.", default=None)
@click.option("--band", help="Band within raster to upload.", default=1)
@click.option(
    "--chunk_size", help="The number of blocks to upload in each chunk.", default=100
)
@click.option(
    "--input_crs", help="The EPSG code of the input raster's CRS.", default=None
)
def upload2(file_path, project, dataset, table, band, chunk_size, input_crs):

    from raster_loader.io import rasterio_to_bigquery

    # TODO: implement hint
    click.echo("Uploading raster file to Google BigQuery")
    if table is None:
        table = os.path.basename(file_path).split(".")[0]
        table = "_".join([table, "band", str(band), str(uuid.uuid4())])

    rasterio_to_bigquery(
        file_path, table, dataset, project, band, chunk_size, input_crs
    )
    click.echo("Raster file uploaded to Google BigQuery")


@bigquery.command(help="Pull and Describe Table from BigQuery")
@click.option("--project", help="The name of the Google Cloud project.", required=True)
@click.option("--dataset", help="The name of the dataset.", required=True)
@click.option("--table", help="The name of the table.", required=True)
@click.option("--limit", help="Limit number of rows returned", default=10)
def describe(project, dataset, table, limit):

    from raster_loader.io import bigquery_to_records

    df = bigquery_to_records(table, dataset, project, limit)
    print(f"Table: {project}.{dataset}.{table}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Column names: {df.columns}")
    print(f"Column types: {df.dtypes}")
    print(f"Column descriptions: {df.describe()}")
    print(f"Column head: {df.head()}")
