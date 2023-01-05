import os
import uuid

import click


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
@click.option("--table", help="The name of the table.", default=None)
@click.option("--band", help="Band within raster to upload.", default=1)
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
@click.option("--test", help="Use Mock BigQuery Client", default=False, is_flag=True)
def upload(
    file_path,
    project,
    dataset,
    table,
    band,
    chunk_size,
    input_crs,
    overwrite=False,
    test=False,
):

    from raster_loader.tests.mocks import bigquery_client
    from raster_loader.io import import_bigquery
    from raster_loader.io import rasterio_to_bigquery
    from raster_loader.io import get_number_of_blocks
    from raster_loader.io import print_band_information
    from raster_loader.io import get_block_dims

    # create default table name if not provided
    if table is None:
        table = os.path.basename(file_path).split(".")[0]
        table = "_".join([table, "band", str(band), str(uuid.uuid4())])

    # swap out BigQuery client for testing purposes
    if test:
        client = bigquery_client()
    else:  # pragma: no cover
        bigquery = import_bigquery()
        client = bigquery.Client(project=project)

    # introspect raster file
    num_blocks = get_number_of_blocks(file_path)
    file_size_mb = os.path.getsize(file_path) / 1024 / 1024

    click.echo("Preparing to upload raster file to BigQuery...")
    click.echo("File Path: {}".format(file_path))
    click.echo("File Size: {} MB".format(file_size_mb))
    print_band_information(file_path)
    click.echo("Source Band: {}".format(band))
    click.echo("Number of Blocks: {}".format(num_blocks))
    click.echo("Block Dims: {}".format(get_block_dims(file_path)))
    click.echo("Project: {}".format(project))
    click.echo("Dataset: {}".format(dataset))
    click.echo("Table: {}".format(table))
    click.echo("Number of Records Per BigQuery Append: {}".format(chunk_size))
    click.echo("Input CRS: {}".format(input_crs))

    click.echo("Uploading Raster to BigQuery")

    rasterio_to_bigquery(
        file_path,
        table,
        dataset,
        project,
        band,
        chunk_size,
        input_crs,
        client=client,
        overwrite=overwrite,
    )

    click.echo("Raster file uploaded to Google BigQuery")
    return 0


@bigquery.command(help="Load and describe a table from BigQuery")
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
