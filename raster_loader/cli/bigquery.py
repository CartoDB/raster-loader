import os
import uuid

import click
from functools import wraps, partial

try:
    import google.cloud.bigquery
except ImportError:  # pragma: no cover
    _has_bigquery = False
else:
    _has_bigquery = True


def catch_exception(func=None, *, handle=Exception):
    if not func:
        return partial(catch_exception, handle=handle)

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except handle as e:
            raise click.ClickException(e)

    return wrapper


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
@click.option(
    "--band",
    help="Band(s) within raster to upload. "
    "Could repeat --band to specify multiple bands.",
    default=[1],
    multiple=True,
)
@click.option(
    "--band_name",
    help="Column name(s) used to store band (Default: band_<band_num>). "
    "Could repeat --band_name to specify multiple bands column names. "
    "List of columns names HAVE to pair --band list with the same order.",
    default=[None],
    multiple=True,
)
@click.option(
    "--chunk_size", help="The number of blocks to upload in each chunk.", default=1000
)
@click.option(
    "--overwrite",
    help="Overwrite existing data in the table if it already exists.",
    default=False,
    is_flag=True,
)
@click.option(
    "--append",
    help="Append records into a table if it already exists.",
    default=False,
    is_flag=True,
)
@click.option("--test", help="Use Mock BigQuery Client", default=False, is_flag=True)
@catch_exception()
def upload(
    file_path,
    project,
    dataset,
    table,
    band,
    band_name,
    chunk_size,
    overwrite=False,
    append=False,
    test=False,
):
    from raster_loader.tests.mocks import bigquery_client
    from raster_loader.io import import_error_bigquery
    from raster_loader.io import rasterio_to_bigquery
    from raster_loader.io import get_number_of_blocks
    from raster_loader.io import print_band_information
    from raster_loader.io import get_block_dims

    # check that band and band_name are the same length
    # if band_name provided
    if band_name != (None,):
        if len(band) != len(band_name):
            raise ValueError("The number of bands must equal the number of band names.")
    else:
        band_name = [None] * len(band)

    # pair band and band_name in a list of tuple
    bands_info = list(zip(band, band_name))

    # create default table name if not provided
    if table is None:
        table = os.path.basename(file_path).split(".")[0]
        table = "_".join([table, "band", str(band), str(uuid.uuid4())])

    # swap out BigQuery client for testing purposes
    if test:
        client = bigquery_client()
    else:  # pragma: no cover
        """Requires bigquery."""
        if not _has_bigquery:  # pragma: no cover
            import_error_bigquery()
        client = google.cloud.bigquery.Client(project=project)

    # introspect raster file
    num_blocks = get_number_of_blocks(file_path)
    file_size_mb = os.path.getsize(file_path) / 1024 / 1024

    click.echo("Preparing to upload raster file to BigQuery...")
    click.echo("File Path: {}".format(file_path))
    click.echo("File Size: {} MB".format(file_size_mb))
    print_band_information(file_path)
    click.echo("Source Band: {}".format(band))
    click.echo("Band Name: {}".format(band_name))
    click.echo("Number of Blocks: {}".format(num_blocks))
    click.echo("Block Dims: {}".format(get_block_dims(file_path)))
    click.echo("Project: {}".format(project))
    click.echo("Dataset: {}".format(dataset))
    click.echo("Table: {}".format(table))
    click.echo("Number of Records Per BigQuery Append: {}".format(chunk_size))

    click.echo("Uploading Raster to BigQuery")

    rasterio_to_bigquery(
        file_path,
        table,
        dataset,
        project,
        bands_info,
        chunk_size,
        client=client,
        overwrite=overwrite,
        append=append,
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
    print(f"Column head: {df.head()}")
