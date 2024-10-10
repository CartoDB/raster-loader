import click
from functools import wraps, partial

from raster_loader.utils import get_default_table_name
from raster_loader.io.databricks import DatabricksConnection


def catch_exception(func=None, *, handle=Exception):
    if not func:
        return partial(catch_exception, handle=handle)

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except handle as e:
            raise click.ClickException(str(e))

    return wrapper


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def databricks(args=None):
    """
    Manage Databricks resources.
    """
    pass


@databricks.command(help="Upload a raster file to Databricks.")
@click.option("--host", help="The Databricks host URL.", required=True)
@click.option("--token", help="The Databricks access token.", required=True)
@click.option(
    "--cluster-id", help="The Databricks cluster ID.", required=True
)  # New option
@click.option(
    "--file_path", help="The path to the raster file.", required=False, default=None
)
@click.option(
    "--file_url", help="The URL to the raster file.", required=False, default=None
)
@click.option("--catalog", help="The name of the catalog.", required=True)
@click.option("--schema", help="The name of the schema.", required=True)
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
    "List of column names HAVE to pair with --band list in the same order.",
    default=[None],
    multiple=True,
)
@click.option(
    "--chunk_size", help="The number of blocks to upload in each chunk.", default=10000
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
@click.option(
    "--cleanup-on-failure",
    help="Clean up resources if the upload fails. Useful for non-interactive scripts.",
    default=False,
    is_flag=True,
)
@catch_exception()
def upload(
    host,
    token,
    cluster_id,  # Accept cluster ID
    file_path,
    file_url,
    catalog,
    schema,
    table,
    band,
    band_name,
    chunk_size,
    overwrite=False,
    append=False,
    cleanup_on_failure=False,
):
    from raster_loader.io.common import (
        get_number_of_blocks,
        print_band_information,
        get_block_dims,
    )
    import os
    from urllib.parse import urlparse

    if file_path is None and file_url is None:
        raise ValueError("Need either a --file_path or --file_url")

    if file_path and file_url:
        raise ValueError("Only one of --file_path or --file_url must be provided.")

    is_local_file = file_path is not None

    # Check that band and band_name are the same length if band_name provided
    if band_name != (None,):
        if len(band) != len(band_name):
            raise ValueError("Must supply the same number of band_names as bands")
    else:
        band_name = [None] * len(band)

    # Pair band and band_name in a list of tuples
    bands_info = list(zip(band, band_name))

    # Create default table name if not provided
    if table is None:
        table = get_default_table_name(
            file_path if is_local_file else urlparse(file_url).path, band
        )

    connector = DatabricksConnection(
        host=host,
        token=token,
        cluster_id=cluster_id,  # Pass cluster_id to DatabricksConnection
        catalog=catalog,
        schema=schema,
    )

    source = file_path if is_local_file else file_url

    # Introspect raster file
    num_blocks = get_number_of_blocks(source)
    file_size_mb = 0
    if is_local_file:
        file_size_mb = os.path.getsize(file_path) / 1024 / 1024

    click.echo("Preparing to upload raster file to Databricks...")
    click.echo(f"File Path: {source}")
    click.echo(f"File Size: {file_size_mb} MB")
    print_band_information(source)
    click.echo(f"Source Band(s): {band}")
    click.echo(f"Band Name(s): {band_name}")
    click.echo(f"Number of Blocks: {num_blocks}")
    click.echo(f"Block Dimensions: {get_block_dims(source)}")
    click.echo(f"Catalog: {catalog}")
    click.echo(f"Schema: {schema}")
    click.echo(f"Table: {table}")
    click.echo(f"Number of Records Per Batch: {chunk_size}")

    click.echo("Uploading Raster to Databricks")

    connector.upload_raster(
        source,
        table,
        bands_info,
        chunk_size,
        overwrite=overwrite,
        append=append,
        cleanup_on_failure=cleanup_on_failure,
    )

    click.echo("Raster file uploaded to Databricks")
    exit(0)
