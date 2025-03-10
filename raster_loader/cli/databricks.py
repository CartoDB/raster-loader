import os
from urllib.parse import urlparse

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
            raise click.ClickException(e)

    return wrapper


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def databricks(args=None):
    """
    Manage Databricks resources.
    """
    pass


@databricks.command(help="Upload a raster file to Databricks.")
@click.option("--url", help="The Databricks workspace URL.", required=True)
@click.option("--token", help="The Databricks access token.", required=True)
@click.option("--http-path", help="The HTTP path for the SQL warehouse.", required=True)
@click.option(
    "--cluster-id",
    help="The Databricks cluster ID for Spark operations.",
    required=True,
)
@click.option(
    "--file_path", help="The path to the raster file.", required=False, default=None
)
@click.option(
    "--file_url", help="The path to the raster file.", required=False, default=None
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
    "List of columns names HAVE to pair --band list with the same order.",
    default=[None],
    multiple=True,
)
@click.option(
    "--chunk_size", help="The number of blocks to upload in each chunk.", default=10000
)
@click.option(
    "--parallelism",
    help="Number of partitions when uploading each chunk.",
    default=200,
    type=int,
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
@click.option(
    "--exact_stats",
    help="Compute exact statistics for the raster bands.",
    default=False,
    is_flag=True,
)
@click.option(
    "--basic_stats",
    help="Compute basic stats and omit quantiles and most frequent values.",
    required=False,
    is_flag=True,
)
@click.option(
    "--compress",
    help="Compress band data using zlib.",
    is_flag=True,
    default=False,
)
@click.option(
    "--compression-level",
    help="Compression level (1-9, higher = better compression but slower)",
    type=int,
    default=6,
)
@catch_exception()
def upload(
    url,
    token,
    http_path,
    cluster_id,
    file_path,
    file_url,
    catalog,
    schema,
    table,
    band,
    band_name,
    chunk_size,
    parallelism,
    compress,
    overwrite=False,
    append=False,
    cleanup_on_failure=False,
    exact_stats=False,
    basic_stats=False,
    compression_level=6,
):
    from raster_loader.io.common import (
        get_number_of_blocks,
        print_band_information,
        get_block_dims,
    )

    if file_path is None and file_url is None:
        raise ValueError("Either --file_path or --file_url must be provided.")

    if file_path and file_url:
        raise ValueError("Only one of --file_path or --file_url must be provided.")

    is_local_file = file_path is not None

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
        table = get_default_table_name(
            file_path if is_local_file else urlparse(file_url).path, band
        )

    connector = DatabricksConnection(
        server_hostname=url.replace("https://", ""),
        http_path=http_path,
        access_token=token,
        cluster_id=cluster_id,
    )

    source = file_path if is_local_file else file_url

    # introspect raster file
    num_blocks = get_number_of_blocks(source)
    file_size_mb = 0
    if is_local_file:
        file_size_mb = os.path.getsize(file_path) / 1024 / 1024

    click.echo("Preparing to upload raster file to Databricks...")
    click.echo("File Path: {}".format(source))
    click.echo("File Size: {} MB".format(file_size_mb))
    print_band_information(source)
    click.echo("Source Band: {}".format(band))
    click.echo("Band Name: {}".format(band_name))
    click.echo("Number of Blocks: {}".format(num_blocks))
    click.echo("Block Dims: {}".format(get_block_dims(source)))
    click.echo("Catalog: {}".format(catalog))
    click.echo("Schema: {}".format(schema))
    click.echo("Table: {}".format(table))
    click.echo("Number of Records Per Databricks Append: {}".format(chunk_size))
    click.echo("Parallelism: {}".format(parallelism))
    click.echo("Compress: {}".format(compress))

    click.echo("Uploading Raster to Databricks")

    fqn = f"`{catalog}`.`{schema}`.`{table}`"
    connector.upload_raster(
        source,
        fqn,
        bands_info,
        chunk_size,
        parallelism,
        overwrite=overwrite,
        append=append,
        cleanup_on_failure=cleanup_on_failure,
        exact_stats=exact_stats,
        basic_stats=basic_stats,
        compress=compress,
        compression_level=compression_level,
    )

    click.echo("Raster file uploaded to Databricks")
    return 0


@databricks.command(help="Load and describe a table from Databricks")
@click.option("--url", help="The Databricks workspace URL.", required=True)
@click.option("--token", help="The Databricks access token.", required=True)
@click.option("--http-path", help="The HTTP path for the SQL warehouse.", required=True)
@click.option(
    "--cluster-id",
    help="The Databricks cluster ID for Spark operations.",
    required=True,
)
@click.option("--catalog", help="The name of the catalog.", required=True)
@click.option("--schema", help="The name of the schema.", required=True)
@click.option("--table", help="The name of the table.", required=True)
@click.option("--limit", help="Limit number of rows returned", default=10)
def describe(
    url,
    token,
    http_path,
    cluster_id,
    catalog,
    schema,
    table,
    limit,
):
    fqn = f"`{catalog}`.`{schema}`.`{table}`"
    connector = DatabricksConnection(
        server_hostname=url.replace("https://", ""),
        http_path=http_path,
        access_token=token,
        cluster_id=cluster_id,
    )

    df = connector.execute_to_dataframe(f"SELECT * FROM {fqn} LIMIT {limit}")
    print(f"Table: {fqn}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Column names: {df.columns}")
    print(f"Column types: {df.dtypes}")
    print(f"Column head: {df.head()}")
