import os
from urllib.parse import urlparse

import click
from functools import wraps, partial

from raster_loader.utils import get_default_table_name
from raster_loader.io.snowflake import SnowflakeConnection


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
def snowflake(args=None):
    """
    Manage Snowflake resources.
    """
    pass


@snowflake.command(help="Upload a raster file to Snowflake.")
@click.option("--account", help="The Swnoflake account.", required=True)
@click.option("--username", help="The username.", required=False, default=None)
@click.option("--password", help="The password.", required=False, default=None)
@click.option(
    "--token",
    help="An access token to authenticate with.",
    required=False,
    default=None,
)
@click.option("--role", help="The role to use for the file upload.", default=None)
@click.option(
    "--file_path", help="The path to the raster file.", required=False, default=None
)
@click.option(
    "--file_url", help="The path to the raster file.", required=False, default=None
)
@click.option("--database", help="The name of the database.", required=True)
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
    account,
    username,
    password,
    token,
    role,
    file_path,
    file_url,
    database,
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

    if (token is None and (username is None or password is None)) or all(
        v is not None for v in [token, username, password]
    ):
        raise ValueError(
            "Either --token or --username and --password must be provided."
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

    connector = SnowflakeConnection(
        username=username,
        password=password,
        token=token,
        account=account,
        database=database,
        schema=schema,
        role=role,
    )

    source = file_path if is_local_file else file_url

    # introspect raster file
    num_blocks = get_number_of_blocks(source)
    file_size_mb = 0
    if is_local_file:
        file_size_mb = os.path.getsize(file_path) / 1024 / 1024

    click.echo("Preparing to upload raster file to Snowflake...")
    click.echo("File Path: {}".format(source))
    click.echo("File Size: {} MB".format(file_size_mb))
    print_band_information(source)
    click.echo("Source Band: {}".format(band))
    click.echo("Band Name: {}".format(band_name))
    click.echo("Number of Blocks: {}".format(num_blocks))
    click.echo("Block Dims: {}".format(get_block_dims(source)))
    click.echo("Database: {}".format(database))
    click.echo("Schema: {}".format(schema))
    click.echo("Table: {}".format(table))
    click.echo("Number of Records Per Snowflake Append: {}".format(chunk_size))

    click.echo("Uploading Raster to Snowflake")

    fqn = f"{database}.{schema}.{table}"
    connector.upload_raster(
        source,
        fqn,
        bands_info,
        chunk_size,
        overwrite=overwrite,
        append=append,
        cleanup_on_failure=cleanup_on_failure,
    )

    click.echo("Raster file uploaded to Snowflake")
    return 0


@snowflake.command(help="Load and describe a table from Snowflake")
@click.option("--account", help="The Swnoflake account.", required=True)
@click.option("--username", help="The username.", required=False, default=None)
@click.option("--password", help="The password.", required=False, default=None)
@click.option(
    "--token",
    help="An access token to authenticate with.",
    required=False,
    default=None,
)
@click.option("--role", help="The role to use for the file upload.", default=None)
@click.option("--database", help="The name of the database.", required=True)
@click.option("--schema", help="The name of the schema.", required=True)
@click.option("--table", help="The name of the table.", required=True)
@click.option("--limit", help="Limit number of rows returned", default=10)
def describe(account, username, password, token, role, database, schema, table, limit):

    if (token is None and (username is None or password is None)) or all(
        v is not None for v in [token, username, password]
    ):
        raise ValueError(
            "Either --token or --username and --password must be provided."
        )

    fqn = f"{database}.{schema}.{table}"
    connector = SnowflakeConnection(
        username=username,
        password=password,
        token=token,
        account=account,
        database=database,
        schema=schema,
        role=role,
    )
    df = connector.get_records(fqn, limit)
    print(f"Table: {fqn}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Column names: {df.columns}")
    print(f"Column types: {df.dtypes}")
    print(f"Column head: {df.head()}")
