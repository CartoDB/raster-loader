import os
from urllib.parse import urlparse

import click
from functools import wraps, partial

from raster_loader.utils import get_default_table_name, check_private_key
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
@click.option(
    "--private-key-path",
    help="The path to the private key file. (PEM format)",
    required=False,
    default=None,
)
@click.option(
    "--private-key-passphrase",
    help="The passphrase for the private key.",
    required=False,
    default=None,
)
@click.option("--role", help="The role to use for the file upload.", default=None)
@click.option("--warehouse", help="Name of the default warehouse to use.", default=None)
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
@catch_exception()
def upload(
    account,
    username,
    password,
    token,
    private_key_path,
    private_key_passphrase,
    role,
    warehouse,
    file_path,
    file_url,
    database,
    schema,
    table,
    band,
    band_name,
    chunk_size,
    compress,
    overwrite=False,
    append=False,
    cleanup_on_failure=False,
    exact_stats=False,
    basic_stats=False,
):
    from raster_loader.io.common import (
        get_number_of_blocks,
        print_band_information,
        get_block_dims,
    )

    if not (
        (token is not None and username is None)
        or (
            token is None
            and username is not None
            and password is not None
            and private_key_path is None
        )
        or (
            token is None
            and username is not None
            and password is None
            and private_key_path is not None
        )
    ):
        raise ValueError(
            "Either (--token) or (--username and --private-key-path) or"
            " (--username and --password) must be provided."
        )

    if private_key_path is not None:
        check_private_key(private_key_path, private_key_passphrase)
        if username is None:
            raise ValueError("--username must be provided when using a private key.")

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
        private_key_path=private_key_path,
        private_key_passphrase=private_key_passphrase,
        token=token,
        account=account,
        database=database,
        schema=schema,
        role=role,
        warehouse=warehouse,
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
    click.echo("Compress: {}".format(compress))

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
        exact_stats=exact_stats,
        basic_stats=basic_stats,
        compress=compress,
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
@click.option(
    "--private-key-path",
    help="The path to the private key file. (PEM format)",
    required=False,
    default=None,
)
@click.option(
    "--private-key-passphrase",
    help="The passphrase for the private key.",
    required=False,
    default=None,
)
@click.option("--role", help="The role to use for the file upload.", default=None)
@click.option("--warehouse", help="Name of the default warehouse to use.", default=None)
@click.option("--database", help="The name of the database.", required=True)
@click.option("--schema", help="The name of the schema.", required=True)
@click.option("--table", help="The name of the table.", required=True)
@click.option("--limit", help="Limit number of rows returned", default=10)
def describe(
    account,
    username,
    password,
    token,
    private_key_path,
    private_key_passphrase,
    role,
    warehouse,
    database,
    schema,
    table,
    limit,
):

    if not (
        (token is not None and username is None)
        or (
            token is None
            and username is not None
            and password is not None
            and private_key_path is None
        )
        or (
            token is None
            and username is not None
            and password is None
            and private_key_path is not None
        )
    ):
        raise ValueError(
            "Either (--token) or (--username and --private-key-path) or"
            " (--username and --password) must be provided."
        )

    if private_key_path is not None:
        check_private_key(private_key_path, private_key_passphrase)
        if username is None:
            raise ValueError("--username must be provided when using a private key.")

    fqn = f"{database}.{schema}.{table}"
    connector = SnowflakeConnection(
        username=username,
        password=password,
        private_key_path=private_key_path,
        private_key_passphrase=private_key_passphrase,
        token=token,
        account=account,
        database=database,
        schema=schema,
        role=role,
        warehouse=warehouse,
    )
    df = connector.get_records(fqn, limit)
    print(f"Table: {fqn}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Column names: {df.columns}")
    print(f"Column types: {df.dtypes}")
    print(f"Column head: {df.head()}")
