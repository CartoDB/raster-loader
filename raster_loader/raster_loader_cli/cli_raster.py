"""Raster functions for CLI"""

from raster_loader.upload import RasterLoader

import click


@click.group()
def raster():
    """Raster functions for CLI"""
    pass


@raster.command()
# @click.option("--file", "-f", help="File to upload")
# @click.option("--table", "-t", help="BigQuery table to upload to")
@click.argument("file", type=click.Path(exists=True))
@click.argument("table", type=str)
def load(file, table):
    """Load a raster file"""
    # Parse TABLE into project, dataset, and table
    table_args = table.split(".")
    if len(table_args) != 3:
        raise click.BadParameter(
            "TABLE name must be in the format: project.dataset.table"
        )

    # load the FILE into TABLE
    click.echo(f"Loading {file} to {table}")
    raster_loader = RasterLoader(file)
    project = table_args[0]
    dataset = table_args[1]
    table = table_args[2]
    # raster_loader.to_bigquery("raster-loader", project, dataset, table)
