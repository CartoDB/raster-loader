"""Raster functions for CLI"""

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
    print(f"TBD: Loading file {file} to table {table}")
