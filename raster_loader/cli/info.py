"""Module for info subcommand."""

import platform
import sys

import click

import raster_loader


@click.command(help="Display system information.")
def info():
    """Display system information."""
    click.echo(f"Raster Loader version: {raster_loader.__version__}")
    click.echo(f"Python version: {sys.version.split(' (')[0]}")
    click.echo(f"Platform: {platform.platform()}")
    click.echo(f"System version: {platform.system()} {platform.release()}")
    click.echo(f"Machine: {platform.machine()}")
    click.echo(f"Processor: {platform.processor()}")
    click.echo(f"Architecture: {platform.architecture()[0]}")
