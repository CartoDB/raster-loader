"""CLI functions for raster loader CLI"""

import click

from cli_info import system_info
from cli_raster import raster


@click.group()
def raster_loader_cli():
    """Command-line interface to set up and run raster uploading."""
    pass


# add subcommands for raster operations
raster_loader_cli.add_command(raster)

# add command for system info
@raster_loader_cli.command()
def info():
    """Print system information for debugging."""
    system_info()


if __name__ == "__main__":
    raster_loader_cli()
