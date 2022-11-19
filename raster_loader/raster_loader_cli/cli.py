"""Command-line interface for the raster loader."""

import click

from cli_info import system_info
from cli_raster import raster

# from ..raster_loader.utils import some_function

# TBD: Won't be necessary once the main package is available
import sys

sys.path.append("../../")

# # importing
from raster_loader.utils import some_function

# import raster_loader


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
    sys.exit()


if __name__ == "__main__":
    raster_loader_cli()
