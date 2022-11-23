"""System info functions for CLI"""

import platform
import sys

import click

from raster_loader import _version


def system_info():
    """Generate a list of system information"""
    python_version = sys.version.replace("\n", "")

    click.echo(f"RasterLoader version: {_version.__version__}")
    click.echo(f"Python version:       {python_version}")
    click.echo(f"Operating system:     {platform.platform()}")
