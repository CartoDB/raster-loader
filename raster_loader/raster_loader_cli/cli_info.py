"""System info functions for CLI"""

import platform
import sys
from os import environ

import click


def system_info():
    """Generate a list of system information"""
    python_version = sys.version.replace("\n", "")

    click.echo(f"Python version:    {python_version}")
    # click.echo(f"Conda environment: {environ.get('CONDA_DEFAULT_ENV')}")
    click.echo(f"Operating system:  {platform.platform()}")
