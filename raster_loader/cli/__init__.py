try:
    from importlib.metadata import entry_points
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import entry_points

import click
from click_plugins import with_plugins


@with_plugins(cmd for cmd in list(entry_points(group="raster_loader.cli")))
@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main(args=None):
    """
    The ``carto`` command line interface.
    """
    pass


if __name__ == "__main__":  # pragma: no cover
    main()
