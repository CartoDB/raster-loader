try:
    from importlib.metadata import entry_points
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import entry_points

import click
from click_plugins import with_plugins

# Handle different entry_points API versions
try:
    # Python 3.10+ with group parameter
    eps = entry_points(group="raster_loader.cli")
except TypeError:
    # Python 3.8-3.9 without group parameter
    eps = entry_points().get("raster_loader.cli", [])

@with_plugins(cmd for cmd in list(eps))
@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main(args=None):
    """
    The ``carto`` command line interface.
    """
    pass


if __name__ == "__main__":  # pragma: no cover
    main()
