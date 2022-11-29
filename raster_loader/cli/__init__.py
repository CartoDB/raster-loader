from pkg_resources import iter_entry_points as entry_points

import click
from click_plugins import with_plugins


@with_plugins(cmd for cmd in list(entry_points("raster_loader.cli")))
@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main(args=None):
    """
    The ``carto`` command line interface.
    """
    pass


if __name__ == "__main__":  # pragma: no cover
    main()
