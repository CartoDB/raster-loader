from importlib.metadata import entry_points as _entry_points

import click
from click_plugins import with_plugins


def entry_points(group):
    """The installed entry points of ``group``.

    ``importlib.metadata`` replaces ``pkg_resources``, which setuptools 81
    removed. Python 3.9 returns every group as a dict; 3.10+ selects by group.
    """
    found = _entry_points()
    if hasattr(found, "select"):
        return found.select(group=group)
    return found.get(group, [])


@with_plugins(cmd for cmd in list(entry_points("raster_loader.cli")))
@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def main(args=None):
    """
    The ``carto`` command line interface.
    """
    pass


if __name__ == "__main__":  # pragma: no cover
    main()
