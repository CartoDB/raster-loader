## TBD this needs to be removed or updated once the main package is available

from setuptools import setup

setup(
    name="carto CLI",
    version="0.1.0",
    py_modules=["raster_loader_cli"],
    install_requires=[
        "Click",
    ],
    entry_points={
        "console_scripts": [
            "carto = cli:raster_loader_cli",
        ],
    },
)
