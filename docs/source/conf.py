# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_rtd_theme
import datetime as dt
from os import popen
from shutil import which


def get_version() -> str:
    """Use git to read version from most recent tag.

    Returns:
        str: Version number
    """
    release = "dev"
    if which("git"):
        release = popen("git describe --tags --always").read().strip()
        if release and release[0] == "v":
            release = release[1:]
        else:
            release = f"dev-{release}"

    return release


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Raster Loader"
copyright = f"{dt.datetime.now().year}, Carto"
author = "Carto"
release = version = get_version()

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/carto-logo.png"
html_theme_options = {
    "display_version": True,
}
html_context = {
    "display_github": True,
    "github_user": "CartoDB",
    "github_repo": "raster-loader",
    "github_version": "main/docs/",
}
html_favicon = "_static/carto-logo.png"
