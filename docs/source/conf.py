# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sphinx_rtd_theme
import datetime as dt

import raster_loader

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Raster Loader"
copyright = f"{dt.datetime.now().year}, Carto"
author = "Carto"
copyright = author
release = version = raster_loader.__version__
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx_click",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []
    
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}


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
html_css_files = [
    "custom.css",
]
