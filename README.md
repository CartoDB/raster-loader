# raster-loader

[![PyPI version](https://badge.fury.io/py/raster-loader.svg)](https://badge.fury.io/py/raster-loader)
[![PyPI downloads](https://img.shields.io/pypi/dm/raster-loader.svg)](https://pypistats.org/packages/raster-loader)
[![Tests](https://github.com/cartodb/raster-loader/actions/workflows/ci.yml/badge.svg)](https://github.com/cartodb/raster-loader/actions)
[![Documentation Status](https://readthedocs.org/projects/raster-loader/badge/?version=latest)](https://raster-loader.readthedocs.io/en/latest/?badge=latest)

Python library for loading GIS raster data to standard cloud-based data warehouses that
don't natively support raster data.

Raster Loader is currently tested on Python 3.8, 3.9, 3.10, and 3.11.

## Documentation

The Raster Loader documentation is available at [raster-loader.readthedocs.io](https://raster-loader.readthedocs.io).

## Install

```bash
pip install -U raster-loader

pip install -U raster-loader"[bigquery]"
pip install -U raster-loader"[snowflake]"
```

### Installing from source

```bash
git clone https://github.com/cartodb/raster-loader
cd raster-loader
pip install .
```

## Usage

There are two ways you can use Raster Loader:

* Using the CLI by running `carto` in your terminal
* Using Raster Loader as a Python library (`import raster_loader`)

### CLI

After installing Raster Loader, you can run the CLI by typing `carto` in your terminal.

Currently, Raster Loader supports uploading raster data to [BigQuery](https://cloud.google.com/bigquery).
Accessing BigQuery with Raster Loader requires the
`GOOGLE_APPLICATION_CREDENTIALS` environment variable to be set to the path of a JSON
file containing your BigQuery credentials. See the
[GCP documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key)
for more information.

Two commands are available:

#### Uploading to BigQuery

`carto bigquery upload` loads raster data from a local file to a BigQuery table.
At a minimum, the `carto bigquery upload` command requires a `file_path` to a local
raster file that can be [read by GDAL](https://gdal.org/drivers/raster/index.html) and processed with [rasterio](https://rasterio.readthedocs.io/en/latest/). It also requires
the `project` (the [GCP project name](https://cloud.google.com/resource-manager/docs/creating-managing-projects))
and `dataset` (the [BigQuery dataset name](https://cloud.google.com/bigquery/docs/datasets-intro))
parameters. There are also additional parameters, such as `table` ([BigQuery table
name](https://cloud.google.com/bigquery/docs/tables-intro)) and `overwrite` (to
overwrite existing data).

For example:

``` bash

carto bigquery upload \
    --file_path /path/to/my/raster/file.tif \
    --project my-gcp-project \
    --dataset my-bigquery-dataset \
    --table my-bigquery-table \
    --overwrite

```

This command uploads the TIFF file from `/path/to/my/raster/file.tif` to a BigQuery
project named `my-gcp-project`, a dataset named `my-bigquery-dataset`, and a table
named `my-bigquery-table`. If the table already contains data, this data will be
overwritten because the `--overwrite` flag is set.

#### Inspecting a raster file on BigQuery

Use the `carto bigquery describe` command to retrieve information about a raster file
stored in a BigQuery table.

At a minimum, this command requires a
[GCP project name](https://cloud.google.com/resource-manager/docs/creating-managing-projects),
a [BigQuery dataset name](https://cloud.google.com/bigquery/docs/datasets-intro), and a
[BigQuery table name](https://cloud.google.com/bigquery/docs/tables-intro).

For example:

``` bash
carto bigquery describe \
    --project my-gcp-project \
    --dataset my-bigquery-dataset \
    --table my-bigquery-table
```

### Using Raster Loader as a Python library

After installing Raster Loader, you can import the package into your Python project. For
example:

``` python
from raster_loader import rasterio_to_bigquery, bigquery_to_records
```

Currently, Raster Loader supports uploading raster data to [BigQuery](https://cloud.google.com/bigquery). Accessing BigQuery with Raster Loader requires the
`GOOGLE_APPLICATION_CREDENTIALS` environment variable to be set to the path of a JSON
file containing your BigQuery credentials. See the
[GCP documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key)
for more information.

You can use Raster Loader to upload a local raster file to an existing
BigQuery table using the `rasterio_to_bigquery()` function:

``` python
rasterio_to_bigquery(
    file_path = 'path/to/raster.tif',
    project_id = 'my-project',
    dataset_id = 'my_dataset',
    table_id = 'my_table',
)
```

This function returns `True` if the upload was successful.

You can also access and inspect a raster file from a BigQuery table using the
`bigquery_to_records()` function:

``` python
records_df = bigquery_to_records(
    project_id = 'my-project',
    dataset_id = 'my_dataset',
    table_id = 'my_table',
)
```

This function returns a DataFrame with some samples from the raster table on BigQuery
(10 rows by default).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information on how to contribute to this
project.

[ROADMAP.md](ROADMAP.md) contains a list of features and improvements planned for future
versions of Raster Loader.

## Releasing

### 1. Create and merge a release PR updating the CHANGELOG

- Branch: `release/X.Y.Z`
- Title: `Release vX.Y.Z`
- Description: CHANGELOG release notes

Example:
```
## [0.7.0] - 2024-06-02

### Added
- Support raster overviews (#140)

### Enhancements
- increase chunk-size to 10000 (#142)

### Bug Fixes
- fix: make the gdalwarp examples consistent (#143)
```

### 2. Create and push a tag `vX.Y.Z`

This will trigger an automatic workflow that will publish the package at https://pypi.org/project/raster-loader.

### 3. Create the GitHub release

Go to the tags page (https://github.com/CartoDB/raster-loader/tags), select the release tag and click on "Create a new release"

- Title: `vX.Y.Z`
- Description: CHANGELOG release notes

Example:
```
### Added
- Support raster overviews (#140)

### Enhancements
- increase chunk-size to 10000 (#142)

### Bug Fixes
- fix: make the gdalwarp examples consistent (#143)
```
