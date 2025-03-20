# raster-loader

[![PyPI version](https://badge.fury.io/py/raster-loader.svg)](https://badge.fury.io/py/raster-loader)
[![PyPI downloads](https://img.shields.io/pypi/dm/raster-loader.svg)](https://pypistats.org/packages/raster-loader)
[![Tests](https://github.com/cartodb/raster-loader/actions/workflows/ci.yml/badge.svg)](https://github.com/cartodb/raster-loader/actions)
[![Documentation Status](https://readthedocs.org/projects/raster-loader/badge/?version=latest)](https://raster-loader.readthedocs.io/en/latest/?badge=latest)

Python library for loading GIS raster data to standard cloud-based data warehouses that
don't natively support raster data.

Raster Loader is currently tested on Python 3.9, 3.10, 3.11 and 3.12.

## Documentation

The Raster Loader documentation is available at [raster-loader.readthedocs.io](https://raster-loader.readthedocs.io).

## Install

```bash
pip install -U raster-loader
```

To install from source:

```bash
git clone https://github.com/cartodb/raster-loader
cd raster-loader
pip install -U .
```

> **Tip**: In most cases, it is recommended to install Raster Loader in a virtual environment. Use [venv](https://docs.python.org/3/library/venv.html) to create and manage your virtual environment.

The above will install the dependencies required to work with all cloud providers (BigQuery, Snowflake, Databricks). If you only want to work with one of them, you can install the dependencies for each separately:

```bash
pip install -U raster-loader[bigquery]
pip install -U raster-loader[snowflake]
pip install -U raster-loader[databricks]
```

For Databricks, you will also need to install the [databricks-connect](https://pypi.org/project/databricks-connect/) package corresponding to your Databricks Runtime Version. For example, if your cluster uses DBR 15.1, install:

```bash
pip install databricks-connect==15.1
```

You can find your cluster's DBR version in the Databricks UI under Compute > Your Cluster > Configuration > Databricks Runtime version.
Or you can run the following SQL query from your cluster:

```sql
   SELECT current_version();
```

To verify the installation was successful, run:

```bash
carto info
```

This command will display system information including the installed Raster Loader version.

## Prerequisites

Before using Raster Loader with each platform, you need to have the following set up:

**BigQuery:**
- A [GCP project](https://cloud.google.com/resource-manager/docs/creating-managing-projects)
- A [BigQuery dataset](https://cloud.google.com/bigquery/docs/datasets-intro)
- The `GOOGLE_APPLICATION_CREDENTIALS` environment variable set to the path of a JSON file containing your BigQuery credentials. See the [GCP documentation](https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key) for more information.

**Snowflake:**
- A Snowflake account
- A Snowflake database
- A Snowflake schema

**Databricks:**
- A [Databricks server hostname](https://docs.databricks.com/aws/en/integrations/compute-details)
- A [Databricks cluster id](https://learn.microsoft.com/en-us/azure/databricks/workspace/workspace-details#cluster-url)
- A [Databricks token](https://docs.databricks.com/aws/en/dev-tools/auth/pat)

**Raster files**

The input raster must be a `GoogleMapsCompatible` raster. You can make your raster compatible by converting it with the following GDAL command:

```bash
gdalwarp -of COG -co TILING_SCHEME=GoogleMapsCompatible -co COMPRESS=DEFLATE -co OVERVIEWS=IGNORE_EXISTING -co ADD_ALPHA=NO -co RESAMPLING=NEAREST -co BLOCKSIZE=512 <input_raster>.tif <output_raster>.tif
```

Your raster file must be in a format that can be [read by GDAL](https://gdal.org/drivers/raster/index.html) and processed with [rasterio](https://rasterio.readthedocs.io/en/latest/).

## Usage

There are two ways you can use Raster Loader:

* Using the CLI by running `carto` in your terminal
* Using Raster Loader as a Python library (`import raster_loader`)

### CLI

After installing Raster Loader, you can run the CLI by typing `carto` in your terminal.

Currently, Raster Loader allows you to upload a local raster file to BigQuery, Snowflake, or Databricks tables. You can also download and inspect raster files from these platforms.

#### Uploading Raster Data

Examples for each platform:

**BigQuery:**
```bash
carto bigquery upload \
    --file_path /path/to/my/raster/file.tif \
    --project my-gcp-project \
    --dataset my-bigquery-dataset \
    --table my-bigquery-table \
    --overwrite
```

**Snowflake:**
```bash
carto snowflake upload \
    --file_path /path/to/my/raster/file.tif \
    --database my-snowflake-database \
    --schema my-snowflake-schema \
    --table my-snowflake-table \
    --account my-snowflake-account \
    --username my-snowflake-user \
    --password my-snowflake-password \
    --overwrite
```

Note that authentication parameters are explicitly required since they are not set up in the environment.

**Databricks:**
```bash
carto databricks upload \
    --file_path /path/to/my/raster/file.tif \
    --catalog my-databricks-catalog \
    --schema my-databricks-schema \
    --table my-databricks-table \
    --server-hostname my-databricks-server-hostname \
    --cluster-id my-databricks-cluster-id \
    --token my-databricks-token \
    --overwrite
```

Note that authentication parameters are explicitly required since they are not set up in the environment.

Additional features include:
- Specifying bands with `--band` and `--band_name`
- Enabling compression with `--compress` and `--compression-level`
- Chunking large uploads with `--chunk_size`

#### Inspecting Raster Data

To inspect a raster file stored in any platform, use the `describe` command:

**BigQuery:**
```bash
carto bigquery describe \
    --project my-gcp-project \
    --dataset my-bigquery-dataset \
    --table my-bigquery-table
```

**Snowflake:**
```bash
carto snowflake describe \
    --database my-snowflake-database \
    --schema my-snowflake-schema \
    --table my-snowflake-table \
    --account my-snowflake-account \
    --username my-snowflake-user \
    --password my-snowflake-password
```

Note that authentication parameters are explicitly required since they are not set up in the environment.

**Databricks:**
```bash
carto databricks describe \
    --catalog my-databricks-catalog \
    --schema my-databricks-schema \
    --table my-databricks-table \
    --server-hostname my-databricks-server-hostname \
    --cluster-id my-databricks-cluster-id \
    --token my-databricks-token
```

Note that authentication parameters are explicitly required since they are not set up in the environment.

For a complete list of options and commands, run `carto --help` or see the [full documentation](https://raster-loader.readthedocs.io/en/latest/user_guide/cli.html).

### Using Raster Loader as a Python library

After installing Raster Loader, you can use it in your Python project.

First, import the corresponding connection class for your platform:

```python
# For BigQuery
from raster_loader import BigQueryConnection

# For Snowflake
from raster_loader import SnowflakeConnection

# For Databricks
from raster_loader import DatabricksConnection
```

Then, create a connection object with the appropriate parameters:

```python
# For BigQuery
connection = BigQueryConnection('my-project')

# For Snowflake
connection = SnowflakeConnection('my-user', 'my-password', 'my-account', 'my-database', 'my-schema')

# For Databricks
connection = DatabricksConnection('my-server-hostname', 'my-token', 'my-cluster-id')
```

#### Uploading a raster file

To upload a raster file, use the `upload_raster` function:

```python
connection.upload_raster(
    file_path = 'path/to/raster.tif',
    fqn = 'database.schema.tablename'
)
```

This function returns `True` if the upload was successful.

You can enable compression of the band data to reduce storage size:

```python
connection.upload_raster(
    file_path = 'path/to/raster.tif',
    fqn = 'database.schema.tablename',
    compress = True,  # Enable gzip compression of band data
    compression_level = 3  # Optional: Set compression level (1-9, default=6)
)
```

#### Inspecting a raster file

To access and inspect a raster file stored in any platform, use the `get_records` function:

```python
records = connection.get_records(
    fqn = 'database.schema.tablename'
)
```

This function returns a DataFrame with some samples from the raster table (10 rows by default).

For more details, see the [full documentation](https://raster-loader.readthedocs.io/en/latest/user_guide/use_with_python.html).

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
