def import_error_bigquery():  # pragma: no cover
    msg = (
        "Google Cloud BigQuery client is not installed.\n"
        "Please install Google Cloud BigQuery dependencies to use this function.\n"
        'run `pip install -U raster-loader"[bigquery]"` to install from pypi.'
    )
    raise ImportError(msg)


def import_error_snowflake():  # pragma: no cover
    msg = (
        "Snowflake client is not installed.\n"
        "Please install Snowflake dependencies to use this function.\n"
        'run `pip install -U raster-loader"[snowflake]"` to install from pypi.'
    )
    raise ImportError(msg)


def import_error_databricks():
    raise ImportError(
        "The databricks-connect package is required and must match your "
        "Databricks Runtime version.\n"
        "For example, if your cluster uses DBR 15.1, "
        "run `pip install databricks-connect==15.1`\n"
        "You can find your cluster's DBR version in the Databricks UI "
        "under Compute > Your Cluster > Configuration > Databricks Runtime version.\n"
        "Or you can run the following SQL query from your cluster:\n"
        "`SELECT current_version()`"
    )


class IncompatibleRasterException(Exception):
    def __init__(self):
        self.message = (
            "The input raster must be a GoogleMapsCompatible raster.\n"
            "You can make your raster compatible "
            "by converting it using the following command:\n"
            "gdalwarp -of COG -co TILING_SCHEME=GoogleMapsCompatible "
            "-co COMPRESS=DEFLATE -co OVERVIEWS=IGNORE_EXISTING -co ADD_ALPHA=NO "
            "-co RESAMPLING=NEAREST -co BLOCKSIZE=512 "
            "<input_raster>.tif <output_raster>.tif"
        )


def error_not_google_compatible():  # pragma: no cover
    raise IncompatibleRasterException()
