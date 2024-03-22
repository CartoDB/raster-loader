def import_error_bigquery():  # pragma: no cover
    msg = (
        "Google Cloud BigQuery is not installed.\n"
        "Please install Google Cloud BigQuery to use this function.\n"
        "run `pip install -U raster-loader[bigquery]` to install from pypi."
    )
    raise ImportError(msg)


def import_error_snowflake():  # pragma: no cover
    msg = (
        "Google Snowflake is not installed.\n"
        "Please install Snowflake to use this function.\n"
        "See https://docs.snowflake.com/en/developer-guide/python-connector\n"
        "run `pip install -U raster-loader[snowflake]` to install from pypi."
    )
    raise ImportError(msg)


def import_error_rio_cogeo():  # pragma: no cover
    msg = (
        "Rasterio is not installed.\n"
        "Please install rio_cogeo to use this function.\n"
        "See https://cogeotiff.github.io/rio-cogeo/\n"
        "for installation instructions.\n"
        "Alternatively, run `pip install rio-cogeo` to install from pypi."
    )
    raise ImportError(msg)


def import_error_quadbin():  # pragma: no cover
    msg = (
        "Quadbin is not installed.\n"
        "Please install quadbin to use this function.\n"
        "See https://github.com/CartoDB/quadbin-py\n"
        "for installation instructions.\n"
        "Alternatively, run `pip install quadbin` to install from pypi."
    )
    raise ImportError(msg)


class IncompatibleRasterException(Exception):
    def __init__(self):
        self.message = (
            "The input raster must be a GoogleMapsCompatible raster.\n"
            "You can make your raster compatible "
            "by converting it using the following command:\n"
            "gdalwarp -of COG -co TILING_SCHEME=GoogleMapsCompatible "
            "-co COMPRESS=DEFLATE -co OVERVIEWS=NONE -co ADD_ALPHA=NO "
            "-co RESAMPLING=NEAREST -co BLOCKSIZE=512 "
            "<input_raster>.tif <output_raster>.tif"
        )


def error_not_google_compatible():  # pragma: no cover
    raise IncompatibleRasterException()
