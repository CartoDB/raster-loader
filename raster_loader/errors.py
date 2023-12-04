def import_error_bigquery():  # pragma: no cover
    msg = (
        "Google Cloud BigQuery is not installed.\n"
        "Please install Google Cloud BigQuery to use this function.\n"
        "See https://googleapis.dev/python/bigquery/latest/index.html\n"
        "for installation instructions.\n"
        "OR, run `pip install google-cloud-bigquery` to install from pypi."
    )
    raise ImportError(msg)


def import_error_rasterio():  # pragma: no cover
    msg = (
        "Rasterio is not installed.\n"
        "Please install rasterio to use this function.\n"
        "See https://rasterio.readthedocs.io/en/latest/installation.html\n"
        "for installation instructions.\n"
        "Alternatively, run `pip install rasterio` to install from pypi."
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
            "-co RESAMPLING=NEAREST <input_raster>.tif <output_raster>.tif"
        )


def error_not_google_compatible():  # pragma: no cover
    raise IncompatibleRasterException()
