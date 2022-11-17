import pytest
import pathlib

from datetime import datetime, timedelta

from raster_loader import RasterLoader, UploadError

HERE = pathlib.Path(__file__).parent


def test_some():
    assert True
