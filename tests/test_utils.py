import pytest
import pathlib

from datetime import datetime, timedelta
from raster_loader.errors import UploadError
from raster_loader.utils import some_function

HERE = pathlib.Path(__file__).parent


def test_some_function():
    data = some_function(HERE / "fixtures/token_ok.json")
    assert data == "eyJhbGciOiI6IkpX"
