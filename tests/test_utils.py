import pyproj
import rasterio

from raster_loader import utils


def test_calculate_coordinate():
    raster_dataset = rasterio.open("tests/fixtures/mosaic.tif")

    src_crs = pyproj.CRS.from_wkt(raster_dataset.crs.to_wkt())
    transformer = pyproj.Transformer.from_crs(src_crs, 4326)

    window = [window for _, window in raster_dataset.block_windows()][0]

    coordinates = utils.calculate_coordinate(
        transformer,
        raster_dataset.transform,
        window.row_off + 0,
        window.col_off + 0,
    )

    assert coordinates == (11.347432500000002, 46.2529675)


def test_chunks():
    raster_dataset = rasterio.open("tests/fixtures/mosaic.tif")

    windows = [window for _, window in raster_dataset.block_windows()]

    chunks = utils.chunks(windows, 100)
    assert len([i for i in chunks][0]) == 90


def test_get_dataset_crs():
    raster_dataset = rasterio.open("tests/fixtures/mosaic.tif")

    crs = utils.get_dataset_crs(raster_dataset)

    assert crs == pyproj.CRS.from_wkt(raster_dataset.crs.to_wkt())
