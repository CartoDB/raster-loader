import pyproj
import rasterio


def calculate_coordinate(pyproj_transformer, dataset_transform, row, col):
    """Calculate the coordinate of a pixel in a raster.

    Args:
        pyproj_transformer (pyproj.Transformer): A pyproj transformer object.
        dataset_transform (rasterio.transform.Affine): A rasterio transform object.
        row (int): The row of the pixel.
        col (int): The column of the pixel.

    Returns:
        tuple: The latitude and longitude of the pixel.
    """
    return pyproj_transformer.transform(
        *rasterio.transform.xy(dataset_transform, row, col)
    )


def chunks(lst, n):
    """Yield successive n-sized chunks from lst.

    Args:
        lst (list): A list of elements.
        n (int): The number of elements in each chunk.

    Returns:
        list: A list of chunks.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_dataset_crs(dataset):
    """Get the CRS of a raster dataset.

    Args:
        dataset (rasterio.io.DatasetReader): A rasterio dataset object.

    Returns:
        pyproj.CRS: A pyproj CRS object.
    """
    return pyproj.CRS.from_wkt(dataset.crs.to_wkt())
