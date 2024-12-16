import numpy as np
import quadbin


MAX_LONGITUDE = 180.0
MIN_LONGITUDE = -MAX_LONGITUDE
MAX_LATITUDE = 89.0
MIN_LATITUDE = -MAX_LATITUDE


def tiles_to_longitudes(tiles, offset):
    """Compute the longitudes for a set of tiles with an offset.

    Parameters
    ----------
    tiles : np.ndarray [[x, y, z]]
    offset : float
        Inner position of the tile. From 0 to 1.

    Returns
    -------
    longitudes : np.ndarray
    """
    x, _, z = tiles.T
    return 180 * (2.0 * (x + offset) / (1 << z) - 1.0)


def tiles_to_latitudes(tiles, offset):
    """Compute the latitudes for a set of tiles with an offset.

    Parameters
    ----------
    tiles : np.ndarray [[x, y, z]]
    offset : float
        Inner position of the tile. From 0 to 1.

    Returns
    -------
    latitudes : np.ndarray
    """
    _, y, z = tiles.T
    expy = np.exp(-(2.0 * (y + offset) / (1 << z) - 1) * np.pi)
    return 360 * (np.arctan(expy) / np.pi - 0.25)


def cells_to_bounding_boxes(cells):
    """Convert cells into geographic bounding boxes.

    Parameters
    ----------
    cells : np.ndarray

    Returns
    -------
    np.ndarray
        Bounding boxes in degrees [[xmin, ymin, xmax, ymax], ...]
    """
    tiles = np.array([cells_to_tiles(cell) for cell in cells])

    xmins = tiles_to_longitudes(tiles, 0)
    xmaxs = tiles_to_longitudes(tiles, 1)
    ymins = tiles_to_latitudes(tiles, 1)
    ymaxs = tiles_to_latitudes(tiles, 0)

    return np.column_stack((xmins, ymins, xmaxs, ymaxs))


def clip_numbers(nums, lower, upper):
    """Limit input numbers by lower and upper limits.

    Parameters
    ----------
    nums : np.ndarray
    lower : float
    upper : float

    Returns
    -------
    np.ndarray
    """
    # return np.clip(nums, lower, upper)
    # return np.minimum(upper, np.(lower, nums))
    return np.maximum(np.minimum(nums, upper), lower)


def clip_longitudes(longitudes):
    """Limit longitudes bounds.

    Parameters
    ----------
    longitudes : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return clip_numbers(longitudes, MIN_LONGITUDE, MAX_LONGITUDE)


def clip_latitudes(latitudes):
    """Limit latitudes by the web mercator bounds.

    Parameters
    ----------
    latitudes : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return clip_numbers(latitudes, MIN_LATITUDE, MAX_LATITUDE)


def points_to_tiles(longitudes, latitudes, resolution):
    """Compute the tiles for longitudes and latitudes in a specific resolution.

    Parameters
    ----------
    longitudes : np.ndarray
        Longitudes in decimal degrees.
    latitudes : np.ndarray
        Latitudes in decimal degrees.
    resolution : int
        The resolution of the tiles.

    Returns
    -------
    tiles: np.ndarray
    """
    x, y, z = points_to_tile_fractions(longitudes, latitudes, resolution)

    x = np.floor(x).astype('uint64')
    y = np.floor(y).astype('uint64')

    return x, y, z


def points_to_cells(longitudes, latitudes, resolution):
    """Convert geographic points into cells.

    Parameters
    ----------
    longitudes : np.ndarray
        Longitudes in decimal degrees.
    latitudes : np.ndarray
        Latitudes in decimal degrees.
    resolution : int
        The resolution of the cells.

    Returns
    -------
    np.ndarray

    Raises
    ------
    ValueError
        If the resolution is out of bounds.
    """
    if resolution < 0 or resolution > 26:
        raise ValueError("Invalid resolution: should be between 0 and 26")

    longitudes = clip_longitudes(longitudes)
    latitudes = clip_latitudes(latitudes)

    x, y, z = points_to_tiles(longitudes, latitudes, resolution)

    return tiles_to_cells(x, y, z)


def get_resolution(indexes):
    """Get the resolution of an index.

    Parameters
    ----------
    indexes : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return (indexes >> 52) & 0x1F


def cells_to_parent(cells, parent_resolution):
    """Compute the parent cell for a specific resolution.

    Parameters
    ----------
    cells : np.ndarray
    parent_resolution : int

    Returns
    -------
    np.ndarray

    Raises
    ------
    ValueError
        If the parent resolution is not valid.
    """
    resolution = get_resolution(cells)
    parent_resolution = np.full_like(resolution, parent_resolution)

    if np.any(parent_resolution < 0) or np.any(parent_resolution > resolution):
        raise ValueError("Invalid resolution")

    return (
        (cells & ~(0x1F << 52))
        | (parent_resolution << 52)
        | (0xFFFFFFFFFFFFF >> (parent_resolution << 1))
    )


def cells_to_children(cells, children_resolution):
    """Compute the children cells for a specific resolution.

    Parameters
    ----------
    cells : np.ndarray
    children_resolution : int

    Returns
    -------
    np.ndarray
        Children cells.

    Raises
    ------
    ValueError
        If the children resolution is not valid.
    """
    cells_resolution = (cells >> 52) & 0x1F
    if np.unique(cells_resolution).size > 1:
        raise ValueError("Cells have different resolutions")

    resolution = int(cells_resolution[0])

    if (
        children_resolution < 0
        or children_resolution > 26
        or children_resolution <= resolution
    ):
        raise ValueError("Invalid resolution")

    resolution_diff = children_resolution - resolution
    block_range = 1 << (resolution_diff << 1)
    block_shift = 52 - (children_resolution << 1)

    child_base = (cells.astype('int64') & ~(0x1F << 52)) | (children_resolution << 52)
    child_base = child_base & ~((block_range - 1) << block_shift)

    children = child_base[:, np.newaxis] | (np.arange(block_range, dtype=child_base.dtype) << block_shift)

    return children


def tiles_to_cells(x, y, zoom):
    """
    Convert tile coordinates to quadbin at specific zoom level.
    This function follows Quadbin implementation developed
    in CARTO Analytics Toolbox (AT):
        https://github.com/CartoDB/analytics-toolbox-core
    """
    x = x.astype('uint64') << (32 - zoom)
    y = y.astype('uint64') << (32 - zoom)

    # interleaved1
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    y = (y | (y << 16)) & 0x0000FFFF0000FFFF

    # interleaved2
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    y = (y | (y << 8)) & 0x00FF00FF00FF00FF

    # interleaved3
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    y = (y | (y << 4)) & 0x0F0F0F0F0F0F0F0F

    # interleaved4
    x = (x | (x << 2)) & 0x3333333333333333
    y = (y | (y << 2)) & 0x3333333333333333

    # interleaved5
    x = (x | (x << 1)) & 0x5555555555555555
    y = (y | (y << 1)) & 0x5555555555555555

    return 0x4000000000000000 | (1 << 59) | (zoom << 52) | ((x | (y << 1)) >> 12) | (0xFFFFFFFFFFFFF >> (zoom * 2))


def cells_to_tiles(cells):
    """Convert cells into tiles.

    Parameters
    ----------
    cells : np.ndarray

    Returns
    -------
    tiles : np.ndarray
    """
    z = (cells >> 52) & 31
    q = (cells & 0xFFFFFFFFFFFFF) << 12
    x = q
    y = q >> 1

    x = x & 0x5555555555555555
    y = y & 0x5555555555555555

    x = (x | (x >> 1)) & 0x3333333333333333
    y = (y | (y >> 1)) & 0x3333333333333333

    x = (x | (x >> 2)) & 0x0F0F0F0F0F0F0F0F
    y = (y | (y >> 2)) & 0x0F0F0F0F0F0F0F0F

    x = (x | (x >> 4)) & 0x00FF00FF00FF00FF
    y = (y | (y >> 4)) & 0x00FF00FF00FF00FF

    x = (x | (x >> 8)) & 0x0000FFFF0000FFFF
    y = (y | (y >> 8)) & 0x0000FFFF0000FFFF

    x = (x | (x >> 16)) & 0x00000000FFFFFFFF
    y = (y | (y >> 16)) & 0x00000000FFFFFFFF

    x = x >> (32 - z)
    y = y >> (32 - z)

    return np.column_stack((x, y, z))


def points_to_tile_fractions(longitudes, latitudes, resolution):
    """Compute the tiles in fractions for longitudes and latitudes in a specific resolution.

    Parameters
    ----------
    longitudes : np.ndarray
        Longitudes in decimal degrees.
    latitudes : np.ndarray
        Latitudes in decimal degrees.
    resolution : int
        The resolution of the tiles.

    Returns
    -------
    tiles: np.ndarray
    """
    z = resolution
    z2 = 1 << z
    sinlat = np.sin(latitudes * np.pi / 180.0)
    x = z2 * (longitudes / 360.0 + 0.5)
    yfraction = 0.5 - 0.25 * np.log((1 + sinlat) / (1 - sinlat)) / np.pi
    y = clip_numbers(z2 * yfraction, 0, z2 - 1)

    x = np.mod(x, z2)
    x = np.where(x < 0, x + z2, x)

    return x, y, z


if __name__ == "__main__":
    lons = np.array([-6.0, -6.5, 0.0, 12, 1.4, -1.2])
    lats = np.array([36.5, 37.1, 42, -25.1, -1.5, -0.6])
    zoom = 22
    ressult_v = points_to_cells(lons, lats, zoom)
    print('Vectorized function result:', ressult_v)

    result_s = []
    for lon, lat in zip(lons, lats):
        res = quadbin.point_to_cell(lon, lat, zoom)
        result_s.append(res)
    print('Scalar function result:', result_s)
