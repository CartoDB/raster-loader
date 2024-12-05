import numpy as np



MAX_LONGITUDE = 180.0
MIN_LONGITUDE = -MAX_LONGITUDE
MAX_LATITUDE = 89.0
MIN_LATITUDE = -MAX_LATITUDE


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

# def tile_to_cell(tile):
#     """Convert a tile into a cell.

#     Parameters
#     ----------
#     tile : tuple (x, y, z)

#     Returns
#     -------
#     int
#     """
#     HEADER = 0x4000000000000000
#     FOOTER = 0xFFFFFFFFFFFFF
#     B = [
#         0x5555555555555555,
#         0x3333333333333333,
#         0x0F0F0F0F0F0F0F0F,
#         0x00FF00FF00FF00FF,
#         0x0000FFFF0000FFFF,
#         0x00000000FFFFFFFF,
#     ]
#     S = [1, 2, 4, 8, 16]
#     if tile is None:
#         return None

#     x, y, z = tile

#     x = x << (32 - z)
#     y = y << (32 - z)

#     x = (x | (x << S[4])) & B[4]
#     y = (y | (y << S[4])) & B[4]

#     x = (x | (x << S[3])) & B[3]
#     y = (y | (y << S[3])) & B[3]

#     x = (x | (x << S[2])) & B[2]
#     y = (y | (y << S[2])) & B[2]

#     x = (x | (x << S[1])) & B[1]
#     y = (y | (y << S[1])) & B[1]

#     x = (x | (x << S[0])) & B[0]
#     y = (y | (y << S[0])) & B[0]

#     print('escalar', 'x', x, 'y', y)

#     # -- | (mode << 59) | (mode_dep << 57)
#     return HEADER | (1 << 59) | (z << 52) | ((x | (y << 1)) >> 12) | (FOOTER >> (z * 2))

# def tiles_to_cells(tiles, resolution):
#     """Convert tiles into cells.

#     Parameters
#     ----------
#     tiles : np.ndarray

#     Returns
#     -------
#     np.ndarray
#     """
#     if tiles is None:
#         return None
    
#     print('tiles ptts', tiles, resolution)

#     HEADER = 0x4000000000000000
#     FOOTER = 0xFFFFFFFFFFFFF
#     B = np.array([
#         0x5555555555555555,
#         0x3333333333333333,
#         0x0F0F0F0F0F0F0F0F,
#         0x00FF00FF00FF00FF,
#         0x0000FFFF0000FFFF,
#         0x00000000FFFFFFFF,
#     ])
#     S = np.array([1, 2, 4, 8, 16])

#     x = tiles[:, 0] << (32 - resolution)
#     y = tiles[:, 1] << (32 - resolution)

#     x = (x | (x << S[4])) & B[4]
#     y = (y | (y << S[4])) & B[4]

#     x = (x | (x << S[3])) & B[3]
#     y = (y | (y << S[3])) & B[3]

#     x = (x | (x << S[2])) & B[2]
#     y = (y | (y << S[2])) & B[2]

#     x = (x | (x << S[1])) & B[1]
#     y = (y | (y << S[1])) & B[1]

#     x = (x | (x << S[0])) & B[0]
#     y = (y | (y << S[0])) & B[0]

#     # print('x', x, 'y', y)

#     res = HEADER | (1 << 59) | (resolution << 52) | ((x | (y << 1)) >> 12) | (FOOTER >> (resolution * 2))
#     # print(res.dtype, res)
#     return res

# def tiles_to_cells(tiles, resolution):
#     """Convert tiles into cells."""
#     if tiles is None:
#         return None

#     HEADER = 0x4000000000000000
#     FOOTER = 0xFFFFFFFFFFFFF
#     B = np.array([
#         0x5555555555555555,
#         0x3333333333333333,
#         0x0F0F0F0F0F0F0F0F,
#         0x00FF00FF00FF00FF,
#         0x0000FFFF0000FFFF,
#         0x00000000FFFFFFFF,
#     ], dtype=np.uint64)
#     S = np.array([1, 2, 4, 8, 16], dtype=np.uint8)

#     x = np.uint64(tiles[:, 0]) << np.uint64(32 - resolution)
#     y = np.uint64(tiles[:, 1]) << np.uint64(32 - resolution)

#     for i in range(5):
#         x = (x | (x << np.uint64(S[i]))) & B[i]
#         y = (y | (y << np.uint64(S[i]))) & B[i]

#     res = HEADER | (1 << 59) | (resolution << 52) | ((x | (y << 1)) >> 12) | (FOOTER >> (resolution * 2))
#     print('tiles_to_cells', res)
#     return res

def tiles_to_cells(x, y, zoom):
    """
    Convert tile coordinates to quadbin at specific zoom level.
    This function follows Quadbin implementation developed
    in CARTO Analytics Toolbox (AT):
        https://github.com/CartoDB/analytics-toolbox-core
    """

    x = x << (32 - zoom)
    y = y << (32 - zoom)

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
    lons, lats = np.array([1.2]), np.array([-0.6])
    # lons, lats = np.array([-6, -6.5, 0, 12, 1.4, -1.2]), np.array([36.5, 37.1, 42, -25.1, -1.5, -0.6])
    zoom = 5
    res = points_to_cells(lons, lats, zoom)
    res2 = points_to_tile_fractions(lons, lats, zoom)
    print('res', res)
    print('res2', res2)

    import quadbin
    results = []
    results2= []
    for lon, lat in zip(lons, lats):
        result = quadbin.point_to_cell(lon, lat, zoom)
        result2 = quadbin.utils.point_to_tile_fraction(lon, lat, zoom)
        results.append(result)
        results2.append(result2)
    print('result', results)
    print('result2', results2)
