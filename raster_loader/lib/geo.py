import math
import functools
import json


def coord_range(start_x, start_y, end_x, end_y, num_subdivisions):
    num_subdivisions = max(num_subdivisions, 1)
    return [
        [
            start_x + (end_x - start_x) * i / num_subdivisions,
            start_y + (end_y - start_y) * i / num_subdivisions,
        ]
        for i in range(0, num_subdivisions + 1)
    ]


def norm_lon(x):
    return x - 360.0 if x > 180.0 else x + 360.0 if x <= -180.0 else x


def norm_coords(coords):
    return [[norm_lon(point[0]), point[1]] for point in coords]


def polygon_geography(rings, format, normalize_coords):
    if normalize_coords:
        rings = [norm_coords(coords) for coords in rings]

    if format == "wkt":
        return polygon_wkt(rings)
    elif format == "geojson":
        return polygon_geojson(rings)
    else:
        raise ValueError(f"Invalid geography format {format}")


def polygon_wkt(rings):
    return (
        "POLYGON("
        + ",".join(
            [
                "("
                + ",".join(
                    [" ".join([str(coord) for coord in point]) for point in coords]
                )
                + ")"
                for coords in rings
            ]
        )
        + ")"
    )


def polygon_geojson(rings):
    return json.dumps({"type": "Polygon", "coordinates": rings})


def coords_to_geography(coords, format, whole_earth):
    # remove too-close coordinates cause they cause errors
    # in BigQuery's ST_GEOGFROMGEOJSON
    def are_too_close(point1, point2):
        return (
            math.fabs(point1[0] - point2[0]) <= 1e-13
            and math.fabs(point1[1] - point2[1]) <= 1e-13
        )

    def filter_near_points(coords, point):
        previous = None if not coords else coords[-1]
        if not previous or not are_too_close(previous, point):
            coords.append(point)
        return coords

    coords = functools.reduce(filter_near_points, coords, [])

    # now let's make sure the initial and final points are exactly the same
    if coords[0] != coords[-1]:
        # replace the last point; never mind, it must be very close
        coords[-1] = coords[0]
    return polygon_geography([coords], format, not whole_earth)


def raster_bounds(raster_dataset, transformer, format):
    min_x = 0
    min_y = 0
    max_x = raster_dataset.width
    max_y = raster_dataset.height

    x_subdivisions = math.ceil((max_x - min_x) / 64.0)
    y_subdivisions = math.ceil((max_y - min_y) / 64.0)
    pixel_coords = (
        # SW -> SE
        coord_range(min_x, max_y, max_x, max_y, x_subdivisions)
        # SE -> NE
        + coord_range(max_x, max_y, max_x, min_y, y_subdivisions)
        # NE -> NW
        + coord_range(max_x, min_y, min_x, min_y, x_subdivisions)
        # NW -> SW
        + coord_range(min_x, min_y, min_x, max_y, y_subdivisions)
    )
    coords = [
        transformer.transform(*(raster_dataset.transform * (x, y)))
        for x, y in pixel_coords
    ]
    lon_NW, _ = transformer.transform(*(raster_dataset.transform * (min_x, min_y)))
    lon_NE, _ = transformer.transform(*(raster_dataset.transform * (max_x, min_y)))
    lon_SW, _ = transformer.transform(*(raster_dataset.transform * (min_x, max_y)))
    lon_SE, _ = transformer.transform(*(raster_dataset.transform * (max_x, max_y)))
    whole_earth = (
        math.fabs(lon_NW - lon_NE) >= 360.0 and math.fabs(lon_SW - lon_SE) >= 360
    )

    return coords_to_geography(coords, format, whole_earth)
