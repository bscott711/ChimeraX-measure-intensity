"""Calculate distance theta and phi that describes a point from a centroid"""
from numpy import arccos, arctan, split, sqrt, subtract


def spherical_measurement(point, centroid):
    """Calculate distance theta and phi that describes a point from a centroid"""

    # pylint: disable-next=unbalanced-tuple-unpacking
    z_coord, y_coord, x_coord = split(subtract(point, centroid), 3, 1)

    z_squared = z_coord ** 2
    y_squared = y_coord ** 2
    x_squared = x_coord ** 2

    dist = sqrt(z_squared + y_squared + x_squared)
    theta = arccos(z_coord / dist)
    phi = arctan(x_coord / sqrt(y_squared + x_squared))

    return dist, theta, phi
