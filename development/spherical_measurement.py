"""Calculate distance theta and phi that describes a point from a centroid"""
"""Test Data location X:\Phagocytosis\sRBC\20190614cs1\track_4\test"""

from numpy import arccos, arctan, split, sqrt, subtract

"""Calculate distance theta and phi that describes a point from a centroid"""


def spherical_measurement(surface, to_cell):
  
    """target centroid: method BLS measure distance..."""
    """to_cell: target image we are using to generate a center point"""
    """surface: the cenn of interest to consider for making our measurments"""

    centroid = to_cell.vertices.mean()

    # pylint: disable-next=unbalanced-tuple-unpacking
    x_coord, y_coord, z_coord = split(subtract(centroid, surface.vertices), 3, 1)

    z_squared = z_coord ** 2
    y_squared = y_coord ** 2
    x_squared = x_coord ** 2

    dist = sqrt(z_squared + y_squared + x_squared)

    distxy = sqrt(x_squared + y_squared)

    theta = sign(y_coord)*arccos(x_coord / distxy)

    phi = arccos(z_coord / dist)

    return dist, theta, phi
