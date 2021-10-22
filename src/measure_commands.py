from chimerax.color_key import show_key
from chimerax.core import colors
from chimerax.core.commands import (BoolArg, Bounded, CmdDesc, ColormapArg,
                                    ColormapRangeArg, IntArg, SurfacesArg)
from chimerax.core.commands.cli import EnumOf
from numpy import (array, full, inf, nanmax, nanmean, nanmin,
                   ravel_multi_index, swapaxes)
from scipy.spatial import KDTree


def distance_series(session, surface, to_surface, knn=5, palette=None,      range=None, key=False):
    """Wrap the distance measurement for list of surfaces."""
    [measure_distance(surface, to_surface, knn)
     for surface, to_surface in zip(surface, to_surface)]
    recolor_surfaces(session, surface, 'distance', palette, range, key)


def intensity_series(session, surface, to_map, radius=15, palette=None, range=None, key=False):
    """Wrap the intensity measurement for list of surfaces."""
    [measure_intensity(surface, to_map, radius)
     for surface, to_map in zip(surface, to_map)]
    recolor_surfaces(session, surface, 'intensity', palette, range, key)


def recolor_surfaces(session, surface, metric='intensity', palette=None, range=None, key=False):
    """Wraps recolor_surface in a list comprehension"""
    keys = full(len(surface), False)
    keys[0] = key
    [recolor_surface(session, surface, metric, palette, range, key)
     for surface, key in zip(surface, keys)]


def recolor_surface(session, surface, metric, palette, range, key):
    """Colors surface based on previously measured intensity or distance"""
    if metric == 'distance' and hasattr(surface, 'distance'):
        measurement = surface.distance
        palette_string = 'brbg'
        max_range = 15
    elif metric == 'intensity' and hasattr(surface, 'intensity'):
        measurement = surface.intensity
        palette_string = 'purples'
        max_range = 5
    else:
        return

    if palette is None:
        palette = colors.BuiltinColormaps[palette_string]

    if range is not None and range != 'full':
        rmin, rmax = range
    elif range == 'full':
        rmin, rmax = nanmin(measurement), nanmax(measurement)
    else:
        rmin, rmax = (0, max_range)

    cmap = palette.rescale_range(rmin, rmax)
    surface.vertex_colors = cmap.interpolated_rgba8(measurement)

    if key:
        show_key(session, cmap)


def measure_distance(surface, to_surface, knn):
    """Measure the local motion of two surfaces."""
    distance, _ = query_tree(surface.vertices, to_surface.vertices, knn=knn)
    surface.distance = distance


def measure_intensity(surface, to_map, radius):
    """Measure the local intensity within radius r of the surface."""
    image_info = get_image(surface, to_map)
    image_coords, *flattened_indices = get_coords(*image_info)
    _, index = query_tree(surface.vertices, image_coords.T, radius)
    face_intensity = local_intensity(*flattened_indices, index)
    surface.intensity = face_intensity


def get_image(surface, to_map):
    """Get the isosurface volume mask and secondary channel."""
    mask_vol = surface.volume.full_matrix().copy()
    image_3d = to_map.volume.full_matrix().copy()
    level = surface.volume.maximum_surface_level
    return mask_vol, level, image_3d


def get_coords(mask, level, image_3d):
    """Get the coords for local intensity"""
    # Mask the secondary channel based on the isosurface
    image_3d *= (mask >= level)
    # ChimeraX uses XYZ for image, but numpy uses ZYX, swap dims
    image_3d = swapaxes(image_3d, 0, 2)
    image_coords = array(image_3d.nonzero())
    flattened_image = image_3d.flatten()
    pixel_indices = ravel_multi_index(image_coords, image_3d.shape)
    return image_coords, flattened_image, pixel_indices


def query_tree(init_verts, to_map, radius=inf, knn=200):
    """Create a KDtree from a set of points and query for nearest neighbors.
    Returns:
    index: index of nearest neighbors within radius
    distance: Mean distance of nearest neighbors"""
    tree = KDTree(to_map)
    dist, index = tree.query(init_verts, k=range(
        1, knn), distance_upper_bound=radius, workers=-1)
    dist[dist == inf] = None
    distance = nanmean(dist, axis=1)
    index = array([_index(ind, tree.n) for ind in index], dtype=object)
    return distance, index


def _index(index, tree_max):
    """Tree query pads with tree_max if there are no neighbors."""
    index = index[index < tree_max]
    return index


def local_intensity(flat_img, pixels, index):
    """Measure local mean intensity normalized to mean of all."""
    face_int = array([nanmean(flat_img[pixels[ind]]) for ind in index])
    return face_int/face_int.mean()


measure_distance_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('to_surface', SurfacesArg),
             ('knn', Bounded(IntArg)),
             ('palette', ColormapArg),
             ('range', ColormapRangeArg),
             ('key', BoolArg)],
    required_arguments=['to_surface'],
    synopsis='Measure local distance between two surfaces')


measure_intensity_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('to_map', SurfacesArg),
             ('radius', Bounded(IntArg, 1, 30)),
             ('palette', ColormapArg),
             ('range', ColormapRangeArg),
             ('key', BoolArg)],
    required_arguments=['to_map'],
    synopsis='Measure local intensity relative to surface')


recolor_surfaces_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('metric', EnumOf(['intensity', 'distance'])),
             ('palette', ColormapArg),
             ('range', ColormapRangeArg),
             ('key', BoolArg)],
    synopsis='Recolor surface based on previous measurement')
