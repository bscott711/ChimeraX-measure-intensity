from chimerax.color_key import show_key
from chimerax.core import colors
from chimerax.core.commands import (BoolArg, CmdDesc, ColormapArg,
                                    ColormapRangeArg, FloatArg, ModelArg,
                                    SurfaceArg)
from numpy import (array, inf, nanmax, nanmean, nanmedian, nanmin,
                   ravel_multi_index, swapaxes)
from scipy.spatial import KDTree


def measure_distance(session, surface, to_surface, radius=15, palette=None, range=None, key=None):
    """Measure the local motion within radius r of two surfaces."""
    _, distance = query_tree(surface.vertices, to_surface.vertices, radius)
    surface.distance = distance

    if palette is None:
        palette = colors.BuiltinColormaps['purples']

    if range is not None and range != 'full':
        rmin, rmax = range
    elif range == 'full':
        rmin, rmax = distance.min(), distance.max()
    else:
        rmin, rmax = (0, radius)

    cmap = palette.rescale_range(rmin, rmax)
    surface.vertex_colors = cmap.interpolated_rgba8(distance)

    if key:
        show_key(session, cmap)


def measure_intensity(session, surface, to_map, radius=15, palette=None, range=None, key=None):
    """Measure the local intensity within radius r of the surface."""
    image_info = get_image(surface, to_map)
    image_coords, *flattened_indices = get_coords(*image_info)
    index, _ = query_tree(surface.vertices, image_coords.T, radius)
    face_intensity = local_intensity(*flattened_indices, index)

    if palette is None:
        palette = colors.BuiltinColormaps['purples']

    if range is not None and range != 'full':
        rmin, rmax = range
    elif range == 'full':
        rmin, rmax = nanmin(face_intensity), nanmax(face_intensity)
    else:
        rmin, rmax = (0, 3)

    cmap = palette.rescale_range(rmin, rmax)
    surface.vertex_colors = cmap.interpolated_rgba8(face_intensity)
    surface.face_intensity = face_intensity

    if key:
        show_key(session, cmap)


def get_image(surface, to_map):
    """Get the isosurface volume mask and secondary channel."""
    mask_vol = surface.volume.full_matrix().copy()
    level = surface.volume.maximum_surface_level
    image_3d = to_map.full_matrix().copy()
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


def query_tree(init_verts, to_map, radius=50, k_nn=200):
    """Create a KDtree from a set of points, and query for nearest neighbors within a given radius.
    Returns:
    index: index of nearest neighbors
    distance: Median distance of neighbors from local search area"""
    tree = KDTree(to_map)
    dist, index = tree.query(
        init_verts, k=range(1, k_nn), distance_upper_bound=radius, workers=-1)
    dist[dist == inf] = None
    distance = nanmedian(dist, axis=1)
    index = array([_remove_index(ind, tree.n)
                   for ind in index], dtype=object)
    return index, distance


def _remove_index(index, tree_max):
    """tree query pads with tree_max if there are no neighbors."""
    index = index[index < tree_max]
    return index


def local_intensity(flattened_image, pixel_indices, index):
    """Measure local mean intensity normalized to mean of all."""
    face_intensities = array(
        [nanmean(flattened_image[pixel_indices[ind]]) for ind in index])
    return face_intensities/nanmean(face_intensities)


measure_distance_desc = CmdDesc(
    required=[('surface', SurfaceArg)],
    keyword=[('to_surface', SurfaceArg),
             ('radius', FloatArg),
             ('palette', ColormapArg),
             ('range', ColormapRangeArg),
             ('key', BoolArg)],
    required_arguments=['to_surface'],
    synopsis='measure local distance between two surfaces')


measure_intensity_desc = CmdDesc(
    required=[('surface', SurfaceArg)],
    keyword=[('to_map', ModelArg),
             ('radius', FloatArg),
             ('palette', ColormapArg),
             ('range', ColormapRangeArg),
             ('key', BoolArg)],
    required_arguments=['to_map'],
    synopsis='measure local intensity relative to surface')
