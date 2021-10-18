from chimerax.color_key import show_key
from chimerax.core.colors import BuiltinColormaps
from chimerax.core.commands import (BoolArg, CmdDesc, ColormapArg,
                                    ColormapRangeArg, FloatArg, SurfaceArg,
                                    register)
from numpy import array, inf, nan, nanmedian
from scipy.spatial import KDTree


def measure_distance(session, surface, to_surface, radius=15, palette=None, range=None, key=None):
    """Measure the local motion within radius r of two surfaces."""
    _, distance = query_tree(surface.vertices, to_surface.vertices, radius)
    surface.distance = distance

    if palette is None:
        palette = BuiltinColormaps['purples-8']

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


def query_tree(init_verts, to_map, radius=50, k_nn=200):
    """Create a KDtree from a set of points, and query for nearest neighbors within a given radius.
    Returns:
    index: index of nearest neighbors
    distance: Median distance of neighbors from local search area"""
    tree = KDTree(to_map)
    dist, index = tree.query(
        init_verts, k=range(1, k_nn), distance_upper_bound=radius, workers=-1)
    dist[dist == inf] = nan
    distance = nanmedian(dist, axis=1)
    index = array([_remove_index(ind, tree.n)
                   for ind in index], dtype=object)
    return index, distance


def _remove_index(index, tree_max):
    """tree query pads with tree_max if there are no neighbors."""
    index = index[index < tree_max]
    return index


def register_command(logger):
    """Register command for use in ChimeraX"""
    desc = CmdDesc(
        required=[('surface', SurfaceArg)],
        keyword=[('to_map', SurfaceArg),
                 ('radius', FloatArg),
                 ('palette', ColormapArg),
                 ('c_range', ColormapRangeArg),
                 ('key', BoolArg)],
        required_arguments=['to_map'],
        synopsis='measure local surface distance')
    register('measure distance', desc, measure_distance, logger=logger)
