def measure_distance(session, surface, to_map, radius=50, palette=None, c_range=None, key=None):
    # All command functions are invoked with ``session`` as its
    # first argument.  Useful session attributes include:
    #   logger: chimerax.core.logger.Logger instance
    #   models: chimerax.core.models.Models instance
    if palette is None:
        from chimerax.core import colors
        palette = colors.BuiltinColormaps['ylgnbu-5']
    if c_range is None:
        c_range = (0, radius)
    rmin, rmax = c_range
    cmap = palette.rescale_range(rmin, rmax)

    _, distance = query_tree(surface.vertices, to_map.vertices, radius)
    surface.vertex_colors = cmap.interpolated_rgba8(distance)

    if key:
        from chimerax.color_key import show_key
        show_key(session, cmap)
    return distance


def query_tree(init_verts, to_map, radius=50, k_nn=200):
    from numpy import array, nanmedian, inf, nan
    from scipy.spatial import KDTree
    """Create a KDtree from a set of points, and query for nearest neighbors within a given radius.
    Returns:
    index: index of nearest neighbors
    distance: Median distance of neighbors from local search area"""
    tree = KDTree(to_map)
    dist, index = tree.query(
        init_verts, k=range(1, k_nn), distance_upper_bound=radius, workers=-1)
    dist[dist == inf] = nan
    distance = nanmedian(dist, axis=1)
    index = array([remove_index(ind, tree.n)
                   for ind in index], dtype=object)
    return index, distance


def remove_index(index, tree_max):
    """tree query pads with tree_max if there are no neighbors."""
    index = index[index < tree_max]
    return index


def register_command(session):
    from chimerax.core.commands import CmdDesc, register, SurfaceArg, ColormapArg, ColormapRangeArg, BoolArg, FloatArg
    desc = CmdDesc(
        required=[('surface', SurfaceArg)],
        keyword=[('to_map', SurfaceArg),
                 ('radius', FloatArg),
                 ('palette', ColormapArg),
                 ('c_range', ColormapRangeArg),
                 ('key', BoolArg)],
        required_arguments=['to_map'],
        synopsis='measure local surface distance')
    register('measure distance', desc, measure_distance, logger=session.logger)


register_command(session)
