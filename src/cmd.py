def measure_distance(session, surface, to_surface, radius=15, palette=None, range=None, key=None):
    """Measure the local motion within radius r of two surfaces."""
    _, distance = query_tree(surface.vertices, to_surface.vertices, radius)
    surface.distance = distance

    if palette is None:
        from chimerax.core import colors
        palette = colors.BuiltinColormaps['purples-8']

    if range is not None and range != 'full':
        rmin, rmax = range
    elif range == 'full':
        rmin, rmax = distance.min(), distance.max()
    else:
        rmin, rmax = (0, radius)

    cmap = palette.rescale_range(rmin, rmax)
    surface.vertex_colors = cmap.interpolated_rgba8(distance)

    if key:
        from chimerax.color_key import show_key
        show_key(session, cmap)


def measure_intensity(session, surface, to_map, radius=9, palette=None, range=None, key=None):
    from numpy import nanmin, nanmax
    """Measure the local intensity within radius r of the surface."""
    image_coords, flattened_image, pixel_indices = get_coords(to_map)
    index, _ = query_tree(surface.vertices, image_coords.T, radius)
    face_intensity = local_intensity(flattened_image, pixel_indices, index)

    if palette is None:
        from chimerax.core import colors
        palette = colors.BuiltinColormaps['spectral-5']

    if range is not None and range != 'full':
        rmin, rmax = range
    elif range == 'full':
        rmin, rmax = nanmin(face_intensity), nanmax(face_intensity)
    else:
        rmin, rmax = (0.8, 1.2)

    cmap = palette.rescale_range(rmin, rmax)
    surface.vertex_colors = cmap.interpolated_rgba8(face_intensity)
    surface.face_intensity = face_intensity

    if key:
        from chimerax.color_key import show_key
        show_key(session, cmap)


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
    index = array([_remove_index(ind, tree.n)
                   for ind in index], dtype=object)
    return index, distance


def _remove_index(index, tree_max):
    """tree query pads with tree_max if there are no neighbors."""
    index = index[index < tree_max]
    return index


def get_coords(volume):
    """Get the coords for local intensity"""
    from numpy import array, ravel_multi_index, swapaxes
    level = volume.maximum_surface_level
    image_3d = volume.full_matrix()
    # ChimeraX uses XYZ for image, but numpy uses ZYX, swap dims
    image_3d = swapaxes(image_3d, 0, 2)
    image_3d *= (image_3d >= level)
    image_coords = array(image_3d.nonzero())
    flattened_image = image_3d.flatten()
    pixel_indices = ravel_multi_index(image_coords, image_3d.shape)

    return image_coords, flattened_image, pixel_indices


def local_intensity(flattened_image, pixel_indices, index):
    """Measure local mean intensity normalized to mean of all."""
    from numpy import array, nanmean
    face_intensities = array(
        [nanmean(flattened_image[pixel_indices[ii]]) for ii in index])
    return face_intensities/nanmean(face_intensities)


def register_command(session):
    from chimerax.core.commands import CmdDesc, register, SurfaceArg, ColormapArg, ColormapRangeArg, BoolArg, FloatArg, ModelArg
    desc_1 = CmdDesc(
        required=[('surface', SurfaceArg)],
        keyword=[('to_surface', SurfaceArg),
                 ('radius', FloatArg),
                 ('palette', ColormapArg),
                 ('range', ColormapRangeArg),
                 ('key', BoolArg)],
        required_arguments=['to_surface'],
        synopsis='measure local surface distance')
    register('measure distance', desc_1,
             measure_distance, logger=session.logger)

    """Register Chimerax command."""
    desc_2 = CmdDesc(
        required=[('surface', SurfaceArg)],
        keyword=[('to_map', ModelArg),
                 ('radius', FloatArg),
                 ('palette', ColormapArg),
                 ('range', ColormapRangeArg),
                 ('key', BoolArg)],
        required_arguments=['to_map'],
        synopsis='measure local surface intensity')
    register('measure intensity', desc_2,
             measure_intensity, logger=session.logger)


register_command(session)
