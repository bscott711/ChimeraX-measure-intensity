from chimerax.color_key import show_key
from chimerax.core import colors
from chimerax.core.commands import (BoolArg, Bounded, CmdDesc, ColormapArg,
                                    ColormapRangeArg, IntArg, Int2Arg, SurfacesArg)
from chimerax.core.commands.cli import EnumOf
from numpy import (all, array, full, inf, nanmax, nanmean, nanmin,
                   ravel_multi_index, swapaxes)
from scipy.ndimage import binary_dilation
from scipy.ndimage import (binary_erosion,
                                      generate_binary_structure,
                                      iterate_structure)
from scipy.spatial import KDTree


def distance_series(session, surface, to_surface, knn=5, palette=None, range=None, key=False):
    """Wrap the distance measurement for list of surfaces."""
    [measure_distance(surface, to_surface, knn)
     for surface, to_surface in zip(surface, to_surface)]
    recolor_surfaces(session, surface, 'distance', palette, range, key)


def intensity_series(session, surface, to_map, radius=15, palette=None, range=None, key=False):
    """Wrap the intensity measurement for list of surfaces."""
    [measure_intensity(surface, to_map, radius)
     for surface, to_map in zip(surface, to_map)]
    recolor_surfaces(session, surface, 'intensity', palette, range, key)


def composite_series(session, surface, green_map, magenta_map, radius=15, palette='green_magenta', green_range=None, magenta_range=None):
    """Wrap the composite measurement for list of surfaces."""
    [measure_composite(surface, green_map, magenta_map, radius)
        for surface, green_map, magenta_map in zip(surface, green_map, magenta_map)]
    recolor_composites(session, surface, palette, green_range, magenta_range)


def recolor_surfaces(session, surface, metric='intensity', palette=None, range=None, key=False):
    """Wraps recolor_surface in a list comprehension"""
    keys = full(len(surface), False)
    keys[0] = key
    [recolor_surface(session, surface, metric, palette, range, key)
     for surface, key in zip(surface, keys)]


def recolor_composites(session, surface, palette='green_magenta', green_range=None, magenta_range=None, palette_range=(40, 240)):
    """Wraps composite_color in a list comprehension"""
    [composite_color(session, surface, palette, green_range, magenta_range, palette_range)
     for surface in surface]


def measure_distance(surface, to_surface, knn):
    """Measure the local motion of two surfaces."""
    distance, _ = query_tree(surface.vertices, to_surface.vertices, knn=knn)
    surface.distance = distance


def measure_intensity(surface, to_map, radius):
    """Measure the local intensity within radius r of the surface."""
    image_info = get_image(surface, to_map)
    masked_image = mask_image(*image_info)
    image_coords, *flattened_indices = get_coords(masked_image)
    _, index = query_tree(surface.vertices, image_coords.T, radius)
    face_intensity = local_intensity(*flattened_indices, index)
    surface.intensity = face_intensity


def measure_composite(surface, green_map, magenta_map, radius):
    """Measure the local intensity for 2 channels within radius r of the surface."""
    green_coords, *green_indices = get_image_coords(surface, green_map)
    _, green_index = query_tree(surface.vertices, green_coords.T, radius)
    green_intensity = local_intensity(*green_indices, green_index)
    magenta_coords, *magenta_indices = get_image_coords(surface, magenta_map)
    _, magenta_index = query_tree(surface.vertices, magenta_coords.T, radius)
    magenta_intensity = local_intensity(*magenta_indices, magenta_index)
    surface.ch1 = green_intensity
    surface.ch2 = magenta_intensity


def get_image(surface, to_map):
    """Get the isosurface volume mask and secondary channel."""
    mask_vol = surface.volume.full_matrix().copy()
    image_3d = to_map.volume.full_matrix().copy()
    level = surface.volume.maximum_surface_level
    return mask_vol, level, image_3d


def get_coords(image_3d):
    """Get the coords for local intensity"""
    # ChimeraX uses XYZ for image, but numpy uses ZYX, swap dims
    image_3d = swapaxes(image_3d, 0, 2)
    image_coords = array(image_3d.nonzero())
    flattened_image = image_3d.flatten()
    pixel_indices = ravel_multi_index(image_coords, image_3d.shape)
    return image_coords, flattened_image, pixel_indices


def mask_image(mask, level, image_3d):
    """Mask the secondary channel based on the isosurface. Uses a 3D ball to dilate and erode with radius 2, then xor to make membrane mask."""
    mask = mask >= level
    se = iterate_structure(generate_binary_structure(3, 1), 2)
    mask_d = binary_dilation(mask, structure=se)
    mask_e = binary_erosion(mask, structure=se, iterations=2)
    masked = mask_d ^ mask_e
    image_3d *= masked
    return image_3d


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


def get_image_coords(surface, image):
    """Get the image coordinates for use in KDTree."""
    image_info = get_image(surface, image)
    masked_image = mask_image(*image_info)
    image_coords, *flattened_indices = get_coords(masked_image)
    return image_coords, *flattened_indices


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

    # If all the measurements are np.nan set them to zero.
    if all(measurement != measurement):
        measurement[:] = 0

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


def composite_color(session, surface, palette, green_range, magenta_range, palette_range):
    """Colors surface based on previously measured intensity or distance"""
    if hasattr(surface, 'ch1') and hasattr(surface, 'ch2'):
        if palette == 'magenta_green':
            green_channel = surface.ch2
            magenta_channel = surface.ch1
        else:
            green_channel = surface.ch1
            magenta_channel = surface.ch2
    else:
        return

    # If all the measurements are np.nan set them to zero.
    if all(green_channel != green_channel):
        green_channel[:] = 0
    if all(magenta_channel != magenta_channel):
        magenta_channel[:] = 0

    if green_range is not None and green_range != 'full':
        green_min, green_max = green_range
    elif green_range == 'full':
        green_min, green_max = nanmin(green_channel), nanmax(green_channel)
    else:
        green_min, green_max = (0, 30)

    if magenta_range is not None and magenta_range != 'full':
        magenta_min, magenta_max = magenta_range
    elif magenta_range == 'full':
        magenta_min, magenta_max = nanmin(
            magenta_channel), nanmax(magenta_channel)
    else:
        magenta_min, magenta_max = (0, 30)

    # Define the color palettes.
    # gvals = ['#003c00', '#00b400']
    low = palette_range[0]
    high = palette_range[1]
    gvals = ["#00{:02x}00".format(low), "#00{:02x}00".format(high)]
    gvals = [colors.Color(v) for v in gvals]
    green = colors.Colormap(None, gvals)
    green_palette = green.rescale_range(green_min, green_max)
    gmap = green_palette.interpolated_rgba8(green_channel)

    # mvals = ['#3c003c', '#b400b4']
    mvals = ["#{:02x}00{:02x}".format(low, low),
             "#{:02x}00{:02x}".format(high, high)]
    mvals = [colors.Color(v) for v in mvals]
    magenta = colors.Colormap(None, mvals)
    magenta_palette = magenta.rescale_range(magenta_min, magenta_max)
    mmap = magenta_palette.interpolated_rgba8(magenta_channel)

    # Build the composite vertex colors
    composite_map = array(gmap)
    composite_map[:, 1] = gmap[:, 1]
    composite_map[:, 0] = mmap[:, 0]
    composite_map[:, 2] = mmap[:, 2]

    surface.vertex_colors = composite_map


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


measure_composite_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('green_map', SurfacesArg),
             ('magenta_map', SurfacesArg),
             ('radius', Bounded(IntArg, 1, 30)),
             ('green_range', ColormapRangeArg),
             ('magenta_range', ColormapRangeArg)],
    required_arguments=['green_map', 'magenta_map'],
    synopsis='Measure local intensities of two channels relative to surface')


recolor_surfaces_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('metric', EnumOf(['intensity', 'distance'])),
             ('palette', ColormapArg),
             ('range', ColormapRangeArg),
             ('key', BoolArg)],
    required_arguments=[],
    synopsis='Recolor surface based on previous measurement')


recolor_composites_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('palette', EnumOf(['green_magenta', 'magenta_green'])),
             ('green_range', ColormapRangeArg),
             ('magenta_range', ColormapRangeArg),
             ('palette_range', Int2Arg)],
    required_arguments=[],
    synopsis='Recolor surface based on previous measurements as a composite')
