"""These commands will measure intensity and distances between surfaces in ChimeraX"""
#pylint: disable=redefined-builtin
#pylint: disable=expression-not-assigned
#pylint: disable=line-too-long
#pylint: disable=unused-argument
#pylint: disable=too-many-arguments

from chimerax.color_key import show_key
from chimerax.core import colors
from chimerax.std_commands.wait import wait
from chimerax.core.commands import (BoolArg, Bounded, CmdDesc, ColormapArg,
                                    ColormapRangeArg, Int2Arg, IntArg,
                                    SurfacesArg, StringArg)
from chimerax.core.commands.cli import EnumOf
"""from chimerax.surface import (surface_area)"""
from chimerax.map.volumecommand import volume
from chimerax.std_commands.cd import (cd)
from os.path import exists
from numpy import (arccos, array, full, inf, isnan, mean, nan, nanmax, nanmean,
                   nanmin, pi, ravel_multi_index, sign, split, sqrt, subtract,
                   count_nonzero, swapaxes, savetxt, column_stack,nansum, isin,min,
                   argwhere, zeros, shape, nanstd, int_)
from scipy.ndimage import (binary_dilation, binary_erosion,
                           generate_binary_structure, iterate_structure)
from scipy.spatial import KDTree


def distance_series(session, surface, to_surface, knn=5, palette=None, color_range=None, key=False):
    """Wrap the distance measurement for list of surfaces."""
    [measure_distance(surface, to_surface, knn)
     for surface, to_surface in zip(surface, to_surface)]
    recolor_surfaces(session, surface, 'distance', palette, color_range, key)


def intensity_series(session, surface, to_map, radius=15, palette=None, color_range=None, key=False):
    """Wrap the intensity measurement for list of surfaces."""
    [measure_intensity(surface, to_map, radius)
     for surface, to_map in zip(surface, to_map)]
    recolor_surfaces(session, surface, 'intensity', palette, color_range, key)


def composite_series(session, surface, green_map, magenta_map, radius=15, palette='green_magenta', green_range=None, magenta_range=None):
    """Wrap the composite measurement for list of surfaces."""
    [measure_composite(surface, green_map, magenta_map, radius)
        for surface, green_map, magenta_map in zip(surface, green_map, magenta_map)]
    recolor_composites(session, surface, palette, green_range, magenta_range)


def topology_series(session, surface, to_cell, radius= 8, metric='RPD', target = 'sRBC', size=(.1028,.1028,.1028), palette=None, color_range= None, key=False, output = 'None'):
    """this is ment to output a color mapped for topology metrics (phi, theta and distance from the target centroid) This is for the whole timeseries move on to the individual outputs"""
    volume(session, voxel_size= size)
    wait(session,frames=1)
    [measure_topology(session, surface, to_cell, radius, target, size, output)
        for surface, to_cell in zip(surface, to_cell)]
    recolor_surfaces(session, surface, palette, color_range, key, metric)


def recolor_surfaces(session, surface, metric='intensity', palette=None, color_range=None, key=False):
    """Wraps recolor_surface in a list comprehension"""
    keys = full(len(surface), False)
    keys[0] = key
    [recolor_surface(session, surface, metric, palette, color_range, key)
     for surface, key in zip(surface, keys)]


def recolor_composites(session, surface, palette='green_magenta', green_range=None, magenta_range=None, palette_range=(40, 240)):
    """Wraps composite_color in a list comprehension"""
    [composite_color(session, surface, palette, green_range, magenta_range, palette_range)
     for surface in surface]


def measure_distance(surface, to_surface, knn):
    """Measure the local motion of two surfaces."""
    distance, _ = query_tree(surface.vertices, to_surface.vertices, knn=knn)
    surface.distance = distance


def measure_topology(session, surface, to_cell, radius=8, target='sRBC', size=[0.1028,0.1028,0.1028], output= 'None'):
    """This command is designed to output a csv file of the surface metrics:
    areal surface roughness, areal surface roughness standard deviation and surface area per frame.
    Additionally this command can color vertices based on their distance from the target centroid.
    Author: Yoseph Loyd
    Date:20230413"""
    
    """Tell the system what target you are computing the areal roughness of."""
    if target == 'sRBC':
        target_r = 2.25
    elif target =='mRBC':
        target_r = 3
    else:
        return
    #Target not recognized

    """Define the target centroid from mid range x,y and z coordinates."""
    centroid = mean(to_cell.vertices, axis=0)

    """Vertice x,y and z distances from centroid"""
    x_coord, y_coord, z_coord = split(subtract(surface.vertices, centroid), 3, 1)

    x_coord = x_coord.flatten()
    y_coord = y_coord.flatten()
    z_coord = z_coord.flatten()
    """Converting the cartisian system into spherical coordinates"""
    z_squared = z_coord ** 2
    y_squared = y_coord ** 2
    x_squared = x_coord ** 2
    
    distance = sqrt(z_squared + y_squared + x_squared)
    distxy = sqrt(x_squared + y_squared)
    theta = sign(y_coord)*arccos(x_coord / distxy)
    phi = arccos(z_coord / distance)

    """Logic to identify vertices in the targets local (defined by radius input) around target's upper hemisphere"""
    abovePhi = phi <= (pi/2)
    radialClose = (distance  < radius) & (distance > target_r)

    """Outputs for coloring vertices as surface. arguments"""
    radialDistanceAbovePhiLimitxy = abovePhi * radialClose * distance
    surface.radialDistanceAbovePhiNoNans= abovePhi * radialClose * distance 
    radialDistanceAbovePhiLimitxy[radialDistanceAbovePhiLimitxy == 0] = nan

    surface.radialDistanceAbovePhi= abovePhi* distance
    surface.radialDistanceAbovePhiLimitxy=radialDistanceAbovePhiLimitxy

    surface.radialDistance = distance
    surface.theta = theta
    surface.phi = phi

    """ reconstructin matrix of bool vetices
    limits = abovePhi * radialClose
    v_x = x_coord*limits
    v_y = y_coord*limits
    v_z = z_coord*limits

    vertices=zeros(shape(surface.vertices))

    vertices[:,0]=v_x
    vertices[:,1]=v_y
    vertices[:,2]=v_z

    '''Identifying triangles by vertice index using numpy module'''
    vertice_index = argwhere(limits)

    '''Retaining all triangles of interest'''
    Bool_triangles = isin(surface.triangles, vertice_index)
    Bool_triangles = Bool_triangles.astype('int32')
    Bool_triangles[Bool_triangles == 0]= nan
    ------
    BoolT[BoolT ==0] = nan
    '''Converting the boolean triangles logic to modify triangles array'''
    nan_triangles[nan_triangles == 0 ] = nan
    dataType = (surface.triangles).dtype
    nan_value = min(nan_triangles.astype(str(dataType)))
    tirangles = delete() """

    """Logic to identify vertices in the targets local (defined by radius input) around target's upper hemisphere"""
    abovePhi = phi <= (pi/2)
    radialClose = (distance  < radius) & (distance > target_r)

    """Single value outputs for definning topology"""
    surface.IRDFCarray = nanmean(radialDistanceAbovePhiLimitxy)
    surface.Sum = nansum(radialDistanceAbovePhiLimitxy)
    """ surface.area = surface_area(vertices, triangles) """

    surface.area = count_nonzero(surface.radialDistanceAbovePhiNoNans)
    surface.ArealRoughness = sqrt(surface.IRDFCarray**2/(2*pi*target_r**2))
    surface.ArealRoughness_STD = nanstd(surface.radialDistanceAbovePhiLimitxy)/(2*pi*target_r**2)
    
    """"Text file output"""
    path = exists(output)
    if path == True:
        cd(session,str(output))
        with open('Areal Surface Roughness.csv', 'ab') as f:
            savetxt(f, column_stack([surface.ArealRoughness, surface.ArealRoughness_STD, surface.area]), header=f"Areal-Surface-Roughness S_q STD_Areal-Rougheness #_Vertices", comments='')
    else:
        return
    

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
    struct_el = iterate_structure(generate_binary_structure(3, 1), 2)
    mask_d = binary_dilation(mask, structure=struct_el)
    mask_e = binary_erosion(mask, structure=struct_el, iterations=2)
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


def recolor_surface(session, surface, metric, palette, color_range, key):
    """Colors surface based on previously measured intensity or distance"""
    if metric == 'distance' and hasattr(surface, 'distance'):
        measurement = surface.distance
        palette_string = 'brbg'
        max_range = 15
    elif metric == 'intensity' and hasattr(surface, 'intensity'):
        measurement = surface.intensity
        palette_string = 'purples'
        max_range = 5
    elif metric == 'R' and hasattr(surface, 'radialDistance'):
        measurement = surface.radialDistance
        palette_string = 'purples'
        max_range = 100
    elif metric == 'Rphi' and hasattr(surface, 'radialDistanceAbovePhi'):
        measurement = surface.radialDistanceAbovePhi
        palette_string = 'purples'
        max_range = 10
    elif metric == 'rpg' and hasattr(surface, 'radialDistanceAbovePhiLimitxy'):
        measurement = surface.radialDistanceAbovePhiLimitxy
        palette_string = 'purples'
        max_range = 10
    elif metric == 'rpd' and hasattr(surface, 'radialDistanceAbovePhiNoNans'):
        measurement = surface.radialDistanceAbovePhiNoNans
        palette_string = 'purples'
        max_range = 10
    elif metric == 'theta' and hasattr(surface, 'theta'):
        measurement = surface.theta
        palette_string = 'brbg'
        if color_range is None:
            color_range = -pi,pi
    elif metric == 'phi' and hasattr(surface, 'phi'):
        measurement = surface.phi
        palette_string = 'brbg'
        max_range = pi
    else:
        return

    # If all the measurements are np.nan set them to zero.
    if isnan(measurement).all():
        measurement[:] = 0

    if palette is None:
        palette = colors.BuiltinColormaps[palette_string]

    if color_range is not None and color_range != 'full':
        rmin, rmax = color_range
    elif color_range == 'full':
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
    if isnan(green_channel).all():
        green_channel[:] = 0
    if isnan(magenta_channel).all():
        magenta_channel[:] = 0

    green_range = scale_range(green_range, green_channel)
    magenta_range = scale_range(magenta_range, magenta_channel)

    # Define the color palettes.
    # gvals = ['#003c00', '#00b400']
    # mvals = ['#3c003c', '#b400b4']
    gmap = make_palette(green_channel, green_range, palette_range)
    mmap = make_palette(magenta_channel, magenta_range, palette_range,'magenta')

    # Build the composite vertex colors
    composite_map = array(gmap)
    composite_map[:, 1] = gmap[:, 1]
    composite_map[:, 0] = mmap[:, 0]
    composite_map[:, 2] = mmap[:, 2]

    surface.vertex_colors = composite_map

def make_palette(channel_data, color_range, palette_range, palette='green'):
    """Helper function to make the new color palette"""
    low, high = palette_range
    if palette == 'green':
        vals = [f'#00{low:02x}00', f'#00{high:02x}00']
    else:
        vals = [f'#{low:02x}00{low:02x}', f'#{high:02x}00{high:02x}']

    vals = [colors.Color(v) for v in vals]
    color = colors.Colormap(None, vals)
    palette = color.rescale_range(color_range[0],color_range[1])
    colormap = palette.interpolated_rgba8(channel_data)

    return colormap

def scale_range(color_range=(0,30), channel=None):
    """Helper function to set the color range"""
    if color_range == 'full':
        color_range = nanmin(channel), nanmax(channel)

    return color_range


measure_distance_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('to_surface', SurfacesArg),
             ('knn', Bounded(IntArg)),
             ('palette', ColormapArg),
             ('color_range', ColormapRangeArg),
             ('key', BoolArg)],
    required_arguments=['to_surface'],
    synopsis='Measure local distance between two surfaces')


measure_intensity_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('to_map', SurfacesArg),
             ('radius', Bounded(IntArg, 1, 30)),
             ('palette', ColormapArg),
             ('color_range', ColormapRangeArg),
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
    keyword=[('metric', EnumOf(['intensity', 'distance','R', 'theta', 'phi', 'Rphi', 'rpg','rpd'])),
             ('palette', ColormapArg),
             ('color_range', ColormapRangeArg),
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

measure_topology_desc = CmdDesc(
    required=[('surface', SurfacesArg)],
    keyword=[('to_cell', SurfacesArg),
             ('metric', EnumOf(['R', 'theta', 'phi', 'Rphi', 'rpg','rpd'])),
             ('palette', ColormapArg),
             ('radius', Bounded(IntArg)),
             ('target', EnumOf(['sRBC', 'mRBC'])),
             ('color_range', ColormapRangeArg),
             ('output', StringArg),
             ('key', BoolArg)],
    required_arguments=['to_cell'],
    synopsis='This measure function will output calculated axial surface roughness values (S_q)'
        'based on inputs surface-Macrophage, tocell- target, radius- search radius (um), targetr- target radius (um)')