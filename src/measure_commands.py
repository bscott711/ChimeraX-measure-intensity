"""These commands will measure intensity and distances between surfaces in ChimeraX"""
#pylint: disable=redefined-builtin
#pylint: disable=expression-not-assigned
#pylint: disable=line-too-long
#pylint: disable=unused-argument
#pylint: disable=too-many-arguments

from chimerax.color_key import show_key
from chimerax.core import colors
from chimerax.std_commands.wait import wait
from chimerax.surface import vertex_convexity
from chimerax.core.commands import (BoolArg, Bounded, CmdDesc, ColormapArg,
                                    ColormapRangeArg, Int2Arg, IntArg,
                                    SurfacesArg, StringArg, FloatArg, SurfaceArg)
from chimerax.core.commands.cli import EnumOf
from chimerax.map.volumecommand import volume
from chimerax.std_commands.cd import (cd)
from os.path import exists
import numpy
from numpy import (arccos, array, full, inf, isnan, mean, nan, nanmax, nanmean,
                   nanmin, pi, ravel_multi_index, sign, split, sqrt, subtract,
                   count_nonzero, swapaxes, savetxt, column_stack,nansum, nanstd,
                   unique, column_stack, round_, int64, abs, digitize, linspace,
                   zeros, where, delete, shape, argmin, min)
from scipy.ndimage import (binary_dilation, binary_erosion,
                           generate_binary_structure, iterate_structure, gaussian_filter)
from scipy.spatial import KDTree
from skimage.morphology import skeletonize


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


def topology_series(session, surface, to_cell, radius= 8, target = 'sRBC',
                     size=(.1028,.1028,.1028), palette=None, color_range= 'full', key=False,
                     phi_lim= 90, output = 'None'):
    """this is ment to output a color mapped for topology metrics (phi, theta and distance from the target centroid) This is for the whole timeseries move on to the individual outputs"""
    volume(session, voxel_size= size)
    wait(session,frames=1)
    [measure_topology(session, surface, to_cell, radius, target, size, phi_lim, output)
        for surface, to_cell in zip(surface, to_cell)]
    recolor_surfaces(session, surface,'rpd', palette, color_range, key)

def ridge_series(session, surface, to_surface, to_cell, radius = 8, size = (.1028,.1028,.1028), smoothing_iterations = 20,
                 thresh = 0.3, knn=10, palette = None, color_range='full', key= False, output= 'None'):
    """This is designed to identify and track ridges that occur at phagosomes - YML"""
    volume(session, voxel_size= size)
    wait(session,frames=1)
    [measure_ridges(session, surface, to_surface, to_cell, radius, smoothing_iterations, thresh, knn,
                    size, output)
        for surface, to_surface, to_cell in zip(surface, to_surface, to_cell)]
    recolor_surfaces(session, surface,'edges', palette, color_range, key)

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


def measure_topology(session, surface, to_cell, radius=8, target='sRBC', size=[0.1028,0.1028,0.1028],
                      phi_lim= 90, output= 'None'):
    """This command is designed to output a csv file of the surface metrics:
    areal surface roughness, areal surface roughness standard deviation and surface area per frame.
    Additionally this command can color vertices based on their distance from the target centroid.
    Author: Yoseph Loyd
    Date:20230614"""
    
    """Tell the system what target you are computing the areal roughness of."""
    if target == 'sRBC':
        target_r = 2
    elif target =='mRBC':
        target_r = 2.7
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
    phi = arccos(z_coord / distance) * (180/pi)

    """Logic to identify vertices in the targets local (defined by radius input) around target's upper hemisphere"""
    abovePhi = phi <= phi_lim
    outerlim = (distance  < radius)
    radialClose = outerlim & (distance > target_r)


    """Logic statments for solving the unique X,Y coordinates in the upper hemisphere search"""
    XYZ_SearchLim = distance*abovePhi
    SearchR = distance*abovePhi*outerlim
    '''XY_SearchR = distance*abovePhi
    XY_deletes = where(XY_SearchR==0)
    XYZ_deletes = where(XYZ_SearchR==0)



    """Solving for unique X,Y,Z coordinates in the upper hemisphere search"""
    xx=x_coord*Points
    yy=y_coord*Points
    zz=z_coord*Points

    xy_raw = column_stack((xx,yy))
    xyz_raw = column_stack((xx,yy,zz))

    xy=unique(delete(xy_raw,XY_deletes,axis=0),axis=0)
    xyz=unique(delete(xyz_raw,XYZ_deletes,axis=0),axis=0)

    """Defining the pixel size from human defined parameter"""
    width = size[0]

    """Defining steps that will are approximately one pixel in length"""
    steps = int64(round_(abs((2*radius)/(width))))

    """Indexing the vertices that fall in one pixel of eachother along each axis""" 
    """Weird nearest neighbors approach"""
    xbins_xy = digitize(xyz[:,0],linspace(-8,8,steps))
    ybins_xy = digitize(xyz[:,1],linspace(-8,8,steps))

    xbins = digitize(xyz[:,0],linspace(-8,8,steps))
    ybins = digitize(xyz[:,1],linspace(-8,8,steps))
    zbins = digitize(xyz[:,2],linspace(-8,8,steps))

    """Making an artificial binary mask of binned vertices into 'pixels' from vertice location"""
    ArtImgxy= zeros([steps,steps])
    ArtImgxy[xbins_xy,ybins_xy]= 1

    ArtImgxyz= zeros([steps,steps,steps])
    ArtImgxyz[xbins,ybins,zbins]= 1

    """Filling holes and cutts in image"""
    ArtImg_Filled= binary_erosion(((gaussian_filter(ArtImgxy,.5))>0),border_value=1,iterations=2)

    ArtImg_Filledxyz = binary_erosion(((gaussian_filter(ArtImgxyz,.2))>0),border_value=1,iterations=1)'''
    ArtImg = ImgReconstruct(SearchR,x_coord,y_coord,z_coord,XYZ_SearchLim)
    """Area of pixels in X,Y plane of the hemispher search"""
    Area_S= count_nonzero(ArtImg) * (size[1] * size[0])

    """Outputs for coloring vertices as surface. arguments"""
    radialDistanceAbovePhiLimitxy = abovePhi * radialClose * distance
    surface.radialDistanceAbovePhiNoNans= abovePhi * radialClose * distance 
    radialDistanceAbovePhiLimitxy[radialDistanceAbovePhiLimitxy == 0] = nan

    surface.radialDistanceAbovePhi= abovePhi* distance
    surface.radialDistanceAbovePhiLimitxy=radialDistanceAbovePhiLimitxy

    surface.radialDistance = distance
    surface.theta = theta
    surface.phi = phi
    surface.areasearch = SearchR

    """Single value outputs for definning topology"""
    surface.IRDFCarray = nanmean(radialDistanceAbovePhiLimitxy)
    surface.Sum = nansum(radialDistanceAbovePhiLimitxy)

    surface.area = count_nonzero(ArtImg) * size[0]*size[1]
    surface.ArealRoughness = sqrt(surface.IRDFCarray**2/(2*pi*target_r**2))
    surface.ArealRoughness_STD = nanstd(surface.radialDistanceAbovePhiLimitxy)/(2*pi*target_r**2)
    surface.ArealRoughnessperArea= surface.ArealRoughness / Area_S

    """Text file output"""
    path = exists(output)
    if path == True:
        cd(session,str(output))
        with open('Areal Surface Roughness.csv', 'ab') as f:
            savetxt(f, column_stack([surface.ArealRoughness, surface.ArealRoughness_STD, surface.area, surface.ArealRoughnessperArea]),
                     header=f"Areal-Surface-Roughness-S_q STD_Areal-Rougheness Surface_Area ArealRoughness/um^2", comments='')
    else:
        return surface.radialDistanceAbovePhiNoNans

def measure_ridges(session, surface, to_surface, to_cell,  radius = 8, smoothing_iterations = 20,
                    thresh = 0.3, knn=10, size=[0.1028,0.1028,0.1028], output= 'None'):
    
    """Define the target centroid from mid range x,y and z coordinates."""
    centroid = mean(to_cell.vertices, axis=0)

    """Vertice x,y and z coordinates from centroid"""
    x_coord, y_coord, z_coord = split(subtract(surface.vertices, centroid), 3, 1)
    x_coord_t, y_coord_t, z_coord_t = split(subtract(to_surface.vertices, centroid), 3, 1)

    """Defining coordinates for surface t"""
    x_coord = x_coord.flatten()
    y_coord = y_coord.flatten()
    z_coord = z_coord.flatten()
    """Converting the cartisian coordinates into spherical coordinates for surface t"""
    z_squared = z_coord ** 2
    y_squared = y_coord ** 2
    x_squared = x_coord ** 2
    
    distance = sqrt(z_squared + y_squared + x_squared)
    distxy = sqrt(x_squared + y_squared)
    theta = sign(y_coord)*arccos(x_coord / distxy)
    phi = arccos(z_coord / distance) * (180/pi)

    """Defining coordinates for surface t+1"""
    x_coord_t = x_coord_t.flatten()
    y_coord_t = y_coord_t.flatten()
    z_coord_t = z_coord_t.flatten()
    """Converting the cartisian coordinates into spherical coordinates for surface t+1"""
    z_squared_t = z_coord_t ** 2
    y_squared_t = y_coord_t ** 2
    x_squared_t = x_coord_t ** 2

    distance_t = sqrt(z_squared_t + y_squared_t + x_squared_t)
    
    """Paletting options for R, phi, and theta for surface t"""
    surface.radialDistance = distance
    surface.theta = theta
    surface.phi = phi
    
    """Defining search restrictions"""
    sphere = distance  < radius
    sphere_t = distance_t  < radius
    try: Clip= z_coord > (min( z_coord[ where(phi<165) ])+0.5)
    except ValueError:  #raised if `Clip` is empty.
        pass

    """Defining surfaces t and t+1 convexity without paletting"""
    con = vertex_convexity(surface.vertices, surface.triangles, smoothing_iterations)
    con_t = vertex_convexity(to_surface.vertices, to_surface.triangles, smoothing_iterations)

    ind = (con > thresh)
    ind_t = (con_t > thresh)

    car = zeros(shape(surface.vertices))
    car_t = zeros(shape(to_surface.vertices))

    car[:,0]= surface.vertices[:,0]*ind*sphere
    car[:,1]= surface.vertices[:,1]*ind*sphere
    car[:,2]= surface.vertices[:,2]*ind*sphere

    car_t[:,0]= to_surface.vertices[:,0]*ind_t*sphere_t
    car_t[:,1]= to_surface.vertices[:,1]*ind_t*sphere_t
    car_t[:,2]= to_surface.vertices[:,2]*ind_t*sphere_t

    query_distance,_=query_tree(car, car_t, knn=knn)

    """The Average distance of the nearest neighbors"""
    surface.q_dist=query_distance

    """Solving for the surface area of high curved regions and the high curve path length"""

    """High curve edges"""
    surface.edges = ind * sphere *Clip

    """search limitations """
    SearchLim = ind * sphere * Clip

    """Reconstructed image"""
    ArtImg = ImgReconstruct(surface.edges, x_coord, y_coord, z_coord,SearchLim, radius, size)

    """Surface Area of high curved regions"""
    surface.Area = count_nonzero(ArtImg) * size[0]*size[1]

    """Skeletonizing the reconstructed image"""
    RidgePathLength = skeletonize((ArtImg*1),method='lee')
    surface.pathlength = RidgePathLength

    """Text file output"""
    path = exists(output)
    if path == True:
        cd(session,str(output))
        with open('RidgeInfo.csv', 'ab') as f:
            savetxt(f, column_stack([surface.Area, surface.pathlength]),
                     header=f"High_Curve_Surface_Area Lamella_pathlength", comments='')
    else:
        return surface.pathlength

def ImgReconstruct(Points, x_coord, y_coord, z_coord, SearchLim, radius, size):
    """This script will reconstruct an image form the location of vertices in your rendered surface"""

    """Logic statments for specific objects we care about"""
    XYZ_SearchR = SearchLim
    XYZ_deletes = where(XYZ_SearchR==0)

    """Solving for unique X,Y,Z coordinates in the search"""
    xx=x_coord[:]*Points
    yy=y_coord[:]*Points
    zz=z_coord[:]*Points

    xyz_raw = column_stack((xx,yy,zz))

    xyz=unique(delete(xyz_raw,XYZ_deletes,axis=0),axis=0)

    """Defining the pixel size from human defined parameter"""
    width = size[0]

    """Defining steps that will are approximately one pixel in length"""
    steps = int64(round_(abs((2*radius)/(width))))

    """Indexing the vertices that fall in one pixel of eachother along each axis""" 
    """Weird nearest neighbors approach"""    
    xbins = digitize(xyz[:,0],linspace(-1*(radius),radius,steps))
    ybins = digitize(xyz[:,1],linspace(-1*(radius),radius,steps))
    zbins = digitize(xyz[:,2],linspace(-1*(radius),radius,steps))

    """Making an artificial binary mask of binned vertices into 'pixels' from vertice location"""

    ArtImgxyz= zeros([steps,steps,steps])
    ArtImgxyz[xbins,ybins,zbins]= 1

    """Filling holes and cutts in image or ArtImg_Filledxyz"""
    ArtImg = binary_erosion(((gaussian_filter(ArtImgxyz,.2))>0),border_value=1,iterations=1)
    ArtImg = ArtImg.astype('int8')
    return ArtImg

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
    elif metric == 'area' and hasattr(surface, 'areasearch'):
        measurement = surface.areasearch
        palette_string = 'purples'
        max_range = 1
    elif metric == 'edges' and hasattr(surface, 'edges'):
        measurement = surface.edges * 1
        palette_string = 'purples'
        max_range = 10
    elif metric == 'qd' and hasattr(surface, 'q_dist'):
        measurement = surface.q_dist * 1
        palette_string = 'brbg'
        max_range = 10
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
    keyword=[('metric', EnumOf(['intensity', 'distance','R', 'theta', 'phi', 'Rphi', 'rpg','rpd', 'area', 'edges', 'qd'])),
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
             ('metric', EnumOf(['R', 'theta', 'phi', 'Rphi', 'rpg','rpd', 'area'])),
             ('palette', ColormapArg),
             ('radius', Bounded(IntArg)),
             ('target', EnumOf(['sRBC', 'mRBC'])),
             ('color_range', ColormapRangeArg),
             ('phi_lim', Bounded(IntArg)),
             ('output', StringArg),
             ('key', BoolArg)],
    required_arguments=['to_cell'],
    synopsis='This measure function will output calculated axial surface roughness values (S_q)'
        'based on inputs surface-Macrophage, tocell- target, radius- search radius (um), targetr- target radius (um)')

measure_ridges_desc = CmdDesc(
    required=[('surface',SurfaceArg)],
    keyword=[('to_surface', SurfaceArg),
             ('to_cell', SurfaceArg),
             ('smoothing_iterations', Bounded(IntArg)),
             ('thresh', Bounded(FloatArg)),
             ('palette', ColormapArg),
             ('radius', Bounded(FloatArg)),
             ('knn', Bounded(IntArg)),
             ('color_range', ColormapRangeArg),
             ('key', BoolArg)],
    required_arguments = ['to_surface','to_cell'],
    synopsis = 'This function is in its first iteration. Current implimentation focuses on identifying high curvature'
        'lamella edges for video recodings')