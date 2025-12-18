#######################################################
# Everything to do with reading the grid
# You can build this using NetCDF output files 
#######################################################

import numpy as np
import os
import xarray as xr
from .interpolation import neighbours
from .constants import region_edges, region_edges_flag, region_names, region_points, shelf_lat, shelf_depth, shelf_point0, region_bounds, region_bathy_bounds
from .utils import remove_disconnected, closest_point, latlon_name, xy_name

# Helper function to get a 3D land mask from a NEMO output file, using either thetao or so.
def build_mask_3d (ds):

    mask_3d = None
    for var in ['thetao', 'so']:
        if var in ds:
            mask_3d = xr.where(ds[var].isel(time_counter=0)==0, 0, 1).squeeze()
            break
    if mask_3d is None:
        raise Exception('No known 3D masked variable is present. Add another variable to the code?')
    return mask_3d
    

# Helper function to calculate a bunch of grid variables (bathymetry, draft, ocean mask, ice shelf mask) from a NEMO output file, only using thkcello/e3t and the mask on a 3D data variable (current options are to look for thetao and so).
# This varies a little if the sea surface height changes, so not perfect, but it does take partial cells into account.
# If keep_time_dim, will preserve any time dimension even if it's of size 1 (useful for timeseries)
def calc_geometry (ds, keep_time_dim=False):

    mask_3d = build_mask_3d(ds)
    # 2D ocean cells are unmasked at some depth
    ocean_mask = mask_3d.sum(dim='deptht')>0
    # 2D ice shelf cells are ocean cells which are masked at the surface
    ice_mask = ocean_mask*(mask_3d.isel(deptht=0)==0)
    # Water column thickness is sum of dz in unmasked cells
    if 'thkcello' in ds:
        dz = ds['thkcello']
    elif 'e3t' in ds:
        dz = ds['e3t']
    else:
        raise Exception('Missing thkcello and e3t')
    wct = (dz*mask_3d).sum(dim='deptht')
    # Now identify the 3D ice shelf cells using cumulative sum of mask
    ice_mask_3d = (mask_3d.cumsum(dim='deptht')==0)*ocean_mask
    # Ice draft is sum of dz in ice shelf cells
    draft = (dz*ice_mask_3d).sum(dim='deptht')
    # Bathymetry is ice draft plus water column thickness
    bathy = draft + wct
    if not keep_time_dim:
        bathy = bathy.squeeze()
        draft = draft.squeeze()
    return bathy, draft, ocean_mask, ice_mask

# Select ice shelf cavities. Pass it the path to an xarray Dataset which contains either 'maskisf' (NEMO3.6 mesh_mask), 'top_level' (NEMO4.2 domain_cfg), or 'thkcello'/'e3t' plus a 3D data variable with a zero-mask applied (NEMO output file - see calc_geometry)
def build_ice_mask (ds):

    if 'ice_mask' in ds:
        # Previously computed
        return ds['ice_mask'], ds
    if 'maskisf' in ds:
        ice_mask = ds['maskisf'].squeeze()
    elif 'top_level' in ds:
        ice_mask = xr.where(ds['top_level']>1, 1, 0).squeeze()
    else:
        ice_mask = calc_geometry(ds)[3]
    # Save to the Dataset in case it's useful later
    ds = ds.assign({'ice_mask':ice_mask})
    return ice_mask, ds


# As above, select the ocean mask
def build_ocean_mask (ds):

    if 'ocean_mask' in ds:
        return ds['ocean_mask'], ds
    if 'tmaskutil' in ds:
        ocean_mask = ds['tmaskutil'].squeeze()
    elif 'bottom_level' in ds:
        ocean_mask = xr.where(ds['bottom_level']>0, 1, 0).squeeze()
    else:
        ocean_mask = calc_geometry(ds)[2]
    ds = ds.assign({'ocean_mask':ocean_mask})
    return ocean_mask, ds

# Select the continental shelf and ice shelf cavities. Pass it the path to an xarray Dataset which contains one of the following combinations:
# 1. nav_lon, nav_lat, bathy, tmaskutil (NEMO3.6 mesh_mask)
# 2. nav_lon, nav_lat, bathy_metry, bottom_level (NEMO4.2 domain_cfg)
# 3. nav_lon, nav_lat, thkcello/e3t, a 3D data variable with a zero-mask applied (current options are thetao or so) (NEMO output file)
# 4. lon, lat, bathymetry (Shenjie's climatology)
def build_shelf_mask (ds):

    if 'shelf_mask' in ds:
        # Previously computed
        return ds['shelf_mask'], ds

    if 'bathy' in ds and 'tmaskutil' in ds:
        bathy = ds['bathy'].squeeze()
        ocean_mask = ds['tmaskutil'].squeeze()
    elif 'bathy_metry' in ds and 'bottom_level' in ds:
        bathy = ds['bathy_metry'].squeeze()
        ocean_mask = xr.where(ds['bottom_level']>0, 1, 0).squeeze()
    elif 'thkcello' in ds or 'e3t' in ds:
        bathy, draft, ocean_mask, ice_mask = calc_geometry(ds)
        # Make sure ice shelves are included in the final mask, by setting bathy to 0 here
        bathy = xr.where(ice_mask, 0, bathy)
    elif 'bathymetry' in ds:
        bathy = ds['bathymetry']
        ocean_mask = xr.where(bathy > 0, 1, 0)
    else:
        raise KeyError('invalid Dataset for build_shelf_mask')
    # Apply lat-lon bounds and bathymetry bound to ocean mask
    lon_name, lat_name = latlon_name(ds)
    mask = ocean_mask*(ds[lat_name] <= shelf_lat)*(bathy <= shelf_depth)
    # Remove disconnected seamounts
    mask_orig = mask.copy()
    point0 = closest_point(ds, shelf_point0)
    mask.data = remove_disconnected(mask, point0)
    # Save to the Dataset in case it's useful later
    ds = ds.assign({'shelf_mask':mask})

    return mask, ds


# Select a mask for a single cavity. Pass it an xarray Dataset as for build_shelf_mask.
def single_cavity_mask (cavity, ds, return_name=False):

    if return_name:
        title = region_names[cavity]

    if cavity+'_single_cavity_mask' in ds:
        # Previously computed
        if return_name:
            return ds[cavity+'_single_cavity_mask'], ds, title
        else:
            return ds[cavity+'_single_cavity_mask'], ds

    ds = ds.load()

    # Get mask for all cavities
    ice_mask, ds = build_ice_mask(ds)
    ice_mask = ice_mask.copy()

    # Select one point in this cavity
    point0 = closest_point(ds, region_points[cavity])
    # Disconnect the other cavities
    mask = remove_disconnected(ice_mask, point0)
    ice_mask.data = mask

    # Save to the Dataset in case it's useful later
    ds = ds.assign({cavity+'_single_cavity_mask':ice_mask})

    if return_name:
        return ice_mask, ds, title
    else:
        return ice_mask, ds

# Select a mask for the given region, either continental shelf only ('shelf'), cavities only ('cavity'), or continental shelf with cavities ('all'). Pass it an xarray Dataset as for build_shelf_mask.
def region_mask (region, ds, option='all', return_name=False):

    x_name, y_name = xy_name(ds)

    if return_name:
        # Construct the title
        title = region_names[region]
        if option in ['shelf', 'all']:
            title += ' continental shelf'
            if option == 'all':
                title += ' and'
        if option in ['cavity', 'all']:
            if region in ['filchner_ronne', 'amery', 'ross']:
                title += ' Ice Shelf cavity'
            else:
                title += ' cavities'

    if region+'_'+option+'_mask' in ds:
        # Previously computed
        if return_name:
            return ds[region+'_'+option+'_mask'], ds, title
        else:
            return ds[region+'_'+option+'_mask'], ds

    # Get mask for entire continental shelf and cavities
    mask, ds = build_shelf_mask(ds)
    mask = mask.copy()
    if region == 'all':
        # No further restrictions needed
        pass
    elif region in region_bounds:
        # Restrict based on lat-lon box
        [xmin, xmax, ymin, ymax] = region_bounds[region]
        lon_name, lat_name = latlon_name(ds)
        lon = ds[lon_name]
        lat = ds[lat_name]
        mask = xr.where((lon >= xmin)*(lon <= xmax)*(lat >= ymin)*(lat <= ymax), mask, 0)
        if region in region_bathy_bounds:
            # Also restrict based on isobaths
            bathy = calc_geometry(ds)[0]
            [z_shallow, z_deep] = region_bathy_bounds[region]
            if z_shallow is None:
                z_shallow = bathy.min()
            if z_deep is None:
                z_deep = bathy.max()
            mask = xr.where((bathy >= z_shallow)*(bathy <= z_deep), mask, 0)
        # In all cases, these are shelf-only regions
        option = 'shelf'
    elif region in region_edges:
        # Restrict to a specific region of the coast
        # Select one point each on western and eastern boundaries
        [coord_W, coord_E] = region_edges[region]
        point0_W = closest_point(ds, coord_W)
        [j_W, i_W] = point0_W
        point0_E = closest_point(ds, coord_E)
        [j_E, i_E] = point0_E

        # Make two cuts to disconnect the region
        # Inner function to cut the mask in the given direction: remove the given point and all of its connected neighbours to the N/S or E/W
        def cut_mask (point0, direction):
            if direction == 'NS':
                i = point0[1]
                # Travel north until disconnected
                for j in range(point0[0], ds.sizes[y_name]):
                    if mask[j,i] == 0:
                        break
                    mask[j,i] = 0
                # Travel south until disconnected
                for j in range(point0[0]-1, -1, -1):
                    if mask[j,i] == 0:
                        break
                    mask[j,i] = 0
            elif direction == 'EW':
                j = point0[0]
                # Travel east until disconnected
                for i in range(point0[1], ds.sizes[x_name]):
                    if mask[j,i] == 0:
                        break
                    mask[j,i] = 0
                # Travel west until disconnected
                for i in range(point0[1]-1, -1, -1):
                    if mask[j,i] == 0:
                        break
                    mask[j,i] = 0
        # Inner function to select one cell "west" of the given point - this might not actually be properly west if the cut is made in the east/west direction, in this case you have to choose one cell north or south depending on the direction of travel.
        def cell_to_west (point0, direction):
            (j,i) = point0
            if direction == 'NS':
                # Cell to the west
                return (j, i-1)
            elif direction == 'EW':
                if j_E > j_W:
                    # Travelling north: cell to the south
                    return (j-1, i)
                elif j_E < j_W:
                    # Travelling south: cell to the north
                    return (j+1, i)
                else:
                    raise Exception('Something is wrong with region_edges')

        [flag_W, flag_E] = region_edges_flag[region]
        # Western boundary is inclusive: cut at cell to "west"
        cut_mask(cell_to_west(point0_W, flag_W), flag_W)
        # Eastern boundary is exclusive: cut at that cell
        cut_mask(point0_E, flag_E)              

        # Run remove_disconnected on western point to disconnect the rest of the continental shelf
        mask_region = remove_disconnected(mask, point0_W)
        # Check if it wraps around the periodic boundary
        if i_E < i_W:
            # Make a second region by running remove_disconnected on one cell "west" from eastern point
            mask_region2 = remove_disconnected(mask, cell_to_west(point0_E, flag_E))
            mask_region += mask_region2
        mask.data = mask_region
        if region == 'dotson_cosgrove':
            # Bear Ridge can interrupt this one
            (j,i) = point0_E
            for n in range(2):
                if np.max(mask[j,:]) > 0:
                    # Need to make a second cut (sometimes even a third)
                    i = np.where(mask[j,:] > 0)[0][0]
                    cut_mask((j,i), flag_E)
                    mask_region = remove_disconnected(mask, point0_W)
                    mask.data = mask_region
    else:
        raise Exception('Undefined region '+region)

    if option != 'shelf':
        # Special cases (where common boundaries didn't agree for eORCA1 and eORCA025)
        if region == 'amundsen_sea':
            # Remove bits of Abbot
            mask_excl, ds = single_cavity_mask('abbot', ds)
        elif region == 'filchner_ronne':
            # Remove bits of Brunt
            mask_excl, ds = single_cavity_mask('brunt', ds)
        elif region == 'thwaites':
            # Remove bits of Pine Island 
            mask_excl, ds = single_cavity_mask('pine_island', ds)
        else:
            mask_excl = None
        if region == 'bellingshausen_sea':
            # Add back in bits of Abbot
            mask_incl, ds = single_cavity_mask('abbot', ds)
        elif region == 'east_antarctica':
            # Add back in bits of Brunt
            mask_incl, ds = single_cavity_mask('brunt', ds)
        else:
            mask_incl = None
        if mask_excl is not None:
            mask *= 1-mask_excl
        if mask_incl is not None:
            mask = xr.where(mask_incl, 1, mask)

    # Now select cavities, shelf, or both
    try:
        ice_mask, ds = build_ice_mask(ds)
        if option == 'cavity':
            mask *= ice_mask
        elif option == 'shelf':
            mask *= 1-ice_mask
    except:
        print('Warning: no cavities in this dataset')

    # Save to the Dataset in case it's useful later
    ds = ds.assign({region+'_'+option+'_mask':mask})

    if return_name:
        return mask, ds, title
    else:
        return mask, ds


# Make any 2D mask 3D, masking out any points which are land or ice shelf (varying with depth). Pass a dataset containing thetao or so on the same grid.
def make_mask_3d (mask, ds):

    land_mask_3d = build_mask_3d(ds)
    return xr.broadcast(mask, land_mask_3d)*land_mask_3d    


# Build and return a T grid mask for coastal points: open-ocean points with at least one neighbour that is land or ice shelf.
def get_coast_mask(mask):
    
    open_ocean = (mask.tmask.isel(time_counter=0) == 1)
    land_ice   = ~open_ocean
    
    num_coast_neighbours = neighbours(land_ice, missing_val=0)[-1]
    coast_mask           = (open_ocean*(num_coast_neighbours > 0)).astype(bool)
    
    return coast_mask


# Build and return a 2D mask for the ice shelf front points of the given ice shelf.
def get_icefront_mask(mask, side='ice'):

    # mask of iceshelves (true for iceshelf, false otherwise)
    # isf  = (mesh_new.misf.isel(time_counter=0) > 0) & (mesh_new.tmask.isel(time_counter=0, nav_lev=0) == 0)
    ice_shelf  = (mask.tmaskutil.isel(time_counter=0) - mask.tmask.isel(time_counter=0, nav_lev=0)).astype(bool)
    open_ocean = (mask.tmask.isel(time_counter=0, nav_lev=0) == 1)

    if side == 'ice':
        # Return ice shelf points with at least 1 open-ocean neighbour
        num_open_ocean_neighbours = neighbours(open_ocean, missing_val=0)[-1]
        
        return (ice_shelf*(num_open_ocean_neighbours > 0)).astype(bool)
    elif side == 'ocean':
        # Return ocean points with at least 1 ice shelf neighbour
        num_ice_shelf_neighbours = neighbours(ice_shelf, missing_val=0)[-1]
        
        return (open_ocean*(num_ice_shelf_neighbours > 0)).astype(bool)

# Function to create a NetCDF file that contains the region masks
# Inputs:
# nemo_mesh : string of nemo meshmask file 
# option    : mask type ('all', 'cavity', or 'shelf')
# out_file  : string of output file path
def create_regions_file(nemo_mesh, option, out_file):

    ds = xr.Dataset(
        coords={'nav_lon':(["y","x"], nemo_mesh.nav_lon.values),
                'nav_lat':(["y","x"], nemo_mesh.nav_lat.values)})

    masks={}
    # later should be for name in region_names
    for name in ['amundsen_sea','bellingshausen_sea','larsen','filchner_ronne','ross', 'amery', 'all']:
        mask, _, region_name = region_mask(name, nemo_mesh, option=option, return_name=True)
        masks[name] = mask
        ds = ds.assign({f'mask_{name}':(["y","x"], masks[name].values)})

    ds.to_netcdf(out_file)

    return ds

# Function to extract a variable masked for a specific region
# Inputs:
# ds_nemo      : xarray dataset to extract variable from
# nemo_var     : string of name of variable to extract
# region       : string of region name
# region_type  : (optional) mask type ('all', 'cavity', or 'shelf')
# regions_file : (optional) string path to regions mask definition netcdf file
# nemo_domain  : (optional) string path to NEMO domain cfg file to create mask from
def extract_var_region(ds_nemo, nemo_var, region, 
                       region_type='all', regions_file='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/output/regions.nc', 
                       nemo_domain='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/domain_cfg-20250715.nc'):

    # Calculate a mask or read masks from a regions file
    if not regions_file:
        nemo_file = xr.open_dataset(nemo_domain)
        mask, _   = region_mask(region, nemo_file, option=region_type, return_name=False)
    else: 
        region_masks = xr.open_dataset(regions_file)
        # if the regions_file contains the mask, read it otherwise create it and add it to the regions file
        if f'mask_{region}_{region_type}' in list(region_masks.keys()): 
            mask = region_masks[f'mask_{region}_{region_type}']
        else:
            # move old regions file to a backup location and delete the original file
            region_masks.to_netcdf(f'{regions_file.split('.nc')[0]}_old.nc')
            os.remove(regions_file)
            # calculate new mask
            nemo_file    = xr.open_dataset(nemo_domain)
            mask, _      = region_mask(region, nemo_file, option=region_type, return_name=False)
            # write mask to new regions file 
            region_masks_new = region_masks.copy()
            region_masks_new[f'mask_{region}_{region_type}'] = mask
            region_masks_new.to_netcdf(regions_file)

    # Make mask 3D if needed
    if len(ds_nemo[nemo_var].dims) == 3:
        mask_d = xr.where(ds_nemo[nemo_var]==0, 0, mask)
    else:
        mask_d = mask

    # Extract variable for the specific region and place NaN for land values or locations outside of the region
    region_var = xr.where(mask_d==0, np.nan, ds_nemo[nemo_var]*mask_d)

    return region_var

# Function takes two grid points in coordinates and finds the coordinates of the points connecting a line between pt0 and pt1 
# Inputs:
# pt0, pt1  : tuples of x and y coordinates
# opt_float : (optional) boolean whether to draw a smooth line connecting the points (so with decimals) or floored to integer steps
def connect_coord_points(pt0, pt1, opt_float=True):
    
    dx = pt1[0] - pt0[0]
    dy = pt1[1] - pt0[1]

    # increment the axis with the most steps by steps of 1, then estimate the coordinates of the other vector
    if np.abs(dy) > np.abs(dx):
        vec_y = np.arange(pt0[1], pt1[1] + np.sign(dy), np.sign(dy))
        if opt_float:
            vec_x = np.linspace(pt0[0], pt1[0], len(vec_y))
        else:
            vec_x = np.floor(np.linspace(pt0[0], pt1[0], len(vec_y)))
    else:
        vec_x = np.arange(pt0[0], pt1[0] + np.sign(dx), np.sign(dx))
        if opt_float:
            vec_y = np.linspace(pt0[1], pt1[1], len(vec_x))
        else:
            vec_y = np.floor(np.linspace(pt0[1], pt1[1], len(vec_x)))
    
    return vec_x, vec_y

# Function finds x, y coordinates associated with the specified transect longitude and latitudes
# Inputs:
# data      : xarray dataset to calculate transect for
# waypoints : list of array of longitudes and array of latitudes that are waypoints identifying the route of a transect, 
#             so the transect consists of lines connecting these coordinates
# opt_float : boolean option whether you want to get an array of coordinates as integers (floored) or floats
# Outputs: vector_x, vector_y are coordinates of the dataset that lie along the specified transect route
def transect_coords_from_latlon_waypoints(data, waypoints, opt_float=True):

    # find the indices associated with the specified corner longitude and latitudes
    transect_data_coord = [closest_point(data, (waypoints[0][i],waypoints[1][i])) for i in range(0,len(waypoints[0]))]
    yind = list(zip(*transect_data_coord))[0]
    xind = list(zip(*transect_data_coord))[1]

    # now, connect the x, y turning points to create a transect connecting these coordinates
    vector_x = np.array([]); vector_y = np.array([]);
    for i in range(0,len(xind)-1):
        vec_x, vec_y = connect_coord_points((xind[i], yind[i]),(xind[i+1], yind[i+1]), opt_float=opt_float)
        vector_x = np.append(vector_x, vec_x)
        vector_y = np.append(vector_y, vec_y)

    if opt_float:
        return vector_x, vector_y
    else:
        return vector_x.astype(int), vector_y.astype(int)


# Given a boolean mask of up to 3 dimensions (eg land mask, ice mask...) and the coordinates of a point in which that mask is True (eg [j,i]), return another mask which is True at all the points which are connected to the original point. For example, to isolate an individual ice shelf, or a subglacial lake.
def connected_mask (point0, mask):

    dim = len(point0)
    if dim == 1:
        [nx] = mask.shape
    elif dim == 2:
        [ny,nx] = mask.shape
    elif dim == 3:
        [nz,ny,nx] = mask.shape
    elif dim > 3:
        print(f'Error (connected_mask): not implemented for more than 3 dimensions. This array has {dim} dimensions.')
        sys.exit()
    if not mask[tuple(point0)]:
        print('Error (connected_mask): the given mask is False at the given point.')
        sys.exit()

    # Initialise connected mask to all False
    connected = np.zeros(mask.shape).astype(bool)

    # Inner function to check if a given neighbour to the given point is valid
    def check_neighbour (point, dimension, increment, neighbours):
        new_point = np.copy(point)
        new_point[dimension] = point[dimension] + increment
        if mask[tuple(new_point)]:
            neighbours.append(new_point)
        return neighbours

    # Inner function to get valid immediate neighbours at any point (up to 2*dim)
    def get_immediate_neighbours (point):
        neighbours = []
        if dim == 1:
            [i] = point
        elif dim == 2:
            [j,i] = point
        elif dim == 3:
            [k,j,i] = point
        if i > 0:
            neighbours = check_neighbour(point, -1, -1, neighbours)
        if i+1 < nx:
            neighbours = check_neighbour(point, -1, 1, neighbours)
        if dim > 1:
            if j > 0:
                neighbours = check_neighbour(point, -2, -1, neighbours)
            if j+1 < ny:
                neighbours = check_neighbour(point, -2, 1, neighbours)
        if dim > 2:
            if k > 0:
                neighbours = check_neighbour(point, -3, -1, neighbours)
            if k+1 < nz:
                neighbours = check_neighbour(point, -3, 1, neighbours)
        return neighbours

    # Get started with just point0 to initialise a LIFO stack
    stack = [point0]

    while len(stack) > 0:
        new_point = stack.pop()
        # Update connected mask
        connected[tuple(new_point)] = True
        # Get its immediate neighbours
        neighbours = get_immediate_neighbours(new_point)
        # Loop over these neighbours and add to the stack if they haven't already been dealt with
        for possible_point in neighbours:
            if not connected[tuple(possible_point)]:
                stack.append(possible_point)

    return connected

# Build and return a mask for grounding line points: ice shelf points with at least one neighbour which is grounded ice.
# The default is to only consider grounding lines connected to the "proper" grounded ice sheet (defined as the central southern point of the domain)
# , and to exclude the grounding lines of pinning points etc, but you can override this by setting pinning_points=True.
# Inputs:
#  mesh --- xarray dataset of NEMO mesh_mask file
def get_grounding_line_mask(mesh, pinning_points=False, return_grounded_mask=False):

    land_mask = (mesh.mbathy.squeeze()==0)
    ice_mask  = (mesh.misf.squeeze()!=0)

    if pinning_points:
        # Consider the grounding lines of pinning points too
        grounded_mask = land_mask
    else:
        # Only consider "proper" grounding lines of the grounded ice sheet
        ice_sheet_point0 = [0, round(mesh.x.size/2)]
        grounded_mask = connected_mask(ice_sheet_point0, land_mask)
        num_grounded_neighbours = neighbours(grounded_mask, missing_val=0)[-1]
        gl_mask = (ice_mask*(num_grounded_neighbours > 0)).astype(bool)
        if return_grounded_mask:
            return gl_mask, grounded_mask
        else:
            return gl_mask
