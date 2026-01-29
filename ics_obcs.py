###########################################################
# Generate initial conditions and open boundary conditions.
###########################################################

import numpy as np
import xarray as xr
import cftime
import tqdm
import gsw
from .interpolation import interp_latlon_cf, neighbours, neighbours_z, extend_into_mask
from .utils import fix_lon_range


# Function to compute edges of the z-levels
# Inputs:
# mesh: xarray dataset of the specified grid type to get depth levels and grid cell sizes from
# mtype: string name specifying the mesh type (currently only includes nemo and SOSE)
# source_file: --- > "sose"
def vertical_edges(mesh, mtype='nemo'):

    if mtype=='nemo':    # 2D
        z_centres = mesh.gdept_0.isel(time_counter=0).values
        dz        = mesh.e3t_0.isel(time_counter=0).values
    elif mtype=='SOSE':  # 1D --> 2D
        z_centres, _ = xr.broadcast(mesh.z, mesh[list(mesh.keys())[0]].isel(z=0))
        dz, _        = xr.broadcast(mesh.drF, mesh[list(mesh.keys())[0]].isel(z=0))
        
        z_centres = z_centres.values
        dz        = dz.values
    elif mytpe=='CESM2': 
        z_centres = mesh #!!! add the right variables here
        dz = mesh
    else:
        print('Only mesh types included are nemo, SOSE, CESM2')

    assert (z_centres >= 0.0).all(), 'There is at least one negative depth value (all depths should be positive)'
    
    z_top_edge = z_centres - 0.5*dz
    z_bot_edge = z_centres + 0.5*dz
    
    z_top_edge[z_top_edge < 0] = 0
    
    return (z_top_edge, z_centres, z_bot_edge)

# Function that is called by vertical_interp to actually perform the vertical interpolation ## eventually move to interpolation.py
# Inputs:
# source: xarray source dataset
# source_edges: 3D numpy array of locations of source grid edges
# nemo_edges: 3D numpy array of locations of NEMO grid edges
# n: integer index of vertical depth level within nemo
def interp_depth(source, source_edges, nemo_edges, n):
    # For a particular input nemo depth level, interpolate from source grid to nemo grid (conservatively nearest-neighbour)
    
    NEMO_top_edge = nemo_edges[0][n,:,:]; NEMO_bot_edge = nemo_edges[2][n,:,:];

    dataset = xr.Dataset({}) #.assign_coords(x=source.x, y=source.y)

    for var in source: # loop over the variables in the source dataset
        # find the source edges that fall within the depth ranges of the NEMO grid cells and weight source variable by depth
        Var_total = np.zeros(NEMO_top_edge.shape); 
        for zs in range(0,source.z.size):
            src_top_edge = source_edges[0][zs,:,:]; src_bot_edge = source_edges[2][zs,:,:];

            # NEMO cells that fall fully within the source cell:
            Var = 0
            
            Var = xr.where((NEMO_top_edge >= src_top_edge)*(NEMO_bot_edge <= src_bot_edge), \
                           source.isel(z=zs)*(NEMO_bot_edge - NEMO_top_edge), 0)
        
            # NEMO cells that have an overlap at the bottom: 
            Var = xr.where((NEMO_top_edge <= src_bot_edge)*(NEMO_bot_edge >= src_bot_edge)*(NEMO_top_edge >= src_top_edge), \
                           source.isel(z=zs)*(src_bot_edge - NEMO_top_edge), Var)
        
            # NEMO cells that have an overlap at the top:
            Var = xr.where((NEMO_top_edge < src_top_edge)*(NEMO_bot_edge >= src_top_edge)*(NEMO_bot_edge < src_bot_edge), \
                           source.isel(z=zs)*(NEMO_bot_edge - src_top_edge), Var)

            # source cell smaller than NEMO cell and fully encapsulated:
            Var = xr.where((NEMO_top_edge <= src_bot_edge)*(NEMO_bot_edge >= src_bot_edge)*(NEMO_top_edge < src_top_edge)*(NEMO_bot_edge <= np.max(source_edges[2])), \
                           Var + source.isel(z=zs)*(src_bot_edge - src_top_edge), Var)

            Var_total += Var

        # If the NEMO cell is deeper than the maximum source depth, fill cell with NaN so that fill_ocean can deal with it later
        Var_total    = xr.where((NEMO_bot_edge > np.max(source_edges[2][~np.isnan(source[var])], axis=0)), \
                                 np.nan, Var_total) 

        # Remove depth weighting
        Var_total    = Var_total/(NEMO_bot_edge - NEMO_top_edge)
        dataset[var] = (('y', 'x'), Var_total[var].values)
        
    return dataset

# Helper function to fill the bottom grid cell with values from above to avoid any issues with edges
# Inputs:
# variable: string name of variable that is being interpolated/filled
# file_interp: string name of file produced by vertical_interp function (horizontally and vertically interpolated variable, with missing points not yet filled)
def fill_near_bottom(variable, file_interp):
    # Load file that contains vertical and horizontally interpolated variable:
    var_interp = xr.open_dataset(file_interp)
    
    # fill only those cells that are nearest the bottom with nearest neighbour instead of NaN
    Var_masked = np.ma.masked_where(np.isnan(var_interp[variable].values), var_interp[variable].values)
    ind_array  = np.ma.notmasked_edges(Var_masked, axis=0)
    bottom_ind = ind_array[1][0][:] + 1 # list of Z level associated with each of the y cells
    bottom_ind[bottom_ind>=var_interp.z.size] = var_interp.z.size-1 # max python grid value
    lat_ind    = ind_array[1][1][:] 
    lon_ind    = ind_array[1][2][:]
        
    for i, bot in enumerate(bottom_ind):
        Var_masked[bottom_ind[i], lat_ind[i], lon_ind[i]] = Var_masked[bottom_ind[i]-1, lat_ind[i], lon_ind[i]]
        
    var_interp[variable] = (('z','y','x'), Var_masked.data)
    
    return var_interp

# Helper function to fill missing values (empty cavities) in interpolated source dataset with the connected nearest neighbour 
# Inputs:
# variable: string name of the variable to be filled
# fill_val: the temporary fill value assigned to values in the source dataset that are masked
# niter: maximum number of iterations used to fill connected nearest neighbours
def fill_ocean(input_dataset, variable, nemo_mask, missing_val=-9999, fill_val=np.nan, niter=100, dim='3D', grid='T'):

    print('Filling gaps with connected nearest neighbours')
    if grid=='T': gridmask=nemo_mask.tmask
    elif grid=='U': gridmask=nemo_mask.umask
    elif grid=='V': gridmask=nemo_mask.vmask
    else: print('Must specify grid as type T, U, or V')
    
    if dim=='3D':
        use_3d=True; use_2d=False;
        nemo_ocn = (gridmask.isel(time_counter=0).values == 1)
        # might need to fix the nemo mask at the deepest cell to fill that cell as well
    elif dim=='2D':
        use_2d=True; use_3d=False;
        nemo_ocn = (gridmask.isel(time_counter=0, nav_lev=0).values == 1)

    # Fill gaps in source dataset with nearest neighbour
    src_to_fill = xr.where(np.isnan(input_dataset[variable].values)*nemo_ocn, missing_val, input_dataset[variable].values)
    var_filled  = extend_into_mask(src_to_fill, missing_val=missing_val, fill_val=fill_val, use_2d=use_2d, use_3d=use_3d, num_iters=niter) 

    # Remove any points with values that are actually in the land
    var_filled = xr.where(~nemo_ocn, np.nan, var_filled)
    
    if dim=='3D':   input_dataset[variable] = (('z','y','x'), var_filled)
    elif dim=='2D': input_dataset[variable] = (('y','x')    , var_filled)
        
    return input_dataset 

# Function to vertically interpolate the specified variable
# Inputs:
# interp_info: dictionary containing 'nemo_mask', 'source_coord', 'variable', 'source' 
# in_file: string name of input file
# out_file: string name of output file
def vertical_interp(interp_info, in_file, out_file, ln_obcs=False, bdy_ind=-2):

    # Load horizontally interpolated variable:
    hinterp_var = xr.open_dataset(f'{in_file}')
    
    # Find edges of the NEMO and source dataset vertical levels: 
    nemo_mask_file    = xr.open_dataset(f"{interp_info['nemo_mask']}")
    source_coord_file = xr.open_dataset(f"{interp_info['source_coord']}")
    if ln_obcs:
        bdy_lat           = nemo_mask_file.nav_lat.isel(y=bdy_ind).max().values
        nemo_mask_file    = nemo_mask_file.isel(y=[bdy_ind])
        source_coord_file = source_coord_file.rename(interp_info['renaming']).sel(lat=slice(bdy_lat-1, bdy_lat+1))
        
    if interp_info['source']=='SOSE':
       source_coord   = hinterp_var.assign(drF=(['z'], source_coord_file.drF.values))
    elif interp_info['source']=='CESM2':
       source_coord   = hinterp_var.assign() ### add the coorect variable

    nemo_edges        = vertical_edges(nemo_mask_file, mtype='nemo')
    source_edges      = vertical_edges(source_coord, mtype=interp_info['source'])   

    # Loop over vertical NEMO levels to interpolate slices from the source dataset:
    print(f"Vertically interpolating variable {interp_info['variable']}")
    model_ICs = []
    for n in tqdm.tqdm(range(len(nemo_mask_file.nav_lev.values))):
        data_interp = interp_depth(hinterp_var, source_edges, nemo_edges, n)
        model_ICs.append(data_interp) # Add interpolated layers to dataset
    
    source_interp = xr.concat(model_ICs, dim='z') 
    
    # write to file
    source_interp.to_netcdf(f'{out_file}')
    
    return

def ics_obcs_horizontal_interp(interp_info, in_file, out_file, ln_obcs=False, bdy_ind=-2):

    import sys
    sys.path.append('/home/users/birgal/')
    from nemo_python_git.utils import convert_to_teos10
    
    # Load files:
    nemo_coord_file   = xr.open_dataset(f"{interp_info['nemo_coord']}").squeeze()
    
    # Read variable and slice to the latitude range of interest to reduce size
    source_var = xr.open_dataset(f'{in_file}').rename(interp_info['renaming']).sel(lat=slice(-90, -48))
    if source_var.lon.max() > 180: # Convert longitudes from 0-360 to -180 to 180 
        source_var['lon'] = fix_lon_range(source_var.lon)
        source_var        = source_var.sortby('lon') 
    
    # If open boundary conditions, subset to specific indices
    if ln_obcs:
        bdy_lat         = nemo_coord_file.nav_lat.isel(y=bdy_ind).max()
        nemo_coord_file = nemo_coord_file.isel(y=slice(bdy_ind-1, bdy_ind+2)) ## check whether I need to copy over the edge for this to work correctly
        source_var      = source_var.sel(lat=slice(bdy_lat-2, bdy_lat+2))
        source_ind      = np.argmin(np.abs(source_var.lat.values - bdy_lat.values))
        source_var      = xr.where(source_var[interp_info['variable']].isel(lat=slice(source_ind-1, source_ind+2)) ==0, 
                                   np.nan, source_var.isel(lat=slice(source_ind-1, source_ind+2)))


    # convert temperature and salinity values to TEOS10:
    if interp_info['variable'] == 'PracSal':
        print(f"Converting {interp_info['variable']} to TEOS10")    
        source_converted = convert_to_teos10(source_var, var=interp_info['variable'])
    elif interp_info['variable'] == 'PotTemp':
        print(f"Converting {interp_info['variable']} to TEOS10")
        print('Note: to convert potential temperature to conservative temperature, TEOS10 uses salinity. Make sure a salinity file is specified.')
        if ln_obcs:
            source_salt        = xr.open_dataset(f"{interp_info['salt_file']}").rename(interp_info['renaming']).sel(lat=slice(bdy_lat-1, bdy_lat+1))    
        else:
            source_salt        = xr.open_dataset(f"{interp_info['salt_file']}").rename(interp_info['renaming']).sel(lat=slice(-90, -48))    
        source_salt['lon'] = fix_lon_range(source_salt.lon)
        source_salt        = source_salt.sortby('lon') 
        source_dataset     = source_var.assign(PracSal=source_salt['PracSal'])
        source_converted   = convert_to_teos10(source_dataset, var=interp_info['variable'])
    else:
        print("Proceeding under the assumption that conservative temperature and/or absolute salinity were provided")
        source_converted   = source_var[interp_info['variable']]

    print(f"Horizontally interpolating variable {interp_info['variable']} at each depth level from {interp_info['source']} to NEMO grid")
    datasets = []
    # Loop over all source dataset depth levels:
    if interp_info['dim']=='3D':   z_levels = range(source_var.depth.size)
    elif interp_info['dim']=='2D': z_levels = [0]
    
    for dl in tqdm.tqdm(z_levels):
        if interp_info['source'] == 'SOSE':
            if interp_info['variable']=='UVEL':
                if interp_info['dim']=='3D':
                    var_source = xr.where(source_var.maskW.isel(depth=dl)==1, source_converted.isel(depth=dl), np.nan)
                    var_source = xr.where(var_source==0, np.nan, var_source) # needed to mask a couple of missing land points
            elif interp_info['variable']=='VVEL':
                if interp_info['dim']=='3D':
                    var_source = xr.where(source_var.maskS.isel(depth=dl)==1, source_converted.isel(depth=dl), np.nan)
                    var_source = xr.where(var_source==0, np.nan, var_source) # needed to mask a couple of missing land points
            else:
                # Mask values that are on land in the source dataset
                if interp_info['dim']=='3D':
                    var_source = xr.where(source_var.maskC.isel(depth=dl)==1, source_converted.isel(depth=dl), np.nan)
                    var_source = xr.where(var_source==0, np.nan, var_source) # needed to mask a couple of missing land points
                elif interp_info['dim']=='2D':
                    var_source = xr.where(source_converted==0, np.nan, source_converted)
                    var_source = xr.where((source_var.maskInC == 1) & np.isnan(var_source), 0, var_source)
                    var_source = xr.where(source_var.maskInC == 1, var_source, np.nan)

            # Now wrap up into a new Dataset
            ds_source = xr.Dataset({'lon':source_var['lon'], 'lat':source_var['lat'], interp_info['variable']:var_source}) 
            
            # Interpolate slices of depth levels along lat-lon (horizontally)
            interp_src = interp_latlon_cf(ds_source, nemo_coord_file, pster_src=False, periodic_src=True, periodic_nemo=True, method='conservative')

            if ln_obcs:
                datasets.append(interp_src.isel(y=slice(1,2))) # since the horizontal slice went from bdy_ind-1 to bdy_ind+1 
            else:
                datasets.append(interp_src)
    
    if interp_info['dim'] =='3D':
       source_interpolated = xr.concat(datasets, dim='z').assign_coords(z=np.abs(source_var.depth.values[0:dl+1])) 
    elif interp_info['dim'] =='2D':
       source_interpolated = interp_src

    source_interpolated.to_netcdf(f'{out_file}')
    
    return
    

# Main function to create inititial conditions for NEMO configuration from a source, currently set up for B-SOSE
# Input:
# in_file: string location of file to be interpolated
# variable: string of name of variable in in_file to be interpolated
# dataset: string specifying the source of the dataset (currently only set up for 'SOSE')
# folder_ICs: string location of folder that contains all the relevant files to read in and that will be used to write output files
# Output:
# Three files: for example, SOSE-SALT-horizontal-interp.nc, SOSE-SALT-vertical-interp.nc, SOSE-SALT-initial-conditions.nc
def create_ics(variable, in_file, out_file,
               source='SOSE',
               source_coord='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/B-SOSE/climatology/SALT_climatology_m01.nc',
               nemo_coord  ='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/coordinates_AIS.nc',
               nemo_mask   ='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20260121.nc',
               salt_file   ='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/B-SOSE/climatology/SALT_climatology_m01.nc', 
               folder      ='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/initial-conditions/',
               fill_value  = np.nan,
               land_value  = np.nan,
               num_iter    = 100,
               grid_type   = 'T'):

    print(f'---- Creating NEMO initial conditions for variable {variable} from {source} ----')
    if source not in ['SOSE', 'NEMO']: raise Exception('Functions only set up for SOSE or NEMO restart file currently')

    # Check number of dimensions of variable (2D or 3D):
    dimension = f"{len(xr.open_dataset(f'{in_file}')[variable].dims)}D"
    if dimension != '2D' and dimension !='3D': raise Exception('Input variable must be either 2D or 3D')
    
    # Specify coordinate names:
    if source=='SOSE':
        if dimension=='2D':   name_remapping = {'XC':'lon', 'YC':'lat'}
        elif dimension=='3D': name_remapping = {'XC':'lon', 'YC':'lat', 'Z':'depth'}
        
    # Dictionary specifying file names and locations for subsequent functions:
    interp_info = {'source': source,
                   'variable': variable,
                   'nemo_coord': nemo_coord,
                   'nemo_mask': nemo_mask,
                   'source_coord': source_coord,
                   'salt_file':salt_file,
                   'dim':dimension,
                   'renaming':name_remapping}
    
    # Horizontally interpolate source dataset to NEMO grid:
    ics_obcs_horizontal_interp(interp_info, in_file, f'{folder}temp/{source}-{variable}-IC-horizontal-interp.nc')
        
    if dimension=='3D':
       # Vertically interpolate the above horizontally interpolated dataset to NEMO grid:
       vertical_interp(interp_info, f'{folder}temp/{source}-{variable}-IC-horizontal-interp.nc', f'{folder}temp/{source}-{variable}-IC-vertical-interp.nc')
       SOSE_interp   = xr.open_dataset(f'{folder}temp/{source}-{variable}-IC-vertical-interp.nc')
    elif dimension=='2D':
       SOSE_interp   = xr.open_dataset(f'{folder}temp/{source}-{variable}-IC-horizontal-interp.nc')

    # Fill areas that are masked in source dataset but not in NEMO with nearest neighbours:
    nemo_mask_ds  = xr.open_dataset(f'{nemo_mask}')
    SOSE_extended = fill_ocean(SOSE_interp, variable, nemo_mask_ds, dim=dimension, niter=num_iter, fill_val=fill_value, grid=grid_type)

    # Final processing (fill NaNs with a real value and shift very deepest grid cell value):
    if ~np.isnan(land_value):
        SOSE_extended[variable] = xr.where(np.isnan(SOSE_extended[variable]), land_value, SOSE_extended[variable])
       # SOSE_extended[variable] = xr.where(np.abs(SOSE_extended[variable]) < 1e-4, land_value, SOSE_extended[variable]) # for sea ice masking
    else:
        SOSE_extended[variable] = xr.where(np.isnan(SOSE_extended[variable]), 9999, SOSE_extended[variable])
    if dimension=='3D':
        SOSE_extended[variable] = xr.where(SOSE_extended.z == SOSE_extended.z[-1], SOSE_extended[variable].isel(z=-2), SOSE_extended[variable])
        SOSE_extended[variable] = ('time_counter','deptht','y','x'), SOSE_extended[variable].values[np.newaxis, ...]
    elif dimension=='2D':
        SOSE_extended[variable] = ('time_counter','y','x'), SOSE_extended[variable].values[np.newaxis, ...]
        
    # Write output to file:
    SOSE_extended.to_netcdf(f'{out_file}', unlimited_dims='time_counter')
    return

# Main function to create boundary conditions for NEMO configuration from a source, currently set up for B-SOSE
# Input:
# in_file: string location of file to be interpolated
# variable: string of name of variable in in_file to be interpolated
# dataset: string specifying the source of the dataset (currently only set up for 'SOSE')
# folder_ICs: string location of folder that contains all the relevant files to read in and that will be used to write output files
# bdy_ind: assumes that the boundary is a full edge of the circumpolar domain. Python based indexing.
# grid_type: 
# Output:
# Three files: for example, SOSE-SALT-horizontal-interp.nc, SOSE-SALT-vertical-interp.nc, SOSE-SALT-initial-conditions.nc
def create_bcs(variable, in_file, out_file,
               source='SOSE',
               source_coord='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/B-SOSE/climatology/SALT_climatology_m01.nc',
               nemo_coord  ='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/coordinates_AIS.nc',
               nemo_mask   ='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20260121.nc',
               salt_file   ='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/B-SOSE/climatology/SALT_climatology_m01.nc', 
               folder      ='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/boundary-conditions/B-SOSE/',
               fill_value  = np.nan,
               land_value  = np.nan,    
               bdy_ind     = 451, 
               grid_type   = 'T'):

    print(f'---- Creating NEMO boundary conditions for variable {variable} from {source} ----')
    if source not in ['SOSE', 'CESM2']: raise Exception('Functions only set up for SOSE and CESM2 currently')

    # Check number of dimensions of variable (2D or 3D):
    dimension = f"{len(xr.open_dataset(f'{in_file}')[variable].dims)}D"
    if dimension != '2D' and dimension !='3D': raise Exception('Input variable must be either 2D or 3D')

    # Specify coordinate names:
    if source=='SOSE':
        if variable=='UVEL':
            if dimension=='3D': name_remapping = {'XG':'lon', 'YC':'lat', 'Z':'depth'}
        elif variable=='VVEL':
            if dimension=='3D': name_remapping = {'XC':'lon', 'YG':'lat', 'Z':'depth'}
        else:
            if dimension=='2D':   name_remapping = {'XC':'lon', 'YC':'lat'}
            elif dimension=='3D': name_remapping = {'XC':'lon', 'YC':'lat', 'Z':'depth'}
    elif source=='CESM2':
        if variable in ['UVEL', 'VVEL']:
            if dimension=='3D': name_remapping = {'XG':'ULONG', 'YC':'ULAT', 'z_t':'depth'}
        else:
            if dimension=='2D':   name_remapping = {'TLONG':'lon', 'TLAT':'lat'}
            elif dimension=='3D': name_remapping = {'TLONG':'lon', 'TLAT':'lat', 'z_t':'depth'}
        
    # Dictionary specifying file names and locations for subsequent functions:
    interp_info = {'source': source,
                   'variable': variable,
                   'nemo_coord': nemo_coord,
                   'nemo_mask': nemo_mask,
                   'source_coord': source_coord,
                   'salt_file': salt_file,
                   'dim': dimension,
                   'renaming': name_remapping}
    
    # Horizontally interpolate source dataset to NEMO grid:
    ics_obcs_horizontal_interp(interp_info, in_file, f'{folder}temp/{source}-{variable}-BC-horizontal-interp.nc', ln_obcs=True, bdy_ind=bdy_ind)

    # take a slice of the nemo mask:
    nemo_coord_ds = xr.open_dataset(f'{nemo_coord}')
    nemo_mask_ds  = xr.open_dataset(f'{nemo_mask}').isel(y=[bdy_ind])
        
    if dimension=='3D':
       # Vertically interpolate the above horizontally interpolated dataset to NEMO grid:
       vertical_interp(interp_info, f'{folder}temp/{source}-{variable}-BC-horizontal-interp.nc', \
                       f'{folder}temp/{source}-{variable}-BC-vertical-interp.nc', ln_obcs=True, bdy_ind=bdy_ind)
       source_interp = xr.open_dataset(f'{folder}temp/{source}-{variable}-BC-vertical-interp.nc')
    elif dimension=='2D':
       # Fill areas that are masked in source dataset but not in NEMO with nearest neighbours:
       source_interp = xr.open_dataset(f'{folder}temp/{source}-{variable}-BC-horizontal-interp.nc').isel(y=1)

    # Fill areas that are masked in source dataset but not in NEMO with nearest neighbours:
    source_extended = fill_ocean(source_interp, variable, nemo_mask_ds, dim=dimension, niter=100, fill_val=fill_value, grid=grid_type)
    if grid_type=='T': gridmask=nemo_mask_ds.tmask
    elif grid_type=='U': gridmask=nemo_mask_ds.umask
    elif grid_type=='V': gridmask=nemo_mask_ds.vmask
    else: print('Must specify grid as type T, U, or V')
    source_extended[variable] = source_extended[variable].roll(x=-1)

    # Final processing (fill NaNs with a real value and shift very deepest grid cell value):
    if ~np.isnan(land_value):
        source_extended[variable] = xr.where(np.isnan(source_extended[variable]), land_value, source_extended[variable])
        #source_extended[variable] = xr.where(np.abs(source_extended[variable]) < 1e-4, land_value, source_extended[variable]) # for sea ice masking
    else:
        source_extended[variable] = xr.where(np.isnan(source_extended[variable]), 9999, source_extended[variable])
    if dimension=='3D':
        source_extended[variable] = xr.where(source_extended.z == source_extended.z[-1], source_extended[variable].isel(z=-2), source_extended[variable])
        source_extended[variable] = ('time_counter','deptht','y','x'), source_extended[variable].values[np.newaxis, ...]
    elif dimension=='2D':
        source_extended[variable] = ('time_counter','y','x'), source_extended[variable].values[np.newaxis, ...]

    # Write output to file:
    # TO DO: replace x, y with normal index values and add nav_lon, nav_lat of boundary to dataset as variables
    source_extended.assign_coords(x=nemo_mask_ds.nav_lon.isel(y=0).values, y=[nemo_mask_ds.nav_lat.isel(x=0,y=0).values]).to_netcdf(f'{out_file}', \
                                                                                                                                  unlimited_dims='time_counter')
    return
