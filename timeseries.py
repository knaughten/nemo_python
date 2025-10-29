import xarray as xr
import numpy as np
import os
import glob
from .constants import region_points, region_names, rho_fw, rho_ice, sec_per_year, deg_string, gkg_string, drake_passage_lon0, drake_passage_lat_bounds
from .utils import add_months, closest_point, month_convert, bwsalt_abs, xy_name
from .grid import single_cavity_mask, region_mask, calc_geometry
from .diagnostics import transport, gyre_transport
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

# Calculate a timeseries of the given preset variable from an xarray Dataset of NEMO output (must have halo removed). Returns DataArrays of the timeseries data, the associated time values, and the variable title. Specify whether there is a halo (true for periodic boundaries in NEMO 3.6).
# Preset variables include:
# <region>_massloss: basal mass loss from the given ice shelf or region of multiple ice shelves (eg brunt, amundsen_sea)
# <region>_draft: area-averaged ice draft from the given ice shelf or region of multiple ice shelves - only useful if there's a coupled ice sheet
# <region>_bwtemp, <region>_bwsalt: area-averaged bottom water temperature or salinity from the given region or cavity (eg ross_cavity, ross_shelf, ross)
# <region>_bwSA: as for bwsalt, but convert from practical salinity (assumes NEMO3.6 input) to absolute salinity
# <region>_temp, <region>_salt: volume-averaged temperature or salinity from the given region or cavity
# <region>_temp_btw_xxx_yyy_m, <region>_salt_btw_xxx_yyy_m: volume-averaged temperature or salinity from the given region or cavity, between xxx and yyy metres (positive integers, shallowest first)
# drake_passage_transport: zonal transport across Drake Passage (need to pass path to domain_cfg)
# <region>_iceberg_melt: iceberg melt flux integrated over the given region
# <region>_pminuse: precipitation minus evaporation integrated over the given region
# <region>_runoff: runoff integrated over the given region
# <region>_seaice_meltfreeze: sea ice to ocean water flux (melting minus freezing) integrated over the given region
# Inputs:
# name_remapping: optional dictionary of dimensions and variable names that need to be remapped to match the code below (depends on the runset)
# nemo_mesh: optional string of the location of a bathymetry meshmask file for calculating the region masks (otherwise calculates it from ds_nemo)
def calc_timeseries (var, ds_nemo, name_remapping='', nemo_mesh='', 
                     domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc', halo=True, periodic=True):

    # Remap NetCDF variable names to match the generalized case:
    if name_remapping:
        try:
            ds_nemo = ds_nemo.rename(name_remapping)
        except: # if it doesn't seem to need to be renamed, continue looping through
            pass
    
    # Parse variable name
    factor = 1
    region_type = None
    region = None
    nemo_var = None
    if var.endswith('_massloss'):
        option = 'area_int'
        region = var[:var.index('_massloss')]
        region_type = 'cavity'
        nemo_var = 'sowflisf'
        # Convert from kg/s to Gt/y and swap sign
        factor = -rho_ice/rho_fw*1e-12*sec_per_year
        units = 'Gt/y'
        title = 'Basal mass loss'
    elif var.endswith('_draft'):
        option = 'area_avg'
        region = var[:var.index('_draft')]
        region_type = 'cavity'
        nemo_var = 'draft'  # Will trigger a special case later
        units = 'm'
        title = 'Mean ice draft'
    elif var.endswith('_bwtemp'):
        option = 'area_avg'
        region = var[:var.index('_bwtemp')]
        nemo_var = 'tob'
        units = deg_string+'C'
        title = 'Bottom temperature'
    elif var.endswith('_bwsalt'):
        option = 'area_avg'
        region = var[:var.index('_bwsalt')]
        nemo_var = 'sob'
        units = gkg_string
        title = 'Bottom salinity'
    elif var.endswith('_bwSA'):
        option = 'area_avg'
        region = var[:var.index('_bwSA')]
        nemo_var = 'bwSA'  # will trigger special case
        units = gkg_string
        title = 'Bottom salinity'
    elif var.endswith('_ssh'):
        option = 'area_avg'
        region = var[:var.index('_ssh')]
        nemo_var = 'zos'
        units = 'm'
        title = 'Sea surface height'
    elif var.endswith('_temp'):
        option = 'volume_avg'
        region = var[:var.index('_temp')]
        nemo_var = 'thetao'
        units = deg_string+'C'
        title = 'Volume-averaged temperature'
    elif var.endswith('_salt'):
        option = 'volume_avg'
        region = var[:var.index('_salt')]
        nemo_var = 'so'
        units = gkg_string
        title = 'Volume-averaged salinity'
    elif '_temp_btw_' in var:
        option = 'avg_btw_depths'
        region = var[:var.index('_temp_btw_')]
        z_vals = var[len(region+'_temp_btw_'):-1]
        z_shallow = int(z_vals[:z_vals.index('_')])
        z_deep = int(z_vals[z_vals.index('_')+1:])
        nemo_var = 'thetao'
        title = 'Average temperature between '+str(z_shallow)+'-'+str(z_deep)+'m'
        units = deg_string+'C'
    elif '_salt_btw_' in var:
        option = 'avg_btw_depths'
        region = var[:var.index('_salt_btw_')]
        z_vals = var[len(region+'_salt_btw_'):-1]
        z_shallow = int(z_vals[:z_vals.index('_')])
        z_deep = int(z_vals[z_vals.index('_')+1:])
        nemo_var = 'so'
        title = 'Average salinity between '+str(z_shallow)+'-'+str(z_deep)+'m'
        units = gkg_string
    elif var == 'drake_passage_transport':
        lon0 = drake_passage_lon0
        lat_bounds = drake_passage_lat_bounds
        lat0 = None
        lon_bounds = None
        option = 'transport'
        units = 'Sv'
        title = 'Drake Passage Transport'
    elif var == 'weddell_gyre_transport':
        region = 'weddell'
        option = 'gyre_transport'
        units = 'Sv'
        title = 'Weddell Gyre Transport'
    elif var == 'ross_gyre_transport':
        region = 'ross'
        option = 'gyre_transport'
        units = 'Sv'
        title = 'Ross Gyre Transport'
    elif var.endswith('_iceberg_melt'):
        option = 'area_int'
        region = var[:var.index('_iceberg_melt')]
        region_type = 'shelf'
        nemo_var = 'ficeberg'
        # Convert from kg/s to m^3/y
        factor = 1e-3*sec_per_year
        units = 'm^3/y'
        title = 'Iceberg melt'
    elif var.endswith('_pminuse'):
        option = 'area_int'
        region = var[:var.index('_pminuse')]
        region_type = 'shelf'
        nemo_var = 'pminuse'  # Will trigger special case to do pr+prsn-evs
        factor = 1e-3*sec_per_year
        units = 'm^3/y'
        title = 'Precipitation minus evaporation'
    elif var.endswith('_runoff'):
        option = 'area_int'
        region = var[:var.index('_runoff')]
        region_type = 'shelf'
        nemo_var = 'friver'
        factor = 1e-3*sec_per_year
        units = 'm^3/y'
        title = 'Runoff'
    elif var.endswith('_seaice_meltfreeze'):
        option = 'area_int'
        region = var[:var.index('_seaice_meltfreeze')]
        region_type = 'shelf'
        nemo_var = 'fsitherm'
        factor = 1e-3*sec_per_year
        units = 'm^3/y'
        title = 'Sea ice melting minus freezing'

    if var == 'drake_passage_transport' and 'e2u' not in ds_nemo:
        # Need to add e2u from domain_cfg
        ds_domcfg = xr.open_dataset(domain_cfg, decode_times=time_coder).squeeze()
        if ds_nemo.sizes['y'] < ds_domcfg.sizes['y']:
            # The NEMO dataset was trimmed (eg by MOOSE for UKESM) to the southernmost latitudes. Do the same for domain_cfg.
            ds_domcfg = ds_domcfg.isel(y=slice(0, ds_nemo.sizes['y']))
        if halo:
            ds_domcfg = ds_domcfg.isel(x=slice(1,-1))
        ds_nemo = ds_nemo.assign({'e2u':ds_domcfg['e2u']})

    # Some variables have two equivalent options - allow for either
    if nemo_var == 'sowflisf' and nemo_var not in ds_nemo:
        if 'fwfisf' in ds_nemo:
            nemo_var = 'fwfisf'
            factor *= -1
        else:
            raise Exception('Missing variable '+nemo_var+' or fwfisf')
    if nemo_var == 'tob' and nemo_var not in ds_nemo:
        if 'sbt' in ds_nemo:
            nemo_var = 'sbt'
        else:
            raise Exception('Missing variable '+nemo_var+' or sbt')
    if nemo_var == 'sob' and nemo_var not in ds_nemo:
        if 'sbs' in ds_nemo:
            nemo_var = 'sbs'
        else:
            raise Exception('Missing variable '+nemo_var+' or sbs')
    if 'x_grid_T_inner' in ds_nemo.dims:
        ds_nemo = ds_nemo.swap_dims({'x_grid_T_inner':'x_grid_T', 'y_grid_T_inner':'y_grid_T'})

    # Select region
    if region is not None:
        if region_type is None:
            if region.endswith('cavity'):
                region = region[:region.index('_cavity')]
                region_type = 'cavity'
            elif region.endswith('shelf'):
                region = region[:region.index('_shelf')]
                region_type = 'shelf'
            else:
                region_type = 'all'
        if region in region_points and region_type == 'cavity':
            # Single ice shelf
            if nemo_mesh:
                nemo_file = xr.open_dataset(nemo_mesh, decode_times=time_coder)
                mask, _, region_name = single_cavity_mask(region, nemo_file, return_name=True)
            else:
                mask, ds_nemo, region_name = single_cavity_mask(region, ds_nemo, return_name=True)
        else:
            if nemo_mesh:
                nemo_file = xr.open_dataset(nemo_mesh, decode_times=time_coder)
                mask, _, region_name = region_mask(region, nemo_file, option=region_type, return_name=True)
            else:
                mask, ds_nemo, region_name = region_mask(region, ds_nemo, option=region_type, return_name=True)
      
        title += ' for '+region_name

    def area_name ():
        for v in ['area', 'area_grid_T', 'area_grid_U', 'area_grid_V']:
            if v in ds_nemo:
                return v
    def dz_name ():
        for v in ['thkcello', 'e3t']:
            if v in ds_nemo:
                return v
    x_name, y_name = xy_name(ds_nemo)

    if option == 'area_int':
        # Area integral
        dA = ds_nemo[area_name()]*mask
        if nemo_var == 'pminuse':
            data_xy = ds_nemo['pr'] + ds_nemo['prsn'] - ds_nemo['evs']
        else:
            data_xy = ds_nemo[nemo_var]
        data = (data_xy*dA).sum(dim=[x_name, y_name])
    elif option == 'area_avg':
        # Area average
        dA = ds_nemo[area_name()]*mask
        if nemo_var == 'draft':
            data_xy = calc_geometry(ds_nemo, keep_time_dim=True)[1]
        elif nemo_var == 'bwSA':
            data_xy = bwsalt_abs(ds_nemo)
        else:
            data_xy = ds_nemo[nemo_var]
        data = (data_xy*dA).sum(dim=[x_name, y_name])/dA.sum(dim=[x_name, y_name])
    elif option == 'volume_avg':
        # Volume average
        # First need a 3D mask
        mask_3d = xr.where(ds_nemo[nemo_var]==0, 0, mask)
        dV = ds_nemo[area_name()]*ds_nemo[dz_name()]*mask_3d
        data = (ds_nemo[nemo_var]*dV).sum(dim=[x_name, y_name,'deptht'])/dV.sum(dim=[x_name, y_name, 'deptht'])
    elif option == 'avg_btw_depths':
        # Volume average between two depths
        # Create an extra mask to multiply dV with, which is 1 between the two depths and 0 otherwise
        depth_below = ds_nemo[dz_name()].cumsum(dim='deptht')
        depth_above = depth_below.shift(deptht=1, fill_value=0)
        depth_centres = 0.5*(depth_above + depth_below)
        mask_depth = xr.where((depth_centres >= z_shallow)*(depth_centres <= z_deep), 1, 0)
        mask_3d = xr.where(ds_nemo[nemo_var]==0, 0, mask)
        dV = ds_nemo[area_name()]*ds_nemo[dz_name()]*mask_3d*mask_depth
        data = (ds_nemo[nemo_var]*dV).sum(dim=[x_name, y_name,'deptht'])/dV.sum(dim=[x_name, y_name,'deptht'])
    elif option == 'transport':
        # Calculate zonal or meridional transport
        data = transport(ds_nemo, lon0=lon0, lat0=lat0, lon_bounds=lon_bounds, lat_bounds=lat_bounds)
    elif option == 'gyre_transport':
        ds_domcfg = xr.open_dataset(domain_cfg, decode_times=time_coder).squeeze()
        if ds_nemo.sizes[y_name] < ds_domcfg.sizes[y_name]:
            # The NEMO dataset was trimmed (eg by MOOSE for UKESM) to the southernmost latitudes. Do the same for domain_cfg.
            ds_domcfg = ds_domcfg.isel(y=slice(0, ds_nemo.sizes[y_name]))
        if halo:
            ds_domcfg = ds_domcfg.isel(x=slice(1,-1))

        # No longer needed because gyre_transport allows for any choice of thkcello, e3u, e3v
        '''if 'thkcello' in ds_nemo:
            # dataset already contains thkcello (cell thickness) so don't need to rename e3u/v to thkcello
            data = gyre_transport(region, ds_nemo, ds_nemo, ds_domcfg, periodic=periodic, halo=halo)
        else:
            data = gyre_transport(region, ds_nemo.rename({'e3u':name_remapping['e3u']}), ds_nemo.rename({'e3v':name_remapping['e3v']}), ds_domcfg, periodic=periodic, halo=halo)'''
       
    data *= factor
    data = data.assign_attrs(long_name=title, units=units)

    return data, ds_nemo


# As above, but for PP output files from the UM atmosphere. 
def calc_timeseries_um (var, file_path):

    import iris
    import warnings
    import cftime

    # Parse variable name
    if var == 'global_mean_sat':
        option = 'area_avg'
        um_var = 'air_temperature'
        units = 'K'
        title = 'Global mean near-surface air temperature'

    # Read the correct variable from the file
    # Suppress warnings (year_zero kwarg ignored for idealised calendars)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cube = iris.load_cube(file_path, um_var) 

    if option == 'area_avg':
        # Following code by Jane Mulcahy and Catherine Hardacre
        if cube.coord('latitude').bounds is None:
            cube.coord('latitude').guess_bounds()
        if cube.coord('longitude').bounds is None:
            cube.coord('longitude').guess_bounds()
        grid_areas = iris.analysis.cartography.area_weights(cube)
        data_iris = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)

    # Now convert to a DataArray and get the time dimension to match NEMO conventions
    data = xr.DataArray([float(data_iris.data)], coords={'time_counter':[0.]})
    data = data.assign_coords(time_centered=('time_counter', cftime.num2date(cube.coord('time').points, cube.coord('time').units.name, calendar=cube.coord('time').units.calendar)))
    data.load()

    return data


# Helper function to overwrite file
# Make a temporary file and then rename it to the old file. This is safer than doing it in one line with .to_netcdf, because if that returns an error the original file will be deleted and data will be lost.
def overwrite_file (ds_new, timeseries_file):
    timeseries_file_tmp = timeseries_file.replace('.nc', '_tmp.nc')
    ds_new.to_netcdf(timeseries_file_tmp, mode='w')
    os.rename(timeseries_file_tmp, timeseries_file)
    ds_new.close()


# Precompute the given list of timeseries from the given xarray Dataset of NEMO output (or PP file if pp=True). Save in a NetCDF file which concatenates after each call to the function.
def precompute_timeseries (ds_nemo, timeseries_types, timeseries_file, halo=True, periodic=True, 
                           domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc',
                           name_remapping='', nemo_mesh='', pp=False):

    if halo and not pp:
        # Remove the halo
        ds_nemo = ds_nemo.isel(x=slice(1,-1))

    # Calculate each timeseries and save to a Dataset
    ds_new = None
    for var in timeseries_types:
        #print('...'+var)
        if pp:
            data = calc_timeseries_um(var, ds_nemo)
        else:
            #try:
            data, ds_nemo = calc_timeseries(var, ds_nemo, domain_cfg=domain_cfg, halo=halo, periodic=periodic, name_remapping=name_remapping, nemo_mesh=nemo_mesh)
            '''except(KeyError):
                # Incomplete dataset missing some crucial variables. This can happen when grid-T is present but isf-T is missing, or vice versa. Return a masked value.
                print('Warning: missing variables')
                data = ds_nemo['time_counter'].where(False)'''
        if ds_new is None:            
            ds_new = xr.Dataset({var:data})
        else:
            ds_new = ds_new.assign({var:data})
    # Use time_centered as the dimension as it includes real times - time_counter is reset to 0 every output file
    ds_new = ds_new.swap_dims({'time_counter':'time_centered'})
    if pp:
        ds_new = ds_new.drop_vars({'time_counter'})

    if os.path.isfile(timeseries_file):
        # File already exists; read it
        ds_old = xr.open_dataset(timeseries_file, decode_times=time_coder)
        if 'forecast_period' in ds_old:
            # Old formulation using conversion from Iris (breaks in 2300s with datetime overflow): drop unused coordinates
            ds_old = ds_old.drop_vars(['forecast_period', 'forecast_reference_time', 'height', 'latitude', 'longitude'])
        # Concatenate new data
        ds_new.load()
        ds_new = xr.concat([ds_old, ds_new], dim='time_centered')
        ds_old.close()

    # Save to file
    overwrite_file(ds_new, timeseries_file)


# Precompute timeseries from the given simulation, either from the beginning (timeseries_file does not exist) or picking up where it left off (timeseries_file does exist). Considers all NEMO output files stamped with suite_id in the given directory sim_dir on the given grid (gtype='T', 'U', etc), and assumes the timeseries file is in that directory too (unless timeseries_dir is set).
def update_simulation_timeseries (suite_id, timeseries_types, timeseries_file='timeseries.nc', timeseries_dir=None, config='', 
                                  sim_dir='./', freq='m', halo=True, periodic=True, gtype='T', name_remapping='', nemo_mesh='',
                                  domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc', compressed=False):
    import re
    from datetime import datetime

    if timeseries_dir is None:
        timeseries_dir = sim_dir
    update = os.path.isfile(timeseries_dir+timeseries_file)
    if update:
        # Timeseries file already exists
        # Get last time index
        ds_ts = xr.open_dataset(timeseries_dir+timeseries_file, decode_times=time_coder)
        time_last  = ds_ts['time_centered'][-1].dt
        year_last  = time_last.year
        month_last = time_last.month
        ds_ts.close()

    # Identify NEMO output files in the given directory, constructed as wildcard strings for each date code
    nemo_files = []
    if config=='eANT025':
        if compressed:
            file_tail = f'_{gtype}_compressed.nc'
        else:
            file_tail = f'_{gtype}.nc'
    else:
        file_tail = f'-{gtype}.nc'
        
    for f in os.listdir(sim_dir):
        if os.path.isdir(f'{sim_dir}/{f}'): continue # skip directories
        if not f.endswith(file_tail): continue    # Not a NEMO output file on this grid; skip it

        if config=='eANT025':
            if f.startswith(f'{config}.{suite_id}_1{freq}_'):
                # file naming conventions
                file_head = f'{config}.{suite_id}_1{freq}_' 
            else:
                # Something else; skip it
                continue
        else:
            if f.startswith('nemo_'+suite_id+'o_1'+freq+'_'):
                # UKESM file naming conventions
                file_head = 'nemo_'+suite_id+'o_1'+freq+'_'
            elif f.startswith(suite_id+'_1'+freq+'_'):
                # Standalone NEMO file naming conventions
                file_head = suite_id+'_1'+freq+'_'
            else:
                # Something else; skip it
                continue
        # Extract date code    
        date_code  =  re.findall(r'\d{4}\d{2}\d{2}', f) # find parts of filename that look like a date, typically two dates in NEMO
        file_dates = []
        for d in date_code:
            file_date = datetime.strptime(d, '%Y%m%d').date()
            file_dates.append(file_date) # list containing dates in the filename
        if update:
            # Need to check if date code has already been processed
            year = file_dates[0].year
            month = file_dates[0].month
            if year < year_last or (year==year_last and month<=month_last):
                # Skip it
                continue
        # Now construct wildcard string and add to list if it's not already there
        file_pattern = f'{file_head}{date_code[0]}?{date_code[1]}*{file_tail}'
        if file_pattern not in nemo_files:
            nemo_files.append(file_pattern)        
    # Now sort alphabetically - i.e. by ascending date code
    nemo_files.sort()

    # Loop through each date code and process
    for file_pattern in nemo_files:
        print('Processing '+file_pattern)
        # The following block doesn't work any more - maybe a regex problem? Was just checking for warnings anyway
        '''has_isfT = os.path.isfile(f"{sim_dir}/{file_pattern.replace('*','_isf')}")
        has_gridT = os.path.isfile(f"{sim_dir}/{file_pattern.replace('*','_grid')}")
        if sum([has_isfT, has_gridT]) == 1:
            if has_isfT and not has_gridT:
                print('Warning: isf-T file exists with no matching grid-T file.')
            else:
                print('Warning: grid-T file exists with no matching isf-T file.')
            if file_pattern == nemo_files[-1]:
                print('This is the last file, so it will probably be pulled from MASS later. Stopping.')
                break
            else:
                print('Timeseries file will have some NaNs at this index.')'''

        dsV = None
        ds_SBC = None
        if 'weddell_gyre_transport' in timeseries_types or 'ross_gyre_transport' in timeseries_types: # need to load both gridU and grid V files to be able to calculate this; not currently the neatest approach
            if gtype not in ['U', 'V']:
                raise Exception('Grid type must be specified as either U or V when calculating gyre transport') # should be U and V:
            ds_nemo = xr.open_dataset(glob.glob(f'{sim_dir}/{file_pattern}'.replace('V.nc', 'U.nc'))[0], decode_times=time_coder)[['e3u','uo']].rename({'nav_lon':'nav_lon_grid_U','nav_lat':'nav_lat_grid_U'})
            dsV = xr.open_dataset(glob.glob(f'{sim_dir}/{file_pattern}'.replace('U.nc', 'V.nc'))[0], decode_times=time_coder)[['e3v','vo']].rename({'nav_lon':'nav_lon_grid_V','nav_lat':'nav_lat_grid_V'})
        else:
            ds_nemo = xr.open_mfdataset(f'{sim_dir}/{file_pattern}', decode_times=time_coder)
            if gtype == 'T':
                # Add SBC file to the dataset if it exists
                try:
                    ds_SBC = xr.open_mfdataset(f'{sim_dir}/{file_pattern}'.replace('_T', '_SBC'), decode_times=time_coder)
                except(OSError):
                    pass
        # Loop over time indices to save memory
        # Fastest option is to load datasets after slicing but before merging
        for t in range(ds_nemo.sizes['time_counter']):
            print('...month '+str(t+1))
            # Sneaky selection of slice of size 1, to prevent dimension collapsing
            ds_tmp = ds_nemo.isel(time_counter=slice(t,t+1))
            ds_tmp.load()
            if dsV is not None:
                dsV_tmp = dsV.isel(time_counter=slice(t,t+1))
                dsV_tmp.load()
                ds_tmp = ds_tmp.merge(dsV_tmp)
            if ds_SBC is not None:
                ds_SBC_tmp = ds_SBC.isel(time_counter=slice(t,t+1))
                ds_SBC_tmp.load()
                ds_tmp = ds_tmp.merge(ds_SBC_tmp)
            precompute_timeseries(ds_tmp, timeseries_types, f'{timeseries_dir}/{timeseries_file}', halo=halo, periodic=periodic, domain_cfg=domain_cfg, name_remapping=name_remapping, nemo_mesh=nemo_mesh)
        ds_nemo.close()


# As above, but for PP output files from the UM atmosphere. 
def update_simulation_timeseries_um (suite_id, timeseries_types, timeseries_file='timeseries_um.nc', sim_dir='./', stream='p5'):

    update = os.path.isfile(sim_dir+timeseries_file)
    if update:
        # Timeseries file already exists
        # Get last time index
        ds_ts = xr.open_dataset(sim_dir+timeseries_file, decode_times=time_coder)
        time_last = ds_ts['time_centered'].data[-1]
        year_last = time_last.year
        month_last = time_last.month
        ds_ts.close()

    # Identify all the PP files for the given stream
    date_codes = []
    file_head = suite_id+'a.'+stream
    file_tail = '.pp'
    for f in os.listdir(sim_dir):
        if os.path.isdir(sim_dir+'/'+f):
            # Skip directories
            continue
        if not (f.startswith(file_head)) or not (f.endswith(file_tail)):
            # Not a UM output file for this stream
            continue
        # Extract date code (yyyymmm)
        date_code = f[len(file_head):len(file_head)+7]
        # Replace mmm abbreviation (eg jan) with numbers (eg 01) so we can sort and compare
        date_code = date_code[:4] + month_convert(date_code[4:])
        if update:
            # Need to check if date code has already been processed
            year = int(date_code[:4])
            month = int(date_code[4:6])
            if year < year_last or (year==year_last and month<=month_last):
                # Skip it
                continue
        date_codes.append(date_code)
    # Now sort alphabetically - i.e. by ascending date code
    date_codes.sort()
    
    # Loop through each date code, reconstruct the filename, and process
    for date_code in date_codes:
        fname = sim_dir + '/' + file_head + date_code[:4] + month_convert(date_code[4:]) + file_tail
        print('Processing '+fname)
        precompute_timeseries(fname, timeseries_types, sim_dir+'/'+timeseries_file, pp=True)
        
def calc_hovmoeller_region(var, region,
                           run_folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/output/reference-4.2.2/',
                           nemo_mesh='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20250715.nc'):
    
    # Load gridT files into dataset:
    gridT_files = glob.glob(f'{run_folder}*grid_T*')
    nemo_ds     = xr.open_mfdataset(gridT_files, decode_times=time_coder).isel(x_grid_T=region['x'], y_grid_T=region['y']) # load all the gridT files in the run folder

    nemo_mesh_ds     = xr.open_dataset(f'{nemo_mesh}', decode_times=time_coder)
    nemo_mesh_subset = nemo_mesh_ds.rename({'x':'x_grid_T','y':'y_grid_T','nav_lev':'deptht'}).isel(x_grid_T=region['x'], y_grid_T=region['y'], time_counter=0)
    
    var_ocean  = xr.where(nemo_mesh_subset.tmask==0, np.nan, nemo_ds[var]) 
    area_ocean = xr.where(nemo_mesh_subset.tmask==0, np.nan, nemo_ds['area_grid_T']) 
    region_var = (var_ocean*area_ocean).sum(dim=['x_grid_T','y_grid_T'])/(area_ocean.sum(dim=['x_grid_T','y_grid_T']))

    return region_var


# Check the given timeseries file for missing months (this can happen sometimes when files are missing in MASS). Fill the missing months with NaN.
def fix_missing_months (timeseries_file):

    import cftime

    ds = xr.open_dataset(timeseries_file, decode_times=time_coder)
    t_start = 0
    while True:
        time = ds['time_centered']
        for t in range(t_start, time.size-1):
            # Check if the next time index is 1 month ahead
            year_next, month_next = add_months(time[t].dt.year, time[t].dt.month, 1)
            if year_next != time[t+1].dt.year or month_next != time[t+1].dt.month:
                print('Missing data at '+str(year_next.data)+'-'+str(month_next.data).zfill(2))
                # Create a new time index with NaN data
                # First check data type - can't find a cleaner way to do this
                if isinstance(time[t].data.item(), cftime.Datetime360Day):
                    new_time = cftime.Datetime360Day(year=year_next, month=month_next, day=time[t].dt.day)
                else:
                    raise Exception('Time index uses data type '+type(time[t].data.item())+'; need to add this case to fix_missing_months()')
                ds_tmp = ds.isel(time_centered=t).where(False).copy(deep=True)
                ds_tmp.coords['time_centered'] = [new_time]                
                # Splice into existing data
                ds = xr.concat([ds.isel(time_centered=slice(0,t+1)), ds_tmp, ds.isel(time_centered=slice(t+1,None))], dim='time_centered')
                # Now start a new loop with the larger array, from where we left off
                t_start = t
                break
            if t == time.size-2:
                # Got to the end successfully; set a flag
                t_start = -1
        if t_start == -1:
            # All done
            break
    overwrite_file(ds, timeseries_file)


# Check the given timeseries for big chunks of NaNs (3 or more in a row), which could indicate a chunk of missing files on MASS. Check only the given variables (typically one per file type used to generate the timeseries, eg one from grid-T and one from isf-T).
def check_nans (timeseries_file, var_names=['all_massloss', 'all_bwtemp']):

    limit = 3

    ds = xr.open_dataset(timeseries_file, decode_times=time_coder)
    for var in var_names:
        if np.count_nonzero(ds[var].isnull()):
            # There are some NaNs; check for consecutive ones
            count = 0
            max_count = 0
            for t in range(ds.sizes['time_centered']):
                if ds[var].isel(time_centered=t).isnull():
                    print('Missing data at '+str(ds['time_centered'][t].dt.year.item())+'-'+str(ds['time_centered'][t].dt.month.item()))
                    count += 1
                    max_count = max(count, max_count)
                else:
                    count = 0
            if max_count >= limit:
                print('Problem with '+timeseries_file+': '+str(np.count_nonzero(ds[var].isnull()))+' NaNs, max block '+str(max_count))
                break
                    
            

    
            
        
    
                                

    
        
        
