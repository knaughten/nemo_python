###########################################################
# Generate forcing including atmospheric, runoff etc.
###########################################################
import xarray as xr
import numpy as np
import pandas as pd
import glob
from .utils import distance_btw_points, closest_point, convert_to_teos10, fix_lon_range, dewpoint_to_specific_humidity
from .grid import get_coast_mask, get_icefront_mask
from .ics_obcs import fill_ocean
from .interpolation import regrid_era5_to_cesm2, extend_into_mask, regrid_to_NEMO, neighbours
from .file_io import find_cesm2_file, find_processed_cesm2_file
from .constants import temp_C2K, rho_fw, cesm2_ensemble_members, sec_per_day, sec_per_hour

# Function subsets global forcing files from the same grid to the new domain, and fills any NaN values with connected 
# nearest neighbour and then fill_val.
# Inputs:
# file_path: string of location of forcing file
# nemo_mask: xarray Dataset of nemo meshmask (must contain tmask)
# fill_ocn: boolean to turn on (or off) the connected nearest neighbour fill
# fill_val: float or NaN value to fill any remaining NaN values with
# Returns an xarray dataset of the original forcing file subset to the new domain and filled
def subset_global(file_path, nemo_mask, fill_ocn=False, fill_val=0):

    # this subset is not generalized for other domains; can fix later
    ds = xr.open_dataset(f'{file_path}').isel(y=slice(0,453)) 

    if fill_ocn:
        for var in list(ds.keys()):
            # Check for values that are NaN and in the ocean and fill with nearest neighbour
            ds_filled = fill_ocean(ds.isel(time_counter=0), var, nemo_mask, dim='2D', fill_val=fill_val)
            ds[var]   = ds[var].dims, ds_filled[var].values[np.newaxis, ...]   

    for var in list(ds.keys()):
        # Then fill any NaN values with fill_val
        ds[var] = xr.where(np.isnan(ds[var]), fill_val, ds[var])
    
    new_file_path = file_path.replace(file_path.split('/')[-1], f"AntArc_{file_path.split('/')[-1]}") 

    # save file with time_counter as unlimited dimension, if time_counter is present
    if 'time_counter' in ds.dims:
        ds.to_netcdf(f"{new_file_path}", unlimited_dims='time_counter')
    else:
        ds.to_netcdf(f"{new_file_path}")
    
    return ds

# Function identifies locations where calving does not occur at the ocean edge of the iceshelf front 
# Inputs:
# calving: xarray variable of calving proportions (2D)
# nemo_mask: xarray Dataset of nemo meshmask (must contain tmask and tmaskutil)
# Returns three 2D arrays of calving that occurs not at the icefront, but further in the ocean (calving_ocn), 
# on land (calving_land), or on an ice shelf grid point (calving_ice)
def calving_at_coastline(calving, nemo_mask):
    
    calving   = xr.where(calving > 0, calving, np.nan)

    # Boolean arrays to identify regions:
    ocean     = (nemo_mask.tmask.isel(time_counter=0, nav_lev=0) == 1)
    iceshelf  = (nemo_mask.tmaskutil.isel(time_counter=0) - nemo_mask.tmask.isel(time_counter=0, nav_lev=0)).astype(bool);
    land      = (nemo_mask.tmaskutil.isel(time_counter=0) == 0)

    # Cases where calving does not occur at the ocean edge of the icefront:
    icefront_mask_ocn = get_icefront_mask(nemo_mask, side='ocean')
    calving_ocn  = xr.where((~icefront_mask_ocn & ocean), calving, np.nan)  # calving occurs in the ocean but not right by the icefront
    calving_land = xr.where(land      , calving, np.nan)  # calving occurs on land
    calving_ice  = xr.where(iceshelf  , calving, np.nan)  # calving occurs on ice shelf

    return calving_ocn, calving_land, calving_ice

# Function shifts the x,y location of a calving point to the nearest iceshelf front ocean point 
# Inputs:
# calving: xarray dataset containing calving proportions 
# mask: locations of cells that need to be shifted
# nemo_mask: xarray dataset of nemo mesh mask file (must contain nav_lon, nav_lat)
# icefront_mask_ocn: mask of ocean points nearest to iceshelf, produced by get_icefront_mask
# max_distance: float of maximum distance (in meters) that an ocean calving point will get moved
# calving_var: string of the calving variable name
def shift_calving(calving, mask, nemo_mask, icefront_mask_ocn, max_distance=2e6, calving_var='soicbclv', ocn=False):
    # NEMO domain grid points
    x, y        = np.meshgrid(nemo_mask.nav_lon.x, nemo_mask.nav_lon.y)
    calving_x   = x[(~np.isnan(mask))]
    calving_y   = y[(~np.isnan(mask))]
    calving_new = np.copy(calving[calving_var].values);

    ice_shelf  = (nemo_mask.tmaskutil.isel(time_counter=0) - nemo_mask.tmask.isel(time_counter=0, nav_lev=0)).astype(bool)
    open_ocean = (nemo_mask.tmask.isel(time_counter=0, nav_lev=0) == 1)
    land       = ~open_ocean
    # Return ocean points with at least 3 ice shelf neighbour
    num_ice_shelf_neighbours = neighbours(ice_shelf, missing_val=0)[-1]
    num_land_neighbours      = neighbours(land, missing_val=0)[-1] # also includes the iceshelf points
    confined_points          = (open_ocean*(num_land_neighbours > 2)).astype(bool)

    # Coordinates of ocean points closest to iceshelf front 
    # increase number of possible points by shifting the icefront mask:
    ocean_neighbours       = neighbours(icefront_mask_ocn, missing_val=0)[-1]
    icefront_mask_extended = (((icefront_mask_ocn)+(ocean_neighbours))*open_ocean).astype(bool)
    icefront_x             = x[icefront_mask_extended]
    icefront_y             = y[icefront_mask_extended]
    icefront_coord         = (nemo_mask.nav_lon.values[icefront_mask_extended], nemo_mask.nav_lat.values[icefront_mask_extended])

    # For each land iceberg calving point, check distance to nearest iceshelf front and move closer if possible:
    for index in list(zip(calving_y, calving_x)):
     
        calving_coord = (nemo_mask.nav_lon.values[index], nemo_mask.nav_lat.values[index])
        distances     = distance_btw_points(calving_coord, icefront_coord)
        distances[distances < 2e4] = np.nan
        # only move cell if it is within a certain distance
        if np.nanmin(np.abs(distances)) < max_distance:     
            new_x         = icefront_x[np.nanargmin(np.abs(distances))]
            new_y         = icefront_y[np.nanargmin(np.abs(distances))]
            # Move calving to the nearest icefront point and add to any pre-existing calving at that point
            calving_new[(new_y, new_x)] = calving_new[(new_y, new_x)] + calving_new[index]    
            calving_new[index]          = 0 # remove calving from originating point

    # For each ocean iceberg calving point, check that it isn't surrounded on three sides by iceshelf points or land points
    # to prevent accumulation of icebergs in small coastal regions
    if ocn:
        # Points that are ocean that have at least three iceshelf neighbour points, these are the ones that I'll need to move
        confined_x = x[confined_points*icefront_mask_ocn]
        confined_y = y[confined_points*icefront_mask_ocn]
        unconfined_x = x[~confined_points*open_ocean]
        unconfined_y = y[~confined_points*open_ocean]
        unconfined_coord = (nemo_mask.nav_lon.values[~confined_points*open_ocean], nemo_mask.nav_lat.values[~confined_points*open_ocean])
        
        for ind in list(zip(confined_y, confined_x)):
            confined_coord = (nemo_mask.nav_lon.values[ind], nemo_mask.nav_lat.values[ind])
            distances      = distance_btw_points(confined_coord, unconfined_coord)#icefront_coord)
            distances[distances < 2e4] = np.nan

            new_x = unconfined_x[np.nanargmin(np.abs(distances))]
            new_y = unconfined_y[np.nanargmin(np.abs(distances))]
            # Move calving to the nearest icefront point and add to any pre-existing calving at that point
            calving_new[new_y, new_x] = calving_new[new_y, new_x] + calving_new[ind]    
            calving_new[ind]        = 0 # remove calving from originating point

    # Write new locations to xarray dataset
    calving_ds = calving.copy()
    calving_ds[calving_var] = calving[calving_var].dims, calving_new

    return calving_ds

# Main function to move pre-existing calving dataset to a new coastline (on the same underlying grid, but subset)
# Inputs:
# nemo_mask: xarray dataset of nemo meshmask file (must contain nav_lon, nav_lat, tmask, tmaskutil)
# calving: xarray dataset of old calving forcing NetCDF file
# calving_var: string of the calving variable name
# new_file_path: string of name and location of new calving forcing file
# Returns: xarray dataset with calving at new locations
def create_calving(calving, nemo_mask, calving_var='soicbclv', new_file_path='./new-calving.nc'):

    # Identify locations with calving that need to be moved
    calving_ocn, calving_land, calving_ice = calving_at_coastline(calving[calving_var], nemo_mask)

    # Mask of ocean grid points nearest to icefront, to move calving to
    icefront_mask_ocn = get_icefront_mask(nemo_mask, side='ocean')
    icefront_mask_ice = get_icefront_mask(nemo_mask, side='ice')

    # Shift calving points to the iceshelf edge
    calv_ocn_new  = shift_calving(calving       , calving_ocn , nemo_mask, icefront_mask_ocn, ocn=True) # from ocean
    calv_land_new = shift_calving(calv_ocn_new  , calving_land, nemo_mask, icefront_mask_ocn, ocn=False) # from land
    calv_ice_new  = shift_calving(calv_land_new , calving_ice , nemo_mask, icefront_mask_ocn, ocn=False) # from ice

    # Check if the icebergs calve in regions shallower than the minimum initial iceberg thickness (40 m)
    calving_depth    = nemo_mask.bathy_metry.squeeze().values*(calv_ice_new[calving_var].squeeze().values.astype(bool))
    if np.sum((calving_depth < 40)*(calving_depth > 0)) != 0:
        print('Warning: number of cells with calving shallower than minimum iceberg thickness is, ', np.sum((calving_depth < 40)*(calving_depth > 0)))

    # Write the calving dataset to a netcdf file:
    calv_ice_new[calving_var] = ('time_counter',) + calving[calving_var].dims, calv_ice_new[calving_var].values[np.newaxis, ...] 
    calv_ice_new.to_netcdf(f"{new_file_path}", unlimited_dims='time_counter')

    # Check that the amount of calving that occurs in the new file is approximately equal to the original file:
    # allow for a tolerance of 0.1% (although it should be essentially equal within machine precision)
    tolerance = 0.001*(np.sum(calv_ice_new[calving_var].values)) 
    if np.abs(np.sum(calv_ice_new[calving_var].values) - np.sum(calving[calving_var].values)) >= tolerance:
        raise Exception('The total amount of calving in the new file is not equal to the original total', \
                np.sum(calv_ice_new[calving_var].values), np.sum(calving[calving_var].values))
    
    return calv_ice_new

# Process ocean conditions from CESM2 scenarios for a single variable and single ensemble member (for initial and boundary conditions).
def cesm2_ocn_forcing (expt, var, ens, out_dir, start_year=1850, end_year=2100):

    if expt not in ['LE2']:
        raise Exception(f'Invalid experiment {expt}')

    freq = 'monthly'
    for year in range(start_year, end_year+1):
        # read in the data and subset to the specified year
        if var in ['aice','sithick','sisnthick']:
            if year > 1850:
                file_path_prev = find_cesm2_file(expt, var, 'ice', freq, ens, year-1)
            file_path = find_cesm2_file(expt, var, 'ice', freq, ens, year)
        else:
            if year > 1850:
                file_path_prev = find_cesm2_file(expt, var, 'ocn', freq, ens, year-1) # load last month from previous file
            file_path = find_cesm2_file(expt, var, 'ocn', freq, ens, year)
        
        if year==1850:
            ds = xr.open_dataset(file_path)
        else:
            if file_path_prev != file_path:
                ds = xr.open_mfdataset([file_path_prev, file_path])
            else:
                ds = xr.open_dataset(file_path)
        data = ds[var].isel(time=(ds.time.dt.year == year))

        # Unit conversions ### need to check that these are still consistent between CESM1 and CESM2
        if var in ['SSH','UVEL','VVEL']:
            # convert length units from cm to m
            data *= 0.01 
        # Convert from practical salinity to absolute salinity??
        elif var == 'TEMP':
            # Convert from potential temperature to conservative temperature
            # need to load absolute salinity from file
            salt_path = find_cesm2_file(expt, 'SALT', 'ocn', freq, ens, year)
            salt_ds   = xr.open_dataset(salt_path)
            salt_data = salt_ds['SALT'].isel(time=(salt_ds.time.dt.year == year))

            dsc  = xr.Dataset({'PotTemp':data, 'AbsSal':salt_data, 'depth':data.z_t, 'lon':data.TLONG, 'lat':data.TLAT})
            data = convert_to_teos10(dsc, var='PotTemp')

        # Convert calendar to Gregorian
        data = data.convert_calendar('gregorian')
        # Convert longitude range from 0-360 to degrees east
        if var in ['SSH','SALT','TEMP']:
            lon_name = 'TLONG'
        elif var in ['aice','sithick','sisnthick']:
            lon_name = 'TLON'
        elif var in ['UVEL','VVEL']:
            lon_name = 'ULONG'
        data[lon_name] = fix_lon_range(data[lon_name])
        
        # Convert depth (z_t) from cm to m
        if var in ['SALT','TEMP','UVEL','VVEL']:
            data['z_t'] = data['z_t']*0.01
            data['z_t'].attrs['units'] = 'meters'
            data['z_t'].attrs['long_name'] = 'depth from surface to midpoint of layer'
  
        # Mask sea ice conditions based on tmask (so that land is NaN and no ice areas are zero)
        if var in ['sithick','sisnthick']:
            data = data.fillna(0)
            data = data.where((ds.isel(time=(ds.time.dt.year == year)).tmask.fillna(0).values) != 0)

        # Change variable names and units in the dataset:
        if var=='TEMP':
            varname = 'ConsTemp'
            data.attrs['long_name'] ='Conservative Temperature'
        elif var=='SALT':
            varname = 'AbsSal'
            data = data.rename(varname)
            data.attrs['long_name'] ='Absolute Salinity'
        else:
            varname=var
            if var=='SSH':
                data.attrs['units'] = 'meters'
            elif var in ['UVEL','VVEL']:
                data.attrs['units'] = 'm/s'

        # Write data
        out_file_name = f'{out_dir}CESM2-{expt}_ens{ens}_{varname}_y{year}.nc'
        data.to_netcdf(out_file_name, unlimited_dims='time')

    return

# Convert wind velocities from the lowest atmospheric model level grid cell to the 10 m wind level 
# (CESM2 output UBOT and VBOT is at 992 hPa) by using the wind speed magnitude from the variable U10 and the wind directions from UBOT and VBOT
def UBOT_to_U10_wind(UBOT, VBOT, U10):

    # calculate the angle between the UBOT and VBOT wind vectors
    theta = np.arctan2(VBOT, UBOT) 
    # then use this angle to create the x and y wind vector components that sum to the magnitude of the U10 wind speed
    U10x = U10*np.cos(theta)
    U10y = U10*np.sin(theta)

    return U10x, U10y

# Get CESM2 wind velocities from the wind speed at 10 meters and the direction from the surface stress 
# (because wind speed vectors are not available for the CESM2 single forcing experiments at daily frequency)
def TAU_to_U10xy(TAUX, TAUY, U10):

    # calculate the angle between the surface stress vectors (need to take opposite direction because of definition difference)
    theta = np.arctan2(-1*TAUY, -1*TAUX)
    # then use this angle to create the x and y wind vector components that sum to the magnitude of the U10 wind speed
    U10x = U10*np.cos(theta)
    U10y = U10*np.sin(theta)

    return U10x, U10y

# Process atmospheric forcing from CESM2 scenarios (LE2, etc.) for a single variable and single ensemble member.
# expt='LE2', var='PRECT', ens='1011.001' etc.
def cesm2_atm_forcing (expt, var, ens, out_dir, start_year=1850, end_year=2100, year_ens_start=1750, freq='daily', shift_wind=False):

    if expt not in ['LE2', 'piControl', 'SF-xAER', 'SF-AAER', 'SF-BMB', 'SF-GHG', 'SF-EE']:
        raise Exception('Invalid experiment {expt}')

    # for each year, read in the data, process it, and save the processed forcing to a file
    for year in range(start_year,  end_year+1):
        
        # helper function to load CESM2 file and select variable for the specified year
        def load_cesm2_file(var, expt=expt, freq=freq, ens=ens, year=year):
            file_path = find_cesm2_file(expt, var, 'atm', freq, ens, year)
            ds_vel    = xr.open_dataset(file_path)
            data_vel  = ds_vel[var].isel(time=(ds_vel.time.dt.year == year))
            return data_vel

        # load cesm2 land-ocean mask
        land_mask  = find_cesm2_file(expt, 'LANDFRAC', 'atm', 'monthly', ens, year)
        cesm2_mask = xr.open_dataset(land_mask).LANDFRAC.isel(time=0)
        # read in variables from forcing files
        if var=='PRECS': # snowfall
            if expt=='piControl': freq='monthly' # only monthly files available
            data_conv  = load_cesm2_file('PRECSC') # convective snow rate
            data_large = load_cesm2_file('PRECSL') # large-scale snow rate
        elif var=='U10':
            if expt=='piControl': var_U = 'U'; var_V='V';
            else: var_U='UBOT'; var_V='VBOT';

            data_U10  = load_cesm2_file('U10')
            data_UBOT = load_cesm2_file(var_U)
            data_VBOT = load_cesm2_file(var_V)

            # For the piControl experiment, only full column winds are available, so select the bottom wind
            if expt=='piControl':
                data_UBOT = data_UBOT.isel(lev=-1) # bottom wind is the last entry (992 hPa)
                data_VBOT = data_VBOT.isel(lev=-1)

            # Convert the wind to the 10 m wind (corrected height)
            if shift_wind:
                Ux, Uy    = UBOT_to_U10_wind(data_UBOT, data_VBOT, data_U10)
                data_UBOT = Ux.rename('U10x')
                data_VBOT = Uy.rename('U10y')
        else:
            data = load_cesm2_file(var)
 
        # Unit conversions #
        # notes: don't think I need to swap FSDS,FLDS signs like Kaitlin did for CESM1, qrefht is specific 
        #        humidity so don't need to convert, but specify read in option in namelist_cfg
        if var=='PRECT': # total precipitation
            # Convert from m/s to kg/m2/s
            data *= rho_fw
        elif var=='PRECS': # snowfall
            # Combine convective and large scale snowfall rates and convert from m of water equivalent to kg/m2/s
            data  = (data_conv + data_large) * rho_fw        
            data  = data.rename('PRECS')

        if var=='U10':
            data_arrays = [data_UBOT, data_VBOT]
        else:
            data_arrays = [data]       

        for arr in data_arrays:
            if var in ['QREFHT','TREFHT','FSDS','FLDS','PSL','PRECT','PRECS','UBOT','VBOT']: 
                # Mask atmospheric forcing over land based on cesm2 land mask (since land values might not be representative for the ocean areas)
                arr = xr.where(cesm2_mask.values != 0, -9999, arr)
                # And then fill masked areas with nearest non-NaN latitude neighbour
                print(f'Filling land for variable {var} in year {year}')
                var_filled_array = np.empty(arr.shape)
                for tind, t in enumerate(arr.time):
                    var_filled_array[tind,:,:] = extend_into_mask(arr.isel(time=tind).values, missing_val=-9999, fill_val=np.nan, use_2d=True, use_3d=False, num_iters=100)
                   
                arr.data = var_filled_array 
     
            # Convert longitude range from (0,360) to (-180,180) degrees east
            arr['lon'] = fix_lon_range(arr['lon'])
            # CESM2 does not do leap years, but NEMO does, so fill 02-29 with 02-28        
            # Also convert calendar to Gregorian
            fill_value = arr.isel(time=((arr.time.dt.month==2)*(arr.time.dt.day==28)))
            if freq=='daily':
                arr = arr.convert_calendar('gregorian', dim='time', missing=fill_value)
            elif freq=='3-hourly': # need to do something slightly different because you need to fill a slice of times
                arr = arr.convert_calendar('gregorian', dim='time', missing=fill_value.isel(time=0))
                #if leap year, properly replace the time slice:
                if sum((arr.time.dt.month==2)*(arr.time.dt.day==29)) > 0:
                    print('leap year')
                    arr.loc[dict(time=(arr.time.dt.month==2)*(arr.time.dt.day==29))] = fill_value.data

            # Change variable names and units in the dataset:
            varname = arr.name 
            if var=='PRECS':
                arr.attrs['long_name'] ='Total snowfall (convective + large-scale)'
                arr.attrs['units'] = 'kg/m2/s'
            elif var=='PRECT':
                arr.attrs['units'] = 'kg/m2/s'
            elif var=='wind':
                if varname=='U10x':
                    arr.attrs['long_name'] = 'zonal wind at 10 m'
                elif varname=='U10y':
                    arr.attrs['long_name'] = 'meridional wind at 10 m'

            # Write data
            out_file_name = f'{out_dir}CESM2-{expt}_ens{ens}_{freq}_{varname}_y{year}.nc'
            arr.to_netcdf(out_file_name, unlimited_dims='time')
    return arr


# Create CESM2 atmospheric forcing for the given scenario, for all variables and ensemble members.
# ens_strs : list of strings of ensemble member names
def cesm2_expt_all_atm_forcing (expt, ens_strs=None, out_dir=None, start_year=1850, end_year=2100, year_ens_start=1750, shift_wind=False):
    
    if out_dir is None:
        raise Exception('Please specify an output directory via optional argument out_dir')

    var_names = ['UBOT','VBOT','FSDS','FLDS','TREFHT','QREFHT','PRECT','PSL','PRECS'] 
    for ens in ens_strs:
        print(f'Processing ensemble member {ens}')
        out_dir = f'{out_dir}ens{ens}/'
        for var in var_names:
            print(f'Processing {var}')
            # specify the forcing frequency to read in
            if var in ['UBOT', 'VBOT']:
                freq='3-hourly'
            else:
                freq='daily'
            cesm2_atm_forcing(expt, var, ens, out_dir, start_year=start_year, end_year=end_year, freq=freq, year_ens_start=year_ens_start, shift_wind=shift_wind)

    return

# Create CESM2 ocean forcing for the given scenario, for all variables and ensemble members.
# ens_strs : list of strings of ensemble member names
def cesm2_expt_all_ocn_forcing(expt, ens_strs=None, out_dir=None, start_year=1850, end_year=2100):

    if out_dir is None:
        raise Exception('Please specify an output directory via optional argument out_dir')

    ocn_var_names = ['TEMP','SALT','UVEL','VVEL','SSH']
    ice_var_names = ['aice','sithick','sisnthick']
    var_names = ocn_var_names + ice_var_names
 
    for ens in ens_strs:
        print(f'Processing ensemble member {ens}')
        for var in var_names:
            print(f'Processing {var}')
            cesm2_ocn_forcing(expt, var, ens, out_dir, start_year=start_year, end_year=end_year)

    return


# Helper function calculates the monthly time-mean over specified year range for ERA5 output (for bias correction)
# Note that the year 1996 is excluded from climatology due to an issue with a cyclone in the Amundsen Sea in ERA5
# Input: 
# - variable : string of forcing variable name (in ERA5 naming convention)
# - (optional) year_start : start year for time averaging
# - (optional) end_year   : end year for time averaging
def era5_time_mean_forcing(variable, year_start=1979, year_end=2024, freq='daily', lat_slice=slice(-90,-50),
                           era5_folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/ERA5-forcing/'):

    if freq=='3-hourly':
        era5_folder_in = f'{era5_folder}hourly/processed/'
    else:
        era5_folder_in = f'{era5_folder}{freq}/processed/'
        
    era5_folder_out = f'{era5_folder}climatology/' 
    if variable=='sph2m':
        varname='specific_humidity'
    else:
        varname=variable

    if freq=='daily' or freq=='hourly':
        if variable in ['wind_speed', 'wind_angle']:
            era5_u  = xr.open_mfdataset(f'{era5_folder_in}u10_*')['u10'].sortby('lat').sel(lat=lat_slice)
            era5_v  = xr.open_mfdataset(f'{era5_folder_in}v10_*')['v10'].sortby('lat').sel(lat=lat_slice)
            era5_u  = era5_u.isel(time=((era5_u.time.dt.year <= year_end)*(era5_u.time.dt.year >= year_start)*(era5_u.time.dt.year != 1996)))
            era5_v  = era5_v.isel(time=((era5_v.time.dt.year <= year_end)*(era5_v.time.dt.year >= year_start)*(era5_v.time.dt.year != 1996)))
            if variable == 'wind_speed':
                era5_ds = np.hypot(era5_u, era5_v).rename(variable).to_dataset()
            elif variable == 'wind_angle':
                era5_ds = np.arctan2(era5_v, era5_u).rename(variable).to_dataset()
        else:
            era5_ds = xr.open_mfdataset(f'{era5_folder_in}{variable}_*.nc')[varname].sortby('lat').sel(lat=lat_slice)
            era5_ds = era5_ds.isel(time=((era5_ds.time.dt.year <= year_end)*(era5_ds.time.dt.year >= year_start)*(era5_ds.time.dt.year != 1996)))

        time_mean = era5_ds.groupby('time.month').mean(dim='time')

    elif freq=='3-hourly':
        # slow due to resampling if you use the same approach as for daily, so do the below instead
        for year in range(year_start, year_end+1):
            if year == 1996:
                print('skipping year 1996')
            else:
                print(year)
                if variable in ['wind_speed', 'wind_angle']:
                    era5_u = xr.open_dataset(f'{era5_folder_in}u10_y{year}.nc')['u10'].sortby('lat').sel(lat=lat_slice)
                    era5_v = xr.open_dataset(f'{era5_folder_in}v10_y{year}.nc')['v10'].sortby('lat').sel(lat=lat_slice)
                    era5_u = era5_u.resample(time='3h').mean()
                    era5_v = era5_v.resample(time='3h').mean()               
                    if variable == 'wind_speed':
                        era5_ds = np.hypot(era5_u, era5_v).rename(variable).groupby('time.month').mean(dim='time').to_dataset()
                    elif variable == 'wind_angle':
                        era5_ds = np.arctan2(era5_v, era5_u).rename(variable).groupby('time.month').mean(dim='time').to_dataset()
                else:
                    era5_ds = xr.open_dataset(f'{era5_folder_in}{variable}_y{year}.nc')[varname].sortby('lat').sel(lat=lat_slice)
                    era5_ds = era5_ds.resample(time='3h').mean().groupby('time.month').to_dataset()
               
                era5_ds.to_netcdf(f'{era5_folder_out}ERA5_{variable}_{freq}_monthly_mean_y{year}.nc')
 
        time_mean = xr.open_mfdataset(f'{era5_folder_out}ERA5_{variable}_{freq}_monthly_mean_y*', concat_dim='year', combine='nested')[varname].mean(dim='year')
    
    time_mean.to_netcdf(f'{era5_folder_out}ERA5_{variable}_{freq}_{year_start}-{year_end}_mean_monthly.nc')

    return 


# Function calculates the monthly time-mean over specified year range for mean of all CESM2 ensemble members in the specified experiment (for bias correction)
# Input:
# - expt : string of CESM2 experiment name (e.g. 'LE2')
# - variable : string of forcing variable name
# - out_dir : directory to write ensemble mean files to
# - (optional) year_start : start year for time averaging
# - (optional) end_year   : end year for time averaging
# - (optional) ensemble_members : list of strings of ensemble members to average (defaults to all the ones that have been downloaded)
def cesm2_ensemble_time_mean_forcing(expt, variable, out_dir, year_start=1979, year_end=2024, ensemble_members=cesm2_ensemble_members, freq='daily', lat_slice=slice(-90,-50)):

    print(f'Calculating ensemble mean for variable {variable} at frequency {freq} from {year_start}-{year_end} for {expt} ensemble members:', ensemble_members)

    # helper function to read in processed CESM2 file for all ensemble members, and subset domain
    def load_cesm2_ensemble(variable, year, expt=expt, freq=freq, ensemble_members=ensemble_members):
        for e, ens in enumerate(ensemble_members):
            cesm2_file = find_processed_cesm2_file(expt, variable, ens, year, freq=freq)
            cesm2_ds   = xr.open_dataset(cesm2_file)
            if e==0:
                cesm2_ds_ens = cesm2_ds.copy().expand_dims(dim='ens')
            else:
                cesm2_ds_ens = xr.concat([cesm2_ds_ens, cesm2_ds], dim='ens')

        cesm2_ds_ens = cesm2_ds_ens.sortby('lat').sel(lat=lat_slice)

        return cesm2_ds_ens

    # For each year within the specified range, calculate the monthly ensemble mean and write to file
    for year in range(year_start, year_end+1):
        print(year)
        if variable in ['wind_speed', 'wind_angle']:
            cesm2_u_ens = load_cesm2_ensemble('UBOT', year)
            cesm2_v_ens = load_cesm2_ensemble('VBOT', year)
            if variable == 'wind_speed':
                cesm2_ds_ens = np.hypot(cesm2_u_ens.UBOT, cesm2_v_ens.VBOT).rename(variable)
            elif variable == 'wind_angle':
                cesm2_ds_ens = np.arctan2(cesm2_v_ens.VBOT, cesm2_u_ens.UBOT).rename(variable)
        else:
            cesm2_ds_ens = load_cesm2_ensemble(variable, year)      

        cesm2_ds_mean = cesm2_ds_ens.mean(dim='ens').groupby("time.month").mean(dim="time")
        cesm2_ds_mean['lon'] = fix_lon_range(cesm2_ds_mean['lon'])
        cesm2_ds_mean = cesm2_ds_mean.sortby('lon')
        cesm2_ds_mean.to_netcdf(f'{out_dir}CESM2-LE2_{variable}_{freq}_y{year}_ensemble_mean_monthly.nc')

    # Next, read in the above ensemble mean files for all years and calculate the mean over the full timeseries
    file_list = [f'{out_dir}CESM2-LE2_{variable}_{freq}_y{year}_ensemble_mean_monthly.nc' for year in range(year_start, year_end+1)]
    ds_ens = xr.open_mfdataset(file_list, concat_dim='year', combine='nested')[variable]
    ds_ens.mean(dim='year').to_netcdf(f'{out_dir}CESM2-LE2_{variable}_{freq}_ensemble_{year_start}-{year_end}_mean_monthly.nc')            

    return

# wrapper function to calculate ensemble mean for each of the atmospheric variables we're interested in:
def cesm2_all_variables_ensemble_mean(expt, out_dir, year_start=1979, year_end=2024, ensemble_members=cesm2_ensemble_members):

    for variable in ['TREFHT','QREFHT','FSDS','FLDS','PRECT','PRECS', 'PSL', 'wind_speed', 'wind_angle']: 
        if variable in ['wind_speed','wind_angle']:
            freq='3-hourly'
        else:
            freq='daily'
        cesm2_ensemble_time_mean_forcing(expt, variable, out_dir, year_start=year_start, year_end=year_end, ensemble_members=ensemble_members, freq=freq)

    return

# Function calculate the bias correction for the atmospheric variable from the specified source type based on 
# the difference between its mean state and the ERA5 mean state. Reads in fields that were already regridded with the NEMO WEIGHTS tool
# ex. CESM2 vs. ERA5. Assumes you've ran the functions: cesm2_all_variables_ensemble_mean() to calculate the ensemble mean of CESM2 variables
# Input:
# - source : string of source type (currently only set up for 'CESM2') 
# - variable : string of the variable from the source dataset to be corrected
# - (optional) expt : cesm2 experiment set
# - (optional) year_start : start year for time averaging
# - (optional) end_year   : end year for time averaging
# - (optional) fill_land  : boolean whether or not to fill grid cells that are land in CESM2 with ocean values along lines of latitudes
# - (optional) freq       : frequency of underlying forcing files (daily, 3-hourly, or hourly)
# - (optional) monthly    : boolean whether to do monthly bias correction or one mean field without seasonality
# - (optional) era5_folder, source_folder : string paths to era5 forcing climatology and source to correct (CESM2) forcing climatology/ensemble mean
# - (optional) out_folder : string to location to save the bias correction file
def calc_bias_correction(source, variable, expt='LE2', year_start=1979, year_end=2024, freq='daily', fill_land=False, monthly=False,
                         era5_folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/ERA5-forcing/climatology/',
                         source_folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/climate-forcing/CESM2/LE2/ensemble_mean/',
                         out_folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/climate-forcing/CESM2/LE2/ensemble_mean/bias_corr/'):

    if source=='CESM2':
        
        # Load regridded CESM2 ensemble mean
        CESM2_ensemble_mean = xr.open_dataset(f'{source_folder}CESM2-{expt}_eANT025_{variable}_{freq}_ensemble_{year_start}-{year_end}_mean_monthly.nc')

        # Load regridded ERA5 climatology
        CESM2_to_ERA5_varnames = {'TREFHT':'t2m','FSDS':'msdwswrf','FLDS':'msdwlwrf','QREFHT':'sph2m', 'PRECS':'msr', 'PRECT':'mtpr', \
                                  'PSL':'msl', 'wind_speed':'wind_speed', 'wind_angle':'wind_angle'}
        filevarname = CESM2_to_ERA5_varnames[variable]
        if variable=='QREFHT':
           varname='specific_humidity' # file with dewpoint temperature converted to specific humidity
        else:
           varname=filevarname
        ERA5_climatology = xr.open_dataset(f'{era5_folder}ERA5_eANT025_{filevarname}_{freq}_{year_start}-{year_end}_mean_monthly.nc').rename({varname:variable})
 
        # thermo(dynamic) correction
        if variable in ['TREFHT','QREFHT','FLDS','FSDS','PRECS','PRECT','wind_speed','wind_angle','PSL']:
            if monthly:
                out_file = f'{out_folder}{source}-{expt}_{variable}_{freq}_bias_corr_monthly.nc'
            else:
                CESM2_ensemble_mean = CESM2_ensemble_mean.mean(dim='month')
                ERA5_climatology    = ERA5_climatology.mean(dim='month')
                out_file = f'{out_folder}{source}-{expt}_{variable}_{freq}_bias_corr.nc'
            
            # Call (thermo)dynamic correction:
            thermo_dynamic_correction(CESM2_ensemble_mean, ERA5_climatology, variable, out_file, fill_land=fill_land, monthly=monthly)
        else:
            raise Exception(f'Variable {variable} does not need bias correction. Is this true?')
    else:
        raise Exception("Bias correction currently only set up to correct CESM2. Sorry you'll need to write some more code!")

    return

# Function to calculate the bias and associated (thermo)dynamic correction between source (CESM2) and ERA5 mean fields
# Inputs:
# - source_mean : xarray Dataset containing the ensemble and time mean of the source dataset variable (currently CESM2)
# - ERA5_mean   : xarray Dataset containing the time mean of the ERA5 variable
# - out_file    : string to path to write NetCDF file to
# - (optional) fill_land  : boolean indicating whether to fill areas that are land in the source mask with nearest values or whether to just leave it as is
# - (optional) monthly    : boolean indicating whether bias correction fields are monthly or not
# - (optional) dist_coast : 
# - (optional) coastal_correction_limit : float indicating the distance from the coastline (in km) considered for the coastal wind angle correction
def thermo_dynamic_correction(source_mean, ERA5_mean, variable, out_file, fill_land=True, monthly=False, 
                              dist_coast='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/distance_coast-20250715.nc', coastal_correction_limit=400):
    if variable=='wind_speed':
        print('Correcting dynamics')
        bias = ERA5_mean / source_mean
    elif variable=='wind_angle':
        print('Correcting dynamics') 
        # open file with distance from coastline for the configuration
        distcoast = xr.open_dataset(dist_coast)
        # apply correction to distance within coastal_correction_limit (excluding south america)
        region_to_correct = (distcoast.distance_coast < coastal_correction_limit)*(distcoast.nav_lat <-59)
        angle_bias = ERA5_mean - source_mean
        angle_bias = xr.where(region_to_correct, angle_bias, 0)
        angle_bias = xr.where(distcoast.distance_coast ==0, 0, angle_bias) # mask land
        # apply a cosine taper to the angle bias to gradually transition from strength 1 to 0
        bias = xr.where(region_to_correct, angle_bias*np.cos((np.pi/2)*(distcoast.distance_coast/coastal_correction_limit)), 0)
    else:
        print('Correcting thermodynamics')
        # Calculate difference:
        bias = ERA5_mean - source_mean
    # Fill land regions along latitudes
    if fill_land:
       # bias = bias.interpolate_na(dim='lat', method='nearest', fill_value="extrapolate")
       src_to_fill = xr.where(np.isnan(bias), -9999, bias) # which cells need to be filled
       var_filled  = extend_into_mask(src_to_fill[variable].values, missing_val=-9999, fill_val=np.nan, use_2d=True, use_3d=False, num_iters=100)
       if monthly:
           bias[variable] = (('time','lat','lon'), var_filled)
       else:
           bias[variable] = (('lat','lon'), var_filled)

    # write to file
    if monthly:
        bias.to_netcdf(out_file, unlimited_dims='month')
    else:
        bias.to_netcdf(out_file)
    
    return

# Main function to apply bias correction files produced from calc_bias_correction to the source files
# Inputs:
# - variable : string of variable name
# - ens      : ensemble member to bias correct
# - (optional) expt       : string specifying CESM2 experiment to correct
# - (optional) start_year : integer start year for files to process
# - (optional) end_year   : integer end year for files to process
# - (optional) monthly    : boolean specifying whether to do monthly bias correction or otherwise a constant spatial field
# - (optional) freq       : string specifying underlying forcing frequency from which bias correction was derived (hourly, 3-hourly, or hourly)
# - (optional) highres    : boolean whether the files to be read in and bias corrected are on the eANT025 (high resolution) grid
# - (optional) out_dir    : string path where to save bias corrected forcing files
# - (optional) bias_dir   : string path to bias correction field files produced by the function calc_bias_correction
def apply_bias_correction(variable, ens, expt='LE2', start_year=1900, end_year=2050, monthly=True, freq='daily', highres=True,
                          out_dir='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/climate-forcing/CESM2/LE2/bias-corrected/',
                          bias_dir='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/climate-forcing/CESM2/LE2/ensemble_mean/bias_corr/'):
    from tqdm import tqdm
    print(f'Processing {variable} files for ensemble member {ens} from {start_year}-{end_year} with bias correction file from {freq} means')

    out_dir = f'{out_dir}ens{ens}/'
    bias_ds = xr.open_dataset(f'{bias_dir}CESM2-LE2_{variable}_{freq}_bias_corr_monthly.nc')[variable]
    if variable=='wind_speed': # also load wind angle bias correction file
        bias_ds_angle = xr.open_dataset(f'{bias_dir}CESM2-LE2_wind_angle_{freq}_bias_corr_monthly.nc').wind_angle

    # helper functions for applying bias correction
    def apply_multiplier(angle):
        speed = angle * bias_ds.sel(month=(angle.time_counter.dt.month[0].values))
        return speed
    def add_bias(ds_var):
        ds_var_corrected = ds_var + bias_ds.sel(month=(ds_var.time_counter.dt.month[0].values))
        return ds_var_corrected
    def add_angle(ds_var):
        ds_var_corrected = ds_var + bias_ds_angle.sel(month=(ds_var.time_counter.dt.month[0].values))
        return ds_var_corrected

    # loop over each year to apply bias correction
    for year in tqdm(range(start_year, end_year+1)):

        # for the u and v wind velocities, bias correction is based on a wind speed multiplier
        if variable=='wind_speed':
            file_pathx = find_processed_cesm2_file(expt, 'UBOT', ens, year, freq='3-hourly', highres=highres)
            file_pathy = find_processed_cesm2_file(expt, 'VBOT', ens, year, freq='3-hourly', highres=highres)
            dsx = xr.open_dataset(file_pathx, use_cftime=True, chunks='auto')
            dsy = xr.open_dataset(file_pathy, use_cftime=True, chunks='auto')
            theta   = np.arctan2(dsy.VBOT, dsx.UBOT)
            speed   = np.hypot(dsx.UBOT, dsy.VBOT)
            #theta_corrected = theta + bias_ds_angle # correct wind angle
            theta_corrected = theta.groupby(f'time_counter.month').apply(add_angle)
            angle_u = np.cos(theta_corrected) * speed
            angle_v = np.sin(theta_corrected) * speed
            if monthly:
                dsx['UBOT'] = angle_u.groupby('time_counter.month').apply(apply_multiplier)
                dsy['VBOT'] = angle_v.groupby('time_counter.month').apply(apply_multiplier)
            else:
                dsx['UBOT'] = angle_u * bias_ds.mean(dim='month')
                dsy['VBOT'] = angle_v * bias_ds.mean(dim='month')
               
            dsx.to_netcdf(f"{out_dir}CESM2-{expt}_ens{ens}_{'eANT025_' if highres else ''}{freq}_UBOT_bias_corr_{'monthly_' if monthly else ''}2D_y{year}.nc", \
                    unlimited_dims=['time_counter'])
            dsy.to_netcdf(f"{out_dir}CESM2-{expt}_ens{ens}_{'eANT025_' if highres else ''}{freq}_VBOT_bias_corr_{'monthly_' if monthly else ''}2D_y{year}.nc", \
                    unlimited_dims=['time_counter'])

        # every other variable has an addition spatial bias field correction
        else:
            file_path = find_processed_cesm2_file(expt, variable, ens, year, freq='daily', highres=highres)
            ds = xr.open_dataset(file_path, use_cftime=True)
            if not highres:
                ds = ds.rename({'time':'time_counter'})

            if monthly:
                ds[variable] = ds[variable].groupby(f'time_counter.month').apply(add_bias)
            else:
                ds[variable] += bias_corr[variable].mean(dim='month')

            if variable=='PRECS':
                # prevent the snowfall rate from exceeding the total precipitation rate
                # read in bias-corrected PRECT file
                try:
                    ds_PRECT = xr.open_dataset(f"{out_dir}CESM2-{expt}_ens{ens}_{'eANT025_' if highres else ''}{freq}_PRECT_bias_corr_{'monthly_' if monthly else ''}y{year}.nc", use_cftime=True)
                except:
                    raise Exception('Need to bias correct PRECT before PRECS for consistency of precipitation rates')
                
                # set the snowfall rate equal to the total precipitation rate if it is higher than it, otherwise leave as is
                ds['PRECS'] = xr.where(ds['PRECS'] > ds_PRECT['PRECT'], ds_PRECT['PRECT'], ds['PRECS'])

            if variable in ['PRECS', 'PRECT', 'FLDS', 'FSDS', 'QREFHT']:                    
                ds[variable] = xr.where(ds[variable] < 0, 0, ds[variable]) # should not be negative as a result of bias correction

            ds.to_netcdf(f"{out_dir}CESM2-{expt}_ens{ens}_{'eANT025_' if highres else ''}{freq}_{variable}_bias_corr_{'monthly_' if monthly else ''}y{year}.nc")

    return

# Function to pre-process ERA5 atmospheric forcing datasets
# Inputs:
# - variable : string of ERA5 variable name
# - (optional) year_start  : start year of range to process
# - (optional) year_end    : end year
# - (optinoal) era5_folder : path to ERA5 forcing files
def process_era5_forcing(variable, year_start=1979, year_end=2024, era5_folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/ERA5-forcing/daily/'):

    for year in range(year_start,year_end+1):
        if variable=='d2m':
            print('Convert dewpoint temperature')
            # convert dewpoint temperature to specific humidity and write to file (for bias correction calc)
            dewpoint_to_specific_humidity(file_dew=f'd2m_y{year}.nc', variable_dew='d2m',
                                          file_slp=f'msl_y{year}.nc', variable_slp='msl',
                                          dataset='ERA5', folder=f'{era5_folder}raw/')

        # fill land with nearest connected point (assume land mask is more or less constant over the reanalysis period)
        landmask = xr.open_dataset(f'{era5_folder}../climatology/land_sea_mask.nc').isel(time=0).lsm.rename({'longitude':'lon','latitude':'lat'})
        landmask['lon'] = fix_lon_range(landmask['lon'])
        landmask = landmask.sortby('lon')

        # convert time dimension to unlimited so that NEMO reads in the calendar correctly
        for filename in glob.glob(f'{era5_folder}raw/{variable}*y{year}.nc'):
            with xr.open_dataset(filename, mode='a') as data:
                print('Processing', filename)
                try:
                    data = data.rename({'valid_time':'time'})
                except:
                    pass

                # tidy up the longitude range and naming conventions
                data        = data.rename({'longitude':'lon', 'latitude':'lat'})
                data['lon'] = fix_lon_range(data['lon'])
                data        = data.sortby('lon')
                name_remap  = {'msdwlwrf':'avg_sdlwrf', 'msdwswrf':'avg_sdswrf', 'mtpr':'avg_tprate', 'msr':'avg_tsrwe'}
                try:
                    data = data.rename({f'{name_remap[variable]}':variable})
                except:
                    pass
                
                variable = filename.split('raw/')[1].split('_')[0]
                if variable in ['msdwlwrf','msdwswrf','t2m','sph2m','d2m','msl','msr','mtpr','t2m']: 
                    print(f'Filling land for variable {variable} year {year}')
                    if variable=='sph2m':
                        varname='specific_humidity'
                    else:
                        varname=variable
                    src_to_fill = xr.where(landmask!=0, -9999, data[varname]) # which cells need to be filled
                    var_filled_array = np.empty(src_to_fill.shape)
                    for tind, t in enumerate(src_to_fill.time):
                        var_filled_array[:,:,tind] = extend_into_mask(src_to_fill.isel(time=tind).values, missing_val=-9999, fill_val=np.nan, 
                                                                      use_2d=True, use_3d=False, num_iters=200)
                    data[varname] = (('lat','lon','time'), var_filled_array)
                    data = data.transpose('time','lat','lon')

                data.to_netcdf(f'{era5_folder}processed/{variable}_y{year}.nc', unlimited_dims={'time':True})

    return
