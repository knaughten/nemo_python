import numpy as np
import xarray as xr
from .constants import deg2rad, rho_fw, sec_per_hour, temp_C2K, Rdry, Rvap, vap_pres_c1, vap_pres_c3, vap_pres_c4, months_per_year, rEarth, rho_ice, sec_per_year

# Given an array containing longitude, make sure it's in the range (max_lon-360, max_lon). Default is (-180, 180). If max_lon is None, nothing will be done to the array.
def fix_lon_range (lon, max_lon=180):

    if isinstance(lon, xr.DataArray):
        lon = xr.where(lon >= max_lon, lon-360, lon)
        lon = xr.where(lon < max_lon-360, lon+360, lon)
    elif isinstance(lon, np.ndarray):
        index = lon >= max_lon
        lon[index] = lon[index] - 360
        index = lon < max_lon-360
        lon[index] = lon[index] + 360
    elif np.isscalar(lon):
        lon = fix_lon_range(np.array([lon]), max_lon=max_lon)[0]
    else:
        raise Exception('unsupported data type')
    return lon


# Convert longitude and latitude to Antarctic polar stereographic projection. Adapted from polarstereo_fwd.m in the MITgcm Matlab toolbox for Bedmap.
def polar_stereo (lon, lat, a=6378137., e=0.08181919, lat_c=-71, lon0=0):

    if lat_c < 0:
        # Southern hemisphere
        pm = -1
    else:
        # Northern hemisphere
        pm = 1

    # Prepare input
    lon_rad = lon*pm*deg2rad
    lat_rad = lat*pm*deg2rad
    lat_c = lat_c*pm*deg2rad
    lon0 = lon0*pm*deg2rad

    # Calculations
    t = np.tan(np.pi/4 - lat_rad/2)/((1 - e*np.sin(lat_rad))/(1 + e*np.sin(lat_rad)))**(e/2)
    t_c = np.tan(np.pi/4 - lat_c/2)/((1 - e*np.sin(lat_c))/(1 + e*np.sin(lat_c)))**(e/2)
    m_c = np.cos(lat_c)/np.sqrt(1 - (e*np.sin(lat_c))**2)
    rho = a*m_c*t/t_c
    x = pm*rho*np.sin(lon_rad - lon0)
    y = -pm*rho*np.cos(lon_rad - lon0)

    if isinstance(x, xr.DataArray) and len(lon.shape)==1:
        # Case that input arrays were 1D: default casting is to have x as the first coordinate; this is not what we want
        lon = lon.transpose()
        lat = lat.transpose()

    return x, y


# Convert from polar stereographic coordinates to lat-lon. Adapated from the function psxy2ll.m used by Ua (with credits to Craig Stewart, Adrian Jenkins, Pierre Dutrieux) and made more consistent with naming convections of function above.
# This is about twice as fast as the pyproj Transformer function (for BedMachine v3 at least), but it is limited to this specific case so could consider changing in the future if I end up using more projections than just these two.
def polar_stereo_inv (x, y, a=6378137., e=0.08181919, lat_c=-71, lon0=0):

    if not isinstance(x, xr.DataArray) and len(x.shape)==1:
        # Need to broadcast dimensions.
        x, y = np.meshgrid(x, y)

    if lat_c < 0:
        pm = -1
    else:
        pm = 1
    lat_c = lat_c*pm*deg2rad
    lon0 = lon0*deg2rad
    epsilon = 1e-12

    tc = np.tan(np.pi/4 - lat_c/2)/((1 - e*np.sin(lat_c))/(1 + e*np.sin(lat_c)))**(e/2)
    mc = np.cos(lat_c)/np.sqrt(1 - e**2*(np.sin(lat_c))**2)
    rho = np.sqrt(x**2 + y**2)
    t = rho*tc/(a*mc)
    lon = lon0 + np.arctan2(x,y)

    lat_new = np.pi/2 - 2*np.arctan(t)
    dlat = 2*epsilon
    while dlat > epsilon:
        lat_old = lat_new
        lat_new = np.pi/2 - 2*np.arctan(t*((1 - e*np.sin(lat_old))/(1 + e*np.sin(lat_old)))**(e/2))
        dlat = np.amax(lat_new - lat_old)
    lat = lat_new

    lat = lat*pm/deg2rad
    lon = fix_lon_range(lon/deg2rad)

    if isinstance(lon, xr.DataArray) and len(x.shape)==1:
        # Case that input arrays were 1D: default casting is to have x as the first coordinate; this is not what we want
        lon = lon.transpose()
        lat = lat.transpose()

    return lon, lat


# Given an array of grid values on the edges (gtype=u, v) or corners (gtype=f) of the grid, extend by one column to the west and/or row to the south so that all of the tracer points have edges defined on both sides.
# Note that the index convention of the resulting array will change relative to the tracer grid. A t-point (j,i) at the centre of the cell originally has the corresponding f-point (j,i) to the northeast corner of the cell, but after this padding of the f-grid, the corresponding f-point (j,i) will be at the southwest corne of the cell.
# This should also work if "array" is a Dataset instead of a DataArray.
def extend_grid_edges (array, gtype, periodic=True, halo=False):

    if gtype in ['u', 'f']:
        # New column to the west
        if periodic:
            # The western edge already exists on the other side
            if halo:
                edge_W = array.isel(x=-3)
            else:
                edge_W = array.isel(x=-1)
        else:
            # Extrapolate
            edge_W = 2*array.isel(x=0) - array.isel(x=1)
        array = xr.concat([edge_W, array], dim='x')
    if gtype in ['v', 'f']:
        # New column to the south: extrapolate
        edge_S = 2*array.isel(y=0) - array.isel(y=1)
        array = xr.concat([edge_S, array], dim='y')
    return array.transpose('y', 'x')


# Return the deepest unmasked values along the named z-dimension of the given xarray DataArray.
# Following https://stackoverflow.com/questions/74172428/calculate-the-first-instance-of-a-value-in-axis-xarray
def select_bottom (array, zdim):

    bottom_depth = array.coords[zdim].where(array.notnull()).max(dim=zdim)
    return array.sel({zdim:bottom_depth.fillna(0).astype(int)}).where(bottom_depth.notnull())


# Given a mask (numpy array, 1='land', 0='ocean') and point0 (j,i) on the "mainland", remove any disconnected "islands" from the mask and return.
def remove_disconnected (mask, point0):

    if not mask[point0]:
        raise Exception('point0 is not on the mainland')

    connected = np.zeros(mask.shape)
    connected[point0] = 1
    ny = mask.shape[0]
    nx = mask.shape[1]

    queue = [point0]
    while len(queue) > 0:
        (j,i) = queue.pop(0)
        neighbours = []
        if j > 0:
            neighbours.append((j-1,i))
        if j < ny-1:
            neighbours.append((j+1,i))
        if i > 0:
            neighbours.append((j,i-1))
        if i < nx-1:
            neighbours.append((j,i+1))
        for point in neighbours:
            if connected[point]:
                continue
            if mask[point]:
                connected[point] = True
                queue.append(point)

    return connected


# Choose the right latitude and longitude name for the given dataset (sometimes it's stamped with the grid)
def latlon_name (ds):

    if 'nav_lat' in ds:
        return 'nav_lon', 'nav_lat'
    elif 'nav_lat_grid_T' in ds:
        return 'nav_lon_grid_T', 'nav_lat_grid_T'
    elif 'nav_lat_grid_V' in ds:
        return 'nav_lon_grid_V', 'nav_lat_grid_V'
    elif 'nav_lat_grid_U' in ds:
        return 'nav_lon_grid_U', 'nav_lat_grid_U'
    elif 'lat' in ds:
        return 'lon', 'lat'
    elif 'latitude' in ds:
        return 'longitude', 'latitude'
    else:
        raise Exception('No valid lat or lon coordinate')


def xy_name (ds):

    if 'y' in ds.dims:
        return 'x', 'y'
    elif 'y_grid_T' in ds.dims:
        return 'x_grid_T', 'y_grid_T'
    elif 'y_grid_V' in ds.dims:
        return 'x_grid_V', 'y_grid_V'
    elif 'y_grid_U' in ds.dims:
        return 'x_grid_U', 'y_grid_U'
    elif 'ny' in ds.dims:
        return 'nx', 'ny'
    else:
        raise Exception('No valid x or y coordinate')


# Find the (y,x) coordinates of the closest model point to the given (lon, lat) coordinates. Pass an xarray Dataset containing nav_lon, nav_lat, and a target point (lon0, lat0).
def closest_point (ds, target):

    lon_name, lat_name = latlon_name(ds)
    lon = ds[lon_name].squeeze()
    lat = ds[lat_name].squeeze()
    [lon0, lat0] = target
    # Calculate distance of every model point to the target
    dist = np.sqrt((lon-lon0)**2 + (lat-lat0)**2)
    # Find the indices of the minimum distance
    x_name, y_name = xy_name(ds)
    point0 = dist.argmin(dim=(y_name, x_name))
    return (int(point0[y_name].data), int(point0[x_name].data))

# Helper function to calculate the Cartesian distance between two longitude and latitude points
# This also works if one of point0, point1 is a 2D array.
def distance_btw_points (point0, point1):
    
    [lon0, lat0] = point0
    [lon1, lat1] = point1
    
    dx = rEarth*np.cos((lat0+lat1)/2*deg2rad)*(lon1-lon0)*deg2rad
    dy = rEarth*(lat1-lat0)*deg2rad
    
    return np.sqrt(dx**2 + dy**2)


# Calculate the distance of every lat-lon point in the model grid to the closest point of the given mask, in km. Works best if mask is True for a small number of points (eg a grounding line or a coastline).
def distance_to_mask (lon, lat, mask):

    mask = mask==1
    min_dist = None
    # Loop over individual points in the mask
    for lon0, lat0 in zip(lon.data[mask.data], lat.data[mask.data]):
        # Calculate distance of every other point to this point
        dist_to_pt = distance_btw_points([lon, lat], [lon0, lat0])*1e-3
        if min_dist is None:
            # Initialise array with distance to the first point
            min_dist = dist_to_pt.copy()
        else:
            min_dist = np.minimum(min_dist, dist_to_pt)
    return min_dist


# Calculate the distance of every lat-lon point in the model grid to the boundary of the given mask, in km. The distance will be 0 where the mask is True but has a neighbour which is False.
def distance_to_bdry (lon, lat, mask, periodic=True):

    # Inner function to pad the edges (flagged with NaN) with a copy of the last row
    def pad_edges (mask_new):
        return xr.where(mask_new.isnull(), mask, mask_new)
    # Find neighbours to the north, south, east, west
    mask_n = pad_edges(mask.shift(y=-1))
    mask_s = pad_edges(mask.shift(y=1))
    if periodic:
        mask_e = mask.roll(x=-1)
        mask_w = mask.roll(x=1)
    else:
        mask_e = pad_edges(mask.shift(x=-1))
        mask_w = pad_edges(mask.shift(x=1))
    # Find points on the boundary: mask is True, but at least one neighbour is False
    bdry = mask.where(mask_n*mask_s*mask_e*mask_w==0)
    # Return distance to that boundary
    return distance_to_mask(lon, lat, bdry)    


# Function calculates distances (km) from each point in a transect to the first point based on lats and lons
def distance_along_transect(data_transect):

    # calculate distance from each point in the transect to the first point in the transect:
    transect_distance = np.array([distance_btw_points((data_transect.nav_lon.values[0], data_transect.nav_lat.values[0]),
                                                      (data_transect.nav_lon.values[i+1], data_transect.nav_lat.values[i+1])) for i in range(0, data_transect.n.size-1)])
    # prepend 0 for the first distance point
    transect_distance = np.insert(transect_distance, 0, 0) 
    # convert from meters to km
    transect_distance = transect_distance/1000
    
    return transect_distance
    

# Function to convert the units of shortwave and longwave radiation to the units expected by NEMO (W m-2)
# Reads the specified variable from the NetCDF file and writes the converted variable to a new file in the same folder
# with the file name starting with "converted_"
# Input: 
# file_rad: string name of the atmospheric forcing NetCDF file 
# variable: string name of the radiation variable within the file specified by file_rad
# dataset: string specifying type of atmospheric forcing dataset (ERA5, JRA etc.)
# folder: string of location that contains the atmospheric forcing files
def convert_radiation(file_rad='era5_strd_1979_daily_averages.nc', variable='strd', 
                      dataset='ERA5', folder='/gws/nopw/j04/terrafirma/birgal/NEMO_AIS/ERA5-forcing/'):
    if dataset=='ERA5':
        # ERA5 is in J m-2, convert to Watt m-2 = J m-2 s-1, so divide by the accumulation period in seconds
        # In this case, the files are daily averages of the original hourly files. So, the J/m-2 is actually the accumulation over an hour. 
        ds = xr.open_dataset(f'{folder}{file_rad}') # shortwave or longwave radiation
        ds[variable] = ds[variable] / sec_per_hour
        ds.to_netcdf(f'{folder}converted_{file_rad}')
        
        return 
    else:
        raise Exception('Only currently set up to convert ERA5 units to nemo units')

# Function to convert the units of precipitation from m of water equivalent to the units expected by NEMO (kg m-2 s-1)
# Reads the specified variable from the NetCDF file and writes the converted variable to a new file in the same folder
# with the file name starting with "converted_"
# Input: 
# file_precip: string name of the atmospheric forcing NetCDF file 
# variable: string name of the precipitation variable within the file specified by file_precip
# dataset: string specifying type of atmospheric forcing dataset (ERA5, JRA etc.)
# folder: string of location that contains the atmospheric forcing files
def convert_precip(file_precip='era5_tp_1979_daily_averages.nc', variable='tp', 
                   dataset='ERA5', folder='/gws/nopw/j04/terrafirma/birgal/NEMO_AIS/ERA5-forcing/'):
    if dataset=='ERA5':
        # ERA5 is in m of water equivalent, convert to kg m-2 s-1, so need to divide by the accumulation period, and convert density
        ds = xr.open_dataset(f'{folder}{file_precip}')
        # m --> m/s --> kg/m2/s
        ds[variable] = (ds[variable] / sec_per_hour) * rho_fw # total precip is in meters of water equivalent
        ds.to_netcdf(f'{folder}converted_{file_precip}')

        return        
    else:
        raise Exception('Only currently set up to convert ERA5 units to nemo units')

# Function to calculate specific humidity from dewpoint temperature and atmospheric pressure
# Reads the specified variable from the NetCDF file and writes the converted variable to a new file in the same folder
# with the file name starting with "converted_"
# Input: 
# file_dew: string name of the dewpoint temperature NetCDF file 
# file_slp: string name of the sea level pressure NetCDF file 
# variable_dew: string name of the dewpoint temperature variable within the file specified by file_dew
# variable_slp: string name of the sea level pressure variable within the file specified by file_slp
# dataset: string specifying type of atmospheric forcing dataset (ERA5, JRA etc.)
# folder: string of location that contains the atmospheric forcing files
def dewpoint_to_specific_humidity(file_dew='d2m_y1979.nc', variable_dew='d2m',
                                  file_slp='msl_y1979.nc', variable_slp='msl',
                                  dataset='ERA5', folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/ERA5-forcing/daily/files/'):
    if dataset=='ERA5':
        # ERA5 does not provide specific humidity, but gives the 2 m dewpoint temperature in K
        # Conversion assumes temperature is in K and pressure in Pa.
        # Based off: https://confluence.ecmwf.int/pages/viewpage.action?pageId=171411214

        ds               = xr.open_dataset(f'{folder}{file_dew}')
        surface_pressure = xr.open_dataset(f'{folder}{file_slp}')[variable_slp]

        dewpoint = ds[variable_dew]
        # calculation:
        vapor_pressure = vap_pres_c1*np.exp(vap_pres_c3*(dewpoint.values - temp_C2K)/(dewpoint.values - vap_pres_c4)) # E saturation water vapour from Teten's formula
        spec_humidity  = (Rdry / Rvap) * vapor_pressure / (surface_pressure - ((1-Rdry/Rvap)*vapor_pressure)) # saturation specific humidity

        ds[variable_dew] = spec_humidity
        ds = ds.rename_vars({variable_dew:'specific_humidity'}) 
        filename = file_dew.replace('d2m', 'sph2m')
        ds.to_netcdf(f'{folder}{filename}')
        
        return
    else:
        raise Exception('Only currently set up to convert ERA5 units to nemo units')     

# Function to ensure the reference time is consistent between atmospheric data sources
# JRA uses days since 1900-01-01 00:00:00 on a Gregorian calendar
# ERA uses days since start of that particular year in proleptic gregorian calendar
# Input:
# ds : xarray dataset containing variable 'time'
# dataset : name of atmospheric forcing dataset
def convert_time_units(ds, dataset='ERA5'):

    if dataset=='ERA5':
        ds['time'] = ds.time.values
        ds['time'].encoding['units'] = "days since 1900-01-01"
        ds['time'].encoding['calendar'] = 'gregorian'
        return ds
    else:
        raise Exception('Only currently set up to convert ERA5 reference period')


# Advance the given date (year and month, both ints) by num_months
def add_months (year, month, num_months):
    
    month += num_months
    while month > months_per_year:
        month -= months_per_year
        year += 1
    return year, month


# Smooth the given DataArray with a moving average of the given window, over the given dimension (default time_centered).
# per_year = the number of time indices per year (default 12 for monthly data); used to interpolate any missing values while preserving the seasonal cycle.
def moving_average (data, window, dim='time_centered', per_year=12):

    if window == 0:
        return data

    # Interpolate any missing values
    if any(data.isnull()):
        # Loop over months (or however many indices are in the seasonal cycle)
        for t in range(per_year):
            # Select this month
            data_tmp = data[t::per_year]
            # Interpolate NaNs
            data_tmp = data_tmp.interpolate_na(dim=dim)
            # Put back into main array
            data_tmp = data_tmp.reindex_like(data)
            index = data.isnull()
            data[index] = data_tmp[index]

    # Find axis number of dimension
    dim_axis = 0
    for var in data.sizes:
        if var == dim:
            break
        dim_axis += 1

    centered = window%2==1
    if centered:
        radius = (window-1)//2
    else:
        radius = window//2
    t_first = radius
    t_last = data.sizes[dim] - radius  # First one not selected, as per python convention
    # Array of zeros of the same shape as a single time index of data
    zero_base = (data.isel({dim:slice(0,1)})*0).data
    # Do the smoothing in two steps, in numpy world
    data_np = np.ma.masked_where(np.isnan(data.data), data.data)    
    data_cumsum = np.ma.concatenate((zero_base.data, np.ma.cumsum(data_np, axis=dim_axis)), axis=dim_axis)
    if centered:
        data_smoothed = (data_cumsum[t_first+radius+1:t_last+radius+1,...] - data_cumsum[t_first-radius:t_last-radius,...])/(2*radius+1)
    else:
        data_smoothed = (data_cumsum[t_first+radius:t_last+radius,...] - data_cumsum[t_first-radius:t_last-radius,...])/(2*radius)
    # Now trim the original array
    data_trimmed = data.isel({dim:slice(radius, -radius)})
    if not centered:
        # Shift time dimension half an index forward
        time1 = data[dim].isel({dim:slice(radius-1, -radius-1)})
        time2 = data[dim].isel({dim:slice(radius, -radius)})
        time_trimmed = time1.data + (time2.data-time1.data)/2
        data_trimmed[dim] = time_trimmed
    data_trimmed.data = data_smoothed
    return data_trimmed


# Given a string representing a month, convert between string representations (3-letter lowercase, eg jan) and int representations (2-digit string, eg 01).
def month_convert (date_code):

    month_str = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_int = [str(n+1).zfill(2) for n in range(months_per_year)]

    if isinstance(date_code, int):
        date_code = str(date_code).zfill(2)
        
    if date_code in month_str:
        n = month_str.index(date_code)
        return month_int[n]
    elif date_code in month_int:
        n = month_int.index(date_code)
        return month_str[n]
    else:
        raise Exception('Invalid month')


# Rotate a vector from the local x-y space to geographic space (with true zonal and meridional components) or do the reverse. Follows subroutines "angle" and "rot_rep" in nemo/src/OCE/SBC/geo2ocean.F90.
# Warning, this might not handle the north pole correctly, so use with caution if you are plotting a global grid and care about the north pole.
# Inputs:
# u, v: xarray DataArrays containing the u and v components of the vector in local x-y space (eg velocities from NEMO output) or if reverse=True containing velocities in geographic space that will be reversed to local x-y space
# domcfg: either the path to the domain_cfg file, or an xarray Dataset containing glamt, gphit, glamu, etc.
# gtype: grid type: 'T', 'U', 'V', or 'F'. In practice you will interpolate both velocities to the T-grid and then call this with gtype='T' (default).
# periodic: whether the grid is periodic
# halo: whether the halo is included in the arrays (generally true for NEMO3.6, false for NEMO4.2). Only matters if periodic=True.
# return_angles: whether to return cos_grid and sin_grid
# Outputs:
# ug, vg: xarray DataArrays containing the zonal and meridional components of the vector in geographic space
# cos_grid, sin_grid (only if return_angles=True): cos and sin of the angle between the grid and east
def rotate_vector (u, v, domcfg, gtype='T', periodic=True, halo=True, return_angles=False, reverse=False):

    # Repeat all necessary import statements within here so the function is self-contained (someone can just copy and paste the whole thing if wanted)
    import xarray as xr
    import numpy as np
    deg2rad = np.pi/180.

    if isinstance(domcfg, str):
        domcfg = xr.open_dataset(domcfg)
    if domcfg.sizes['y'] != u.sizes['y']:
        # The TerraFIRMA overshoot output was trimmed to only cover the Southern Ocean when I pulled it from MASS, while domain_cfg remains global. Assume this is the reason for the mismatch but print a warning.
        print('Warning (rotate_vector): trimming domain_cfg to select only the southernmost part, to align with input vectors - is this what you want?')
        domcfg = domcfg.isel(y=slice(0, u.sizes['y']))
    domcfg = domcfg.squeeze()

    if u.dims != v.dims:
        # Dimensions don't match. Usually this is because 3D velocities have retained 'depthu' and 'depthv' dimensions even though they've been interpolated to the T-grid.
        if gtype in ['T', 't'] and 'depthu' in u.dims and 'depthv' in v.dims:
            u = u.rename({'depthu':'deptht'})
            v = v.rename({'depthv':'deptht'})
        # Check again
        if u.dims != v.dims:
            raise Exception('Mismatch in dimensions')

    # Get lon and lat on this grid
    lon = domcfg['glam'+gtype.lower()]
    lat = domcfg['gphi'+gtype.lower()]

    # Calculate, at each point, the x and y components and the squared-norm of the vector between the point and the North Pole
    vec_NP_x = -2*np.cos(lon*deg2rad)*np.tan(np.pi/4 - lat*deg2rad/2)
    vec_NP_y = -2*np.sin(lon*deg2rad)*np.tan(np.pi/4 - lat*deg2rad/2)
    vec_NP_norm2 = vec_NP_x**2 + vec_NP_y**2

    # Inner function to get adjacent points on an alternate grid.
    def grid_edges (var_name, shift):
        edge1 = domcfg[var_name]
        if shift == 'j-1':
            edge2 = edge1.shift(y=1)
            # Extrapolate southern boundary
            edge2.isel(y=0).data = 2*edge1.isel(y=1).data - edge1.isel(y=0).data
        elif shift == 'j+1':
            edge2 = edge1.shift(y=-1)
            # Extrapolate northern boundary
            edge2.isel(y=-1).data = 2*edge1.isel(y=-2).data - edge1.isel(y=-1).data
        elif shift == 'i-1':
            edge2 = edge1.shift(x=1)
            if periodic:
                # Western boundary already exists on the other side
                if halo:
                    edge2.isel(x=0).data = edge1.isel(x=-3).data
                else:
                    edge2.isel(x=0).data = edge1.isel(x=-1).data
            else:
                # Extrapolate western boundary
                edge2.isel(x=0).data = 2*edge1.isel(x=1).data - edge1.isel(x=0).data        
        return edge1, edge2
    # Call this function for both lon and lat on the given grid.
    def lonlat_edges (gtype2, shift):
        lon_edge1, lon_edge2 = grid_edges('glam'+gtype2.lower(), shift)
        lat_edge1, lat_edge2 = grid_edges('gphi'+gtype2.lower(), shift)
        return lon_edge1, lat_edge1, lon_edge2, lat_edge2            

    # Calculate, at each point, the x and y components and the norm of the vector between adjacent points on an alternate grid.
    if gtype in ['T', 't']:
        # v-points above and below the given t-point
        lon_edge1, lat_edge1, lon_edge2, lat_edge2 = lonlat_edges('v', 'j-1')
    elif gtype in ['U', 'u']:
        # f-points above and below the given u-point
        lon_edge1, lat_edge1, lon_edge2, lat_edge2 = lonlat_edges('f', 'j-1')
    elif gtype in ['V', 'v']:
        # f-points left and right of the given v-point
        lon_edge1, lat_edge1, lon_edge2, lat_edge2 = lonlat_edges('f', 'i-1')
    elif gtype in ['F', 'f']:
        # u-points above and below the given f-point
        # Note reversed order of how we save the outputs
        lon_edge2, lat_edge2, lon_edge1, lat_edge1 = lonlat_edges('u', 'j+1')
    vec_pts_x = 2*np.cos(lon_edge1*deg2rad)*np.tan(np.pi/4 - lat_edge1*deg2rad/2) - 2*np.cos(lon_edge2*deg2rad)*np.tan(np.pi/4 - lat_edge2*deg2rad/2)
    vec_pts_y = 2*np.sin(lon_edge1*deg2rad)*np.tan(np.pi/4 - lat_edge1*deg2rad/2) - 2*np.sin(lon_edge2*deg2rad)*np.tan(np.pi/4 - lat_edge2*deg2rad/2)
    vec_pts_norm = np.maximum(np.sqrt(vec_NP_norm2*(vec_pts_x**2 + vec_pts_y**2)), 1e-14)

    # Now get sin and cos of the angles of the given grid
    if gtype in ['V', 'v']:
        sin_grid = (vec_NP_x*vec_pts_x + vec_NP_y*vec_pts_y)/vec_pts_norm
        cos_grid = -(vec_NP_x*vec_pts_y - vec_NP_y*vec_pts_x)/vec_pts_norm
    else:
        sin_grid = (vec_NP_x*vec_pts_y - vec_NP_y*vec_pts_x)/vec_pts_norm
        cos_grid = (vec_NP_x*vec_pts_x + vec_NP_y*vec_pts_y)/vec_pts_norm

    # Identify places where the adjacent grid cells are essentially equal (can happen with weird patched-together grids etc filling parts of Antarctic land mask with constant values) - no rotation needed here
    eps = 1e-8
    if gtype in ['T', 't']:
        lon_edge1, lon_edge2 = grid_edges('glamv', 'j-1')        
    elif gtype in ['U', 'u']:
        lon_edge1, lon_edge2 = grid_edges('glamf', 'j-1')
    elif gtype in ['V', 'v']:
        lat_edge1, lat_edge2 = grid_edges('gphif', 'i-1')
    elif gtype in ['F', 'f']:
        lon_edge1, lon_edge2 = grid_edges('glamu', 'j+1')
    if gtype in ['V', 'v']:
        index = np.abs(lat_edge1-lat_edge2) < eps
    else:
        index = np.abs(np.mod(lon_edge1-lon_edge2, 360)) < eps
    sin_grid = xr.where(index, 0, sin_grid)
    cos_grid = xr.where(index, 1, cos_grid)

    # Finally, rotate!
    if reverse: # go from grid i-j direction to geographic u, v, such as for boundary conditions
        ug = u*cos_grid + v*sin_grid
        vg = v*cos_grid - u*sin_grid
    else:
        ug = u*cos_grid - v*sin_grid
        vg = v*cos_grid + u*sin_grid

    if return_angles:
        return ug, vg, cos_grid, sin_grid
    else:
        return ug, vg


# Helper function to convert an xarray dataset with 3D T and S to TEOS10 (absolute salinity and conservative temperature)
# Inputs: 
# dataset: xarray dataset containing variables lon, lat, depth, and THETA (potential temperature) or SALT (practical salinity)
# var:     string of variable name to convert: THETA or SALT
def convert_to_teos10(dataset, var='PracSal'):
    import gsw
    # Convert to TEOS10
    # Check if dataset contains pressure, otherwise use depth:
    if 'pressure' in list(dataset.keys()):
        var_press = 'pressure'
    else:
        var_press = 'depth'
    # Need 3D lat, lon, pressure at every point, so if 1D or 2D, broadcast to 3D    
    if dataset.lon.values.ndim <= 2:
        lon   = xr.broadcast(dataset['lon'], dataset[var])[0]
    if dataset.lat.values.ndim <= 2:
        lat   = xr.broadcast(dataset['lat'], dataset[var])[0]
    if dataset[var_press].values.ndim <= 2:
        # Need pressure in dbar at every 3D point: approx depth in m
        press = np.abs(xr.broadcast(dataset[var_press], dataset[var])[0])
    else:
        press = np.abs(dataset[var_press])
    
    if var=='PracSal':
        # Get absolute salinity from practical salinity
        absS  = gsw.SA_from_SP(dataset[var], press, lon, lat)

        return absS.rename('AbsSal')
    elif var=='InsituTemp':    
        if 'PracSal' in list(dataset.keys()):
            # Get absolute salinity from practical salinity
            absS  = gsw.SA_from_SP(dataset['PracSal'], press, lon, lat)
            # Get conservative temperature from potential temperature
            consT  = gsw.CT_from_t(absS, dataset[var], press)
        else:
            raise Exception('Must include practical salinity (PracSal) variable in dataset when converting in-situ temperature')
        
        return consT.rename('ConsTemp')
    elif var=='PotTemp': # potential temperature    
        if 'PracSal' in list(dataset.keys()):
            # Get absolute salinity from practical salinity
            absS  = gsw.SA_from_SP(dataset['PracSal'], press, lon, lat)
            # Get conservative temperature from potential temperature
            consT  = gsw.CT_from_pt(absS.values, dataset[var])
        elif 'AbsSal' in list(dataset.keys()):
            consT = gsw.CT_from_pt(dataset['AbsSal'].values, dataset[var])
        else:
            raise Exception('Must include practical salinity (PracSal) variable in dataset when converting potential temperature')
        
        return consT.rename('ConsTemp')
    else:
        raise Exception('Variable options are PracSal, InsituTemp, PotTemp')


# Convert freshwater flux into the ice shelf (sowflisf) (kg/m^2/s of water, positive means freezing) to ice shelf melt rate (m/y of ice, positive means melting).
def convert_ismr (sowflisf):

    return -sowflisf/rho_ice*sec_per_year


# Read absolute bottom salinity from a NEMO dataset in EOS80.
def bwsalt_abs (ds_nemo):
    import gsw
    SP = ds_nemo['sob']
    # Get depth in metres at every point, with land masked
    depth_3d = xr.broadcast(ds_nemo['deptht'], ds_nemo['so'])[0].where(ds_nemo['so']!=0)
    # Get depth in bottom cell: approximately equal to pressure in dbar
    press = depth_3d.max(dim='deptht')
    return gsw.SA_from_SP(SP, press, ds_nemo['nav_lon'], ds_nemo['nav_lat'])


# Select the correct variable for area in the dataset
def area_name (ds, gtype='T'):

    var_names = ['area', 'area_grid_'+gtype]
    for v in var_names:
        if v in ds:
            return v
    raise Exception('No area variable found')


# Same for cell thickness
def dz_name (ds, gtype='T'):

    var_names = ['thkcell'+gtype.lower()+'o', 'thkcello', 'e3'+gtype.lower()]
    for v in var_names:
        if v in ds:
            return v
    raise Exception('No dz variable found')
    

    

    




    
            
        
        
        

    


    

    
    
