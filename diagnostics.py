import xarray as xr

from .utils import closest_point, remove_disconnected, rotate_vector
from .constants import ross_gyre_point0, region_bounds
from .interpolation import interp_grid

# Choose variable to represent dz; 3 choices depending on grid type
def choose_dz (ds, gtype):
    if gtype not in ['U', 'V']:
        raise Exception('Invalid gtype')
    if 'thkcello' in ds:
        dz = ds['thkcello']
    elif 'e3u' in ds and gtype=='U':
        dz = ds['e3u']
    elif 'e3v' in ds and gtype=='V':
        dz = ds['e3v']
    else:
        raise Exception('No option for dz')
    return dz
        

# Calculate zonal or meridional transport across the given section. The code will choose a constant slice in x or y corresponding to a constant value of latitude or longitude - so maybe not appropriate in highly rotated regions.
# For zonal transport, set lon0 and lat_bounds, and make sure the dataset includes uo, thkcello/e3u, and e2u (can get from domain_cfg).
# For meridional transport, set lat0 and lon_bounds, and make sure the dataset includes vo, thkcello/e3v, and e1v.
# Returns value in Sv.
def transport (ds, lon0=None, lat0=None, lon_bounds=None, lat_bounds=None):

    if lon0 is not None and lat_bounds is not None and lat0 is None and lon_bounds is None:
        # Zonal transport across line of constant longitude
        dz = choose_dz(ds, 'U')
        [j_start, i_start] = closest_point(ds, [lon0, lat_bounds[0]])
        [j_end, i_end] = closest_point(ds, [lon0, lat_bounds[1]])
        # Want a single value for i
        if i_start == i_end:
            # Perfect
            i0 = i_start
        else:
            # Choose the midpoint
            print('Warning (transport): grid is rotated; compromising on constant x-coordinate')
            i0 = int(round(0.5*(i_start+i_end)))
        # Assume velocity is already masked to 0 in land mask
        integrand = (ds['uo']*dz*ds['e2u']).isel(x=i0, y=slice(j_start, j_end+1))
        return integrand.sum(dim={'depthu', 'y'})*1e-6
    elif lat0 is not None and lon_bounds is not None and lon0 is None and lat_bounds is None:
        # Meridional transport across line of constant latitude
        dz = choose_dz(ds, 'V')
        [j_start, i_start] = closest_point(ds, [lon_bounds[0], lat0])
        [j_end, i_end] = closest_point(ds, [lon_bounds[1], lat0])
        if j_start == j_end:
            j0 = j_start
        else:
            print('Warning (transport): grid is rotated; compromising on constant y-coordinate')
            j0 = int(round(0.5*(j_start+j_end)))
        integrand = (ds['vo']*dz*ds['e1v']).isel(x=slice(i_start, i_end+1), y=j0)
        return integrand.sum(dim={'depthv', 'x'})*1e-6


# Calculate the barotropic streamfunction. 
def barotropic_streamfunction (ds_u, ds_v, ds_domcfg, periodic=True, halo=True):

    # Definite integral over depth
    udz = (ds_u['uo']*choose_dz(ds_u, 'U')).sum(dim='depthu')
    vdz = (ds_v['vo']*choose_dz(ds_v, 'V')).sum(dim='depthv')
    # Interpolate to t-grid
    udz_t = interp_grid(udz, 'u', 't', periodic=periodic, halo=halo)
    vdz_t = interp_grid(vdz, 'v', 't', periodic=periodic, halo=halo)
    # Rotate to get geographic velocity components, and save cos and sin of angles
    udz_tg, vdz_tg, cos_grid, sin_grid = rotate_vector(udz_t, vdz_t, ds_domcfg, gtype='T', periodic=periodic, halo=halo, return_angles=True)
    # Get integrand: dy in north-south direction, based on angle of grid
    dy = ds_domcfg['e2t']*cos_grid    
    # Indefinite integral from south to north, and convert to Sv
    return (udz_tg*dy).cumsum(dim='y')*1e-6


# Calculate the easternmost extent of the Ross Gyre: first find the 0 Sv contour of barotropic streamfunction which contains the point (160E, 70S), and then find the easternmost point within this contour.
# The dataset ds must include the variables uo, thkcello/e3u+e3v, e2u, nav_lon, nav_lat.
def ross_gyre_eastern_extent (ds_u, ds_v, ds_domcfg, periodic=True, halo=True):

    # Find all points where the barotropic streamfunction is negative
    strf = barotropic_streamfunction(ds_u, ds_v, ds_domcfg, periodic=periodic, halo=halo)    
    gyre_mask = strf < 0
    # Now only keep the ones connected to the known Ross Gyre point
    connected = gyre_mask.data
    for t in range(connected.shape[0]):
        connected[t,:] = remove_disconnected(gyre_mask.isel(time_counter=t), closest_point(ds, ross_gyre_point0))
    gyre_mask.data = connected
    # Find all longitudes within this mask which are also in the western hemisphere
    gyre_lon = ds_domcfg['nav_lon'].where((gyre_mask==1)*(ds_domcfg['nav_lon']<0))
    # Return the easternmost point
    return gyre_lon.max(dim={'x','y'})

# Calculate the transport within the given gyre: absolute value of the most negative streamfunction within the gyre bounds.
# Pass region = 'weddell' or 'ross'.
def gyre_transport (region, ds_u, ds_v, ds_domcfg, periodic=True, halo=True):

    strf = barotropic_streamfunction(ds_u, ds_v, ds_domcfg, periodic=periodic, halo=halo)
    # Identify Weddell Gyre region:
    [xW, xE, yS, yN] = region_bounds[region+'_gyre']
    if xW > xE:
        # Crosses 180 degrees longitude
        region_mask_lon = (ds_domcfg.nav_lon > xW) or (ds_domcfg.nav_lon < xE)
    else:
        region_mask_lon = (ds_domcfg.nav_lon > xW) and (ds_domcfg.nav_lat < xE)
    region_mask = region_mask_lon and (ds_domcfg.nav_lat > yS) and (ds_domcfg.nav_lat < yN)
    # Find the most negative streamfunction within the gyre bounds
    vmin = strf.where(region_mask).min(dim=['x','y'])

    return -1*vmin


# Function to calculate the coordinate of an isosurface such as a particular isotherm
# Linearly interpolates a coordinate isosurface where a field equals a target
# Code source: https://forum.access-hive.org.au/t/extracting-isopycnals-from-multi-dimensional-arrays/171
# Inputs:
# - field  : xarray DataArray to interpolate the target isosurface, i.e. temperature field if calculating an isotherm
# - target : float target isosurface value, i.e. temperature isotherm value to find
# - dim    : str of the field dimension to interpolate, i.e. 'deptht' to find depth of isotherm
def isosurface(field, target, dim):

    slice0 = {dim: slice(None, -1)}
    slice1 = {dim: slice(1, None)}

    field0 = field.isel(slice0).drop_vars(dim)
    field1 = field.isel(slice1).drop_vars(dim)

    crossing_mask_decr = (field0 > target) & (field1 <= target)
    crossing_mask_incr = (field0 < target) & (field1 >= target)
    crossing_mask = xr.where(crossing_mask_decr | crossing_mask_incr, 1, np.nan)

    coords0 = crossing_mask * field[dim].isel(slice0).drop_vars(dim)
    coords1 = crossing_mask * field[dim].isel(slice1).drop_vars(dim)
    field0 = crossing_mask * field0
    field1 = crossing_mask * field1

    iso = (coords0 + (target - field0) * (coords1 - coords0) / (field1 - field0) )

    return iso.max(dim, skipna=True)

# Function calculates the depth of the thermocline, returns the depth of the thermocline at each point in your input temperature array
# Metric is the depth of the maximum positive temperature gradient with depth (dT/dz)
# Inputs:
# - dsT                  : xarray DataArray of temperature (can deal with 4D)
# - (optional) mesh_mask : path to configuration mesh mask file containing tmask and gdept_0
# - (optional) surface_depth_mask : float depth in metres to mask from calculation (if you'd like to exclude the seasonal mixed layer gradients for example)
def thermocline(dsT, mesh_mask='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20250715.nc', surface_depth_mask=120):
    
    # load mesh mask file and mask temperature with land points
    ds_mesh  = xr.open_dataset(mesh_mask).squeeze().rename({'nav_lev':'deptht','x':'x_grid_T','y':'y_grid_T'}).drop_vars('time_counter')
    T_masked = dsT.where(ds_mesh.tmask==1)
    
    # calculate the derivative of temperature with depth
    dTdz         = T_masked.differentiate('deptht')
    # mask grid locations that are land b/c argmax can't deal with it and mask depths in the upper surface_depth_mask meters due to strong seasonal gradients
    dTdz_masked  = xr.where(dTdz.sum(dim='deptht')==0, 0, dTdz) 
    dTdz_masked  = xr.where(ds_mesh.gdept_0 < surface_depth_mask, 0, dTdz_masked) 
    # find depth of maximum dT/dz gradient, i.e. thermocline
    zind_thermocline  = dTdz_masked.argmax(dim='deptht')               # find z index of the maximum dT/dz gradient
    depth_thermocline = ds_mesh.gdept_0.isel(deptht=zind_thermocline)  # identify depth associated with this index
    # only want thermoclines that are positive; otherwise you get a few at seafloor in deep troughs in the northern southern ocean
    depth_thermocline = xr.where((dTdz_masked.max(dim='deptht') > 0), depth_thermocline, np.nan) 
        
    return depth_thermocline
