import xarray as xr

from .constants import region_points, region_names, rho_fw, rho_ice, sec_per_year, deg_string, gkg_string
from .utils import cavity_mask, region_mask

# Calculate a timeseries of the given preset variable from an xarray Dataset of NEMO output. Returns DataArrays of the timeseries data, the associated time values, and the variable title.
# Also pass the grid file (mesh_mask for NEMO 3.6, domain_cfg for NEMO 4.2) and whether there is a halo in the data (if periodic grid, True for NEMO 3.6, False for NEMO 4.2).
# Preset variables include:
# <region>_massloss: basal mass loss from the given ice shelf or region of multiple ice shelves (eg brunt, amundsen_sea)
# <region>_bwtemp, <region>_bwsalt: area-averaged bottom water temperature or salinity from the given region or cavity (eg ross_cavity, ross_shelf, ross)
def calc_timeseries (var, ds_nemo, grid_file, halo=True):
    
    # Parse variable name
    factor = 1
    region_type = None
    if var.endswith('massloss'):
        option = 'area_int'
        region = var[:var.index('_massloss')]
        region_type = 'cavity'
        nemo_var = 'sowflisf'
        # Convert from kg/s to Gt/y and swap sign
        factor = -rho_ice/rho_fw*1e-12*sec_per_year
        units = 'Gt/y'
        title = 'Basal mass loss'
    elif var.endswith('bwtemp'):
        option = 'area_avg'
        region = var[:var.index('_bwtemp')]
        nemo_var = 'tob'
        units = deg_string+'C'
        title = 'Bottom temperature'
    elif var.endswith('bwsalt'):
        option = 'area_avg'
        region = var[:var.index('_bwsalt')]
        nemo_var = 'sob'
        units = gkg_string
        title = 'Bottom salinity'        

    # Select region
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
        mask, region_name = cavity_mask(region, grid_file, return_name=True)
    else:
        mask, region_name = region_mask(region, grid_file, option=region_type, return_name=True)
    title += ' from '+region_name    

    # Trim datasets as needed
    if ds_nemo.sizes['y'] < mask.sizes['y']:
        # The NEMO dataset was trimmed (eg by MOOSE for UKESM) to the southernmost latitudes. Do the same for the mask.
        mask = mask.isel(y=slice(0, ds_nemo.sizes['y']))
    if halo:
        # Remove the halo
        ds_nemo = ds_nemo.isel(x=slice(1,-1))
        mask = mask.isel(x=slice(1,-1))
        
    if option == 'area_int':
        # Area integral
        data = (ds_nemo[nemo_var]*ds_nemo['area']*mask).sum(dim=['x','y'])
    elif option == 'area_avg':
        # Area average
        data = (ds_nemo[nemo_var]*ds_nemo['area']*mask).sum(dim=['x','y'])/(ds_nemo['area']*mask).sum(dim=['x','y'])

    data *= factor

    return data, ds_nemo['time_centered'], title
        
        
