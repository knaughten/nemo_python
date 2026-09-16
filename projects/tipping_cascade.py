import xarray as xr
import warnings

from ..grid import build_ice_mask, region_mask, build_ocean_mask
from ..plots import circumpolar_plot
from ..file_io import find_files_in_range

# Create a mask file (cn_isfcav_spe_mask in namelist) which is 1 in the cells which should have prescribed melting, and 0 in the cells which should have prognostic melting.
# Pass the name of a region (as in region_mask in grid.py), and whether this region should be the only one prescribed (prescribed=True) or the only one prognostic (prescribed=False).
# IMPORTANT: this only works for static ice (standalone NEMO) configs. We will have to find a new approach for CANOBI runs, unless we can guarantee that the prescribed cavities will not change geometry at all.
def make_prescribed_melt_mask (region, out_file, prescribed=True, domain_cfg='/gws/ssde/j25b/anthrofail/birgal/NEMO_AIS/bathymetry/domain_cfg-20260108.nc'):

    # Build the region mask from domain_cfg
    ds_domcfg = xr.open_dataset(domain_cfg).squeeze()
    mask, ds_domcfg = region_mask(region, ds_domcfg, option='cavity')
    ocean_mask, ds_domcfg = build_ocean_mask(ds_domcfg)
    if prescribed:
        print(region+' prescribed melt, everywhere else prognostic')
        mask = xr.where(mask, 1, 0)
    else:
        print(region+' prognostic melt, everywhere else prescribed')
        mask = xr.where(mask, 0, 1)
    # Fill land mask with prescribed as this might save a bit of time in the loop at runtime
    mask = xr.where(ocean_mask==0, 1, mask)
    # Plot for a sanity check
    circumpolar_plot(mask.where(ocean_mask), ds_domcfg, masked=True, contour_ice=True, shade_land=True, title='Prescribed melt mask', lat_max=-63, ctype='plusminus')
    # Save to out_file
    ds_out = xr.Dataset({'mask':mask})
    ds_out.to_netcdf(out_file)


# Create a monthly climatology of ice shelf melting to prescribe for the cavities in the given mask file (created above by make_prescribed_melt_mask). Calculate this climatology from model output in the given directory, over the given years year_range=[start_year, end_year] (inclusive; default include all years).
def make_prescribed_melt_file (sim_dir, mask_file, out_file, year_range=None):

    file_head = 'eANT025.AntArc_1m_'
    file_tail = '_SBC.nc'
    var_name = 'fwfisf'

    nemo_files = find_files_in_range(sim_dir, file_head, file_tail, year_range=year_range)
    print('Reading files from '+nemo_files[0]+' to '+nemo_files[1]+' inclusive')
    # Open all the files at once
    ds = xr.open_mfdataset(nemo_files)
    # Select only the variable we want plus time index
    ds = ds[var_name, 'time_centered']
    # Now should be reasonable to load the whole thing into memory
    ds.load()
    # Calculate monthly climatology
    ds_clim = ds.groupby('time_centered.month')

    if ds_clim[var_name].mean() < 0:
        warnings.warn(var_name+' is net negative. Check the sign convention is what you expect; multiplying by -1.')
        ds_clim[var_name] = -1*ds_clim[var_name]

    # Now apply the prescribed mask
    ds_mask = xr.open_dataset(mask_file)
    ds_clim[var_name] = xr.where(ds_mask['mask']==1, ds_clim[var_name], 0)

    print('Writing '+out_file)
    ds_clim.to_netcdf(out_file)

            
                    
