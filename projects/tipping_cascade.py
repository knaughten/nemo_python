import xarray as xr

from ..grid import build_ice_mask, region_mask
from ..plots import circumpolar_plot

# Create a mask file (cn_isfcav_spe_mask in namelist) which is 1 in the cells which should have prescribed melting, and 0 in the cells which should have prognostic melting.
# Pass the name of a region (as in region_mask in grid.py), and whether this region should be the only one prescribed (prescribed=True) or the only one prognostic (prescribed=False).
# IMPORTANT: this only works for static ice (standalone NEMO) configs. We will have to find a new approach for CANOBI runs, unless we can guarantee that the prescribed cavities will not change geometry at all.
def make_prescribed_melt_mask (region, out_file, prescribed=True, domain_cfg='/gws/ssde/j25b/anthrofail/birgal/NEMO_AIS/bathymetry/domain_cfg-20260108.nc'):

    # Build the region mask from domain_cfg
    ds_domcfg = xr.open_dataset(domain_cfg).squeeze()
    mask, ds_domcfg = region_mask(region, ds_domcfg, option='cavity')
    ocean_mask = build_ocean_mask(ds_domcfg)
    if prescribed:
        print(region+' prescribed melt, everywhere else prognostic')
        mask = xr.where(mask, 1, 0)
    else:
        print(region+' prognostic melt, everywhere else prescribed')
        mask = xr.where(mask, 0, 1)
    # Fill the land mask with 1s (prescribed to 0) as this might be slightly faster to run
    mask = xr.where(ocean_mask, mask, 1)
    # Plot for a sanity check
    circumpolar_plot(mask, ds_domcfg, masked=True, contour_ice=True)
    # Save to out_file
    ds_out = xr.Dataset({'mask':mask})
    ds_out.to_netcdf(out_file)
