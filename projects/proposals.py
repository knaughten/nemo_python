# Figures for proposals etc

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

from ..plots import circumpolar_plot, finished_plot
from ..grid import build_ocean_mask, build_ice_mask

def frontiers_figure (fig_name='frontiers_domain.pdf'):

    domain_cfg='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/domain_cfg-20260121.nc'
    depth0 = 700
    xlim = [-3e6, -2.7e5]
    ylim = [1e5, 3e6]

    ds = xr.open_dataset(domain_cfg).squeeze()
    ocean_mask = build_ocean_mask(ds)[0]
    ice_mask = build_ice_mask(ds)[0]
    bathy = ds['bathy_metry'].where(ocean_mask*(~ice_mask))

    fig, ax = plt.subplots(figsize=(6,8))
    ax.axis('equal')
    img = circumpolar_plot(bathy, ds, ax=ax, masked=True, title='', ctype='Blues', shade_land=True, contour=[depth0], contour_colour='red', make_cbar=False, contour_ice=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    finished_plot(fig, fig_name=fig_name, dpi=300)
    
    

    

    
