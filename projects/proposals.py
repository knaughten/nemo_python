# Figures for proposals etc

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..plots import circumpolar_plot, finished_plot
from ..grid import build_ocean_mask, build_ice_mask

def frontiers_figure (fig_name='frontiers_domain.png'):

    domain_cfg='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/domain_cfg-20260121.nc'
    depth0 = 700
    xlim = [-1.85e6, 0]
    ylim = [5e5, 3.3e6]
    lon0 = -20

    ds = xr.open_dataset(domain_cfg).squeeze()
    ocean_mask = build_ocean_mask(ds)[0]
    ice_mask = build_ice_mask(ds)[0]
    bathy = ds['bathy_metry'].where((ocean_mask==1)*(ice_mask==0))

    fig, ax = plt.subplots(figsize=(5.5,8))
    fig.tight_layout()
    ax.axis('equal')
    img = circumpolar_plot(bathy, ds, ax=ax, masked=True, title='', ctype='Blues', shade_land=True, contour=[depth0], contour_colour='DarkMagenta', make_cbar=False, contour_ice=True, lon0=lon0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    cax = inset_axes(ax, '3%', '15%', loc='lower right')
    plt.colorbar(img, cax=cax, ticklocation='left')
    finished_plot(fig, fig_name=fig_name, dpi=300)
    
    

    

    
