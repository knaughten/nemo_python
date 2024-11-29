import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import netCDF4 as nc

from ..grid import region_mask
from ..plots import circumpolar_plot, finished_plot


def find_cgrid_issues (grid_file='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA025_v3.nc'):

    from shapely.geometry import Point, Polygon

    ds = xr.open_dataset(grid_file)

    tlon = np.squeeze(ds['glamt'].values)
    tlat = np.squeeze(ds['gphit'].values)
    flon = np.squeeze(ds['glamf'].values)
    flat = np.squeeze(ds['gphif'].values)
    land = np.isnan(ds['closea_mask'].values)

    aligned = np.ones(land.shape)
    for j in range(1, ds.sizes['y']):
        for i in range(1, ds.sizes['x']-1):
            lon_corners = np.array([flon[j,i], flon[j-1,i], flon[j-1,i-1], flon[j,i-1]])
            lat_corners = np.array([flat[j,i], flat[j-1,i], flat[j-1,i-1], flat[j,i-1]])
            if tlon[j,i] < -179 and np.amax(lon_corners) > 179:
                index = lon_corners > 0
                lon_corners[index] = lon_corners[index] - 360
            elif tlon[j,i] > 179 and np.amin(lon_corners) < -179:
                index = lon_corners < 0
                lon_corners[index] = lon_corners[index] + 360
            tpoint = Point(tlon[j,i], tlat[j,i])
            grid_cell = Polygon([(lon_corners[n], lat_corners[n]) for n in range(4)])
            aligned[j,i] = tpoint.within(grid_cell)
    aligned = aligned.astype(bool)
    ocean_good = np.invert(land)*aligned
    ocean_bad = np.invert(land)*np.invert(aligned)
    land_bad = land*np.invert(aligned)

    fig, ax = plt.subplots()
    ax.plot(tlon[ocean_good], tlat[ocean_good], 'o', markersize=1, color='blue')
    ax.plot(tlon[ocean_bad], tlat[ocean_bad], 'o', markersize=1, color='red')
    ax.plot(tlon[land_bad], tlat[land_bad], 'o', markersize=1, color='green')
    ax.set_title('Misaligned cells in ocean (red) and land (green)')
    fig.savefig('misaligned_cells.png')


def plot_region_map (file_path='/gws/nopw/j04/terrafirma/kaight/input_data/grids/mesh_mask_UKESM1.1_ice.nc', option='all', 
                     legend=False, fig_name=None, halo=True):

    regions = ['amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross']
    colours = ['IndianRed', 'SandyBrown', 'LightGreen', 'MediumTurquoise', 'SteelBlue', 'Plum', 'Pink']
    lat_max = -60
    grid = xr.open_dataset(file_path).squeeze()
    if halo:
        # Drop halo
        grid = grid.isel(x=slice(1,-1))

    for n in range(len(regions)):
        print('Processing '+regions[n])
        mask, ds = region_mask(regions[n], grid, option=option)
        if halo:
            mask = mask.isel(x=slice(1,-1))
        if n==0:
            fig, ax = circumpolar_plot(mask, grid, make_cbar=False, return_fig=True, ctype=colours[n], lat_max=lat_max)
        else:
            circumpolar_plot(mask, grid, ax=ax, make_cbar=False, ctype=colours[n], lat_max=lat_max, shade_land=False)

    if legend:
        custom_lines=[]
        for colour in colours:
            custom_lines = custom_lines + [Line2D([0], [0], color=colour, lw=3)]
        ax.legend(custom_lines, regions, frameon=False, loc=(1.05, 0.5))
    finished_plot(fig, fig_name=fig_name)


def plot_bisicles_overview (base_dir='./', suite_id='dj515', fig_dir=None):

    from ..bisicles_utils import read_bisicles_all
    var_names = ['thickness', 'activeBasalThicknessSource', 'activeSurfaceThicknessSource']
    var_titles = ['Ice thickness', 'Basal mass balance', 'Surface mass balance']
    var_units = ['m', 'm/y', 'm/y']
    domains = ['AIS', 'GrIS']
    time_titles = ['first 10 years', 'last 10 years']

    # Read data
    ds_all = []
    for domain in domains:
        file_head = 'bisicles_'+suite_id+'c_1y_'
        file_tail = '_plot-'+domain+'.hdf5'
        ds_domain = read_bisicles_all(base_dir+'/'+suite_id+'/', file_head, file_tail, var_names, level=0, order=0)
        # Mask where thickness is 0
        ds_domain = ds_domain.where(ds_domain['thickness']>0)
        # Average over first 10 years and last 10 years
        ds_avg = [ds_domain.isel(time=slice(0,10)).mean(dim='time'), ds_domain.isel(time=slice(-10,None)).mean(dim='time')]
        ds_all.append(ds_avg)
    
    for var, title, units in zip(var_names, var_titles, var_units):
        fig = plt.figure(figsize=(8,8))
        gs = plt.GridSpec(2,2)
        gs.update(left=0.05, right=0.9, bottom=0.05, top=0.9, hspace=0.1, wspace=0.05)
        for n in range(len(domains)):
            vmin = np.amin([ds[var].min() for ds in ds_all[n]])
            vmax = np.amax([ds[var].max() for ds in ds_all[n]])
            for t in range(len(ds_all[n])):
                ax = plt.subplot(gs[n,t])
                img = ax.pcolormesh(ds_all[n]['x'], ds_all[n]['y'], ds_all[n][t][var], vmin=vmin, vmax=vmax)
                if n==0:
                    ax.set_title(time_titles[t], fontsize=12)
            cax = fig.add_axes([0.92, 0.05+0.45*n, 0.05, 0.3])
            plt.colorbar(img, cax=cax)
        plt.suptitle(title+' ('+units+')', fontsize=16)
        if fig_dir is None:
            fig_name = None
        else:
            fig_name = fig_dir+'/'+var+'.png'
        finished_plot(fig, fig_name=fig_name)
            
            
            
            
        



    

    
