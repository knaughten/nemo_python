import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import netCDF4 as nc
import os

from ..grid import region_mask
from ..plots import circumpolar_plot, finished_plot
from ..plot_utils import set_colours
from ..constants import months_per_year
from ..utils import add_months


def find_cgrid_issues (grid_file='/gws/ssde/j25b/terrafirma/kaight/input_data/grids/domcfg_eORCA025_v3.nc'):

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


def plot_region_map (file_path='/gws/ssde/j25b/terrafirma/kaight/input_data/grids/mesh_mask_UKESM1.1_ice.nc', option='all', 
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
    time_titles = ['first year', 'after 10 years', 'last year']
    ctypes = ['viridis', 'plusminus_r', 'plusminus_r']

    # Read data
    ds_2D = []
    ds_ts = []
    for domain in domains:
        file_head = 'bisicles_'+suite_id+'c_' #1y_'
        file_tail = '_plot-'+domain+'.hdf5'
        ds_domain = read_bisicles_all(base_dir+'/'+suite_id+'/', file_head, file_tail, var_names, level=0, order=0)
        # Mask where initial thickness is 0
        ds_domain = ds_domain.where(ds_domain['thickness'].isel(time=0)>0)
        # Save first year, 10 years in, and last year in 2D
        ds_avg = [ds_domain.isel(time=0), ds_domain.isel(time=10), ds_domain.isel(time=-1)]
        ds_2D.append(ds_avg)
        # Save timeseries
        ds_ts.append(ds_domain.mean(dim=['x','y']))

    # Plot maps of individual years for each variable
    for var, title, units, ctype in zip(var_names, var_titles, var_units, ctypes):
        fig = plt.figure(figsize=(10,6))
        gs = plt.GridSpec(len(domains),len(time_titles))
        gs.update(left=0.03, right=0.88, bottom=0.05, top=0.9, hspace=0.1, wspace=0.05)
        for n in range(len(domains)):
            vmin = np.amin([ds[var].min() for ds in ds_2D[n]])
            vmax = np.amax([ds[var].max() for ds in ds_2D[n]])
            if vmin == vmax:
                continue
            if var == 'activeBasalThicknessSource':
                vmin = -10
                extend = 'min'
            else:
                extend = 'neither'
            cmap = set_colours(ds_2D[n][0][var], ctype=ctype, vmin=vmin, vmax=vmax)[0]
            for t in range(len(ds_2D[n])):
                ax = plt.subplot(gs[n,t])
                img = ax.pcolormesh(ds_2D[n][t]['x'], ds_2D[n][t]['y'], ds_2D[n][t][var], vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('equal')
                if n==0:
                    ax.set_title(time_titles[t], fontsize=12)
            cax = fig.add_axes([0.9, 0.1+0.45*(1-n), 0.03, 0.3])
            plt.colorbar(img, cax=cax, extend=extend)
        plt.suptitle(title+' ('+units+')', fontsize=16)
        if fig_dir is None:
            fig_name = None
        else:
            fig_name = fig_dir+'/'+var+'.png'
        finished_plot(fig, fig_name=fig_name)

    # Plot timeseries of area-mean all variables
    fig = plt.figure(figsize=(12,6))
    gs = plt.GridSpec(len(domains), len(var_names))
    gs.update(left=0.06, right=0.99, bottom=0.05, top=0.9, hspace=0.4, wspace=0.3)
    for n in range(len(domains)):
        for v in range(len(var_names)):
            ax = plt.subplot(gs[n,v])
            ax.plot_date(ds_ts[n]['time'], ds_ts[n][var_names[v]], '-')
            ax.grid(linestyle='dotted')
            ax.set_title(var_titles[v]+'\n('+domains[n]+'), '+var_units[v], fontsize=12)
    if fig_dir is None:
        fig_name = None
    else:
        fig_name = fig_dir+'/bisicles_timeseries.png'
    finished_plot(fig, fig_name=fig_name)


# Precompute decadal averages for plot_cice_decades and save to NetCDF.
def precompute_cice_decades ():

    suite = 'dl286' #'dj515'
    num_decades = 3
    months_per_decade = months_per_year*10
    start_year = 1995 #1982
    decade_titles = [str(start_year+n*10)+'-'+str(start_year+(n+1)*10) for n in range(num_decades)]

    for n in range(num_decades):
        print('Processing '+decade_titles[n])
        ds_accum = None
        num_months = 0
        for year in range(start_year+n*10, start_year+(n+1)*10):
            for month in range(1, months_per_year+1):
                next_year, next_month = add_months(year, month, 1)
                file_path = suite+'/cice_'+suite+'i_1m_'+str(year)+str(month).zfill(2)+'01-'+str(next_year)+str(next_month).zfill(2)+'01.nc'
                print('...'+file_path)
                ds = xr.open_dataset(file_path)
                if ds_accum is None:
                    ds_accum = ds.mean(dim='time')
                else:
                    ds_accum += ds.mean(dim='time')
                ds.close()
                num_months += 1
        ds_accum /= num_months
        out_file = suite+'/cice_'+decade_titles[n]+'_avg.nc'
        print('...writing '+out_file)
        ds_accum.to_netcdf(out_file)    


# Plot a single CICE 2D variable for each decade in the wonky dj515 simulation, showing Arctic and Antarctic projections.
def plot_cice_decades (var, fig_name=None, ctype='viridis', vmin=None, vmax=None):

    suite = 'dl286' #'dj515'
    start_year = 1995 #1982

    # Read variable attributes from single file - they somehow got lost in precomputing above.
    ds = xr.open_dataset(suite+'/cice_'+suite+'i_1m_'+str(start_year)+'0101-'+str(start_year)+'0201.nc')
    title = ds[var].long_name+' ('+ds[var].units+')'
    ds.close()

    file_names = []
    for f in os.listdir(suite):
        if f.startswith('cice_') and f.endswith('_avg.nc'):
            file_names.append(f)
    file_names.sort()
    num_decades = len(file_names)

    data_avg = []
    decade_titles = []
    for n in range(num_decades):
        ds = xr.open_dataset(suite+'/'+file_names[n])
        data_avg.append(ds[var])
        decade_titles.append(file_names[n][file_names[n].index('cice_')+len('cice_'):file_names[n].index('_avg.nc')])        
    # Get global vmin and vmax
    if vmin is None:
        vmin = np.amin([data.min() for data in data_avg])
    if vmax is None:
        vmax = np.amax([data.max() for data in data_avg])
    
    fig = plt.figure(figsize=(7,6))
    gs = plt.GridSpec(2, num_decades)
    gs.update(left=0.05, right=0.95, bottom=0.1, top=0.85, hspace=0.2, wspace=0.05)
    for n in range(num_decades):
        for row, pole in zip(range(2), ['N', 'S']):
            ax = plt.subplot(gs[row,n])
            img = circumpolar_plot(data_avg[n], ds, pole=pole, cice=True, ax=ax, make_cbar=False, title=decade_titles[n], titlesize=14, vmin=vmin, vmax=vmax, ctype=ctype)
            ax.axis('on')
            ax.set_xticks([])
            ax.set_yticks([])
        if n == num_decades-1:
            cax = fig.add_axes([0.3, 0.04, 0.4, 0.03])
            plt.colorbar(img, cax=cax, orientation='horizontal')
        plt.suptitle(title, fontsize=18)
    finished_plot(fig, fig_name=fig_name)
            
        
    
    
            
            
            
            
        



    

    
