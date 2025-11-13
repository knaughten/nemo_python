import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import cmocean
import os
import gsw
from ..utils import select_bottom, distance_along_transect, moving_average, polar_stereo
from ..constants import deg_string, gkg_string, transect_amundsen, months_per_year, region_names, adusumilli_melt, adusumilli_std, transport_obs, transport_std, region_edges, rEarth, deg2rad, zhou_TS, zhou_TS_std
from ..plots import circumpolar_plot, finished_plot, plot_ts_distribution, plot_transect
from ..interpolation import interp_latlon_cf, interp_latlon_cf_blocks
from ..file_io import read_schmidtko, read_woa, read_dutrieux, read_zhou
from ..grid import extract_var_region, transect_coords_from_latlon_waypoints, region_mask, build_shelf_mask
from ..timeseries import update_simulation_timeseries, overwrite_file

# Compare the bottom temperature and salinity in NEMO (time-averaged over the given xarray Dataset) to observations: Schmidtko on the continental shelf, World Ocean Atlas 2018 in the deep ocean.
def bottom_TS_vs_obs (nemo, time_ave=True,
                      schmidtko_file='/gws/nopw/j04/terrafirma/kaight/input_data/schmidtko_TS.txt', 
                      woa_files='/gws/nopw/j04/terrafirma/kaight/input_data/WOA18/woa18_decav_*00_04.nc', 
                      nemo_mesh='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20250715.nc',
                      fig_name=None, amundsen=False, dpi=None, return_fig=False):

    obs = read_schmidtko(schmidtko_file=schmidtko_file, eos='teos10')
    woa = read_woa(woa_files=woa_files, eos='teos10')

    # Regrid to the NEMO grid
    obs_interp = interp_latlon_cf(obs, nemo, method='bilinear')
    woa_interp = interp_latlon_cf(woa, nemo, method='bilinear')
    # Now combine them, giving precedence to the Schmidtko obs where both datasets exist
    obs_plot = xr.where(obs_interp.isnull(), woa_interp, obs_interp)

    # Select the NEMO variables we need and time-average
    if time_ave:
        nemo_plot = xr.Dataset({'temp':nemo['sbt'], 'salt':nemo['sbs']}).mean(dim='time_counter')
    else:
        nemo_plot = xr.Dataset({'temp':nemo['sbt'], 'salt':nemo['sbs']})
    nemo_plot = nemo_plot.rename({'x_grid_T_inner':'x', 'y_grid_T_inner':'y'})
    # Apply NEMO land mask to both
    nemo_plot = nemo_plot.where(nemo_plot['temp']!=0)
    obs_plot = obs_plot.where(nemo_plot['temp'].notnull()*obs_plot.notnull())
    obs_plot = obs_plot.where(nemo_plot['temp']!=0)
    nemo_plot = nemo_plot.where(nemo_plot['temp']!=0)
    # Get difference from obs
    bias = nemo_plot - obs_plot

    if amundsen:
       import cartopy.crs as ccrs
    
       nemo_mesh_ds = xr.open_dataset(nemo_mesh)
       # These indices are based on eANT025; eventually should generalize based on lat, lon
       mesh_sub  = nemo_mesh_ds.isel(x=slice(450, 900), y=slice(130,350), time_counter=0)
       nemo_plt  = nemo_plot.isel(x=slice(450, 900), y=slice(130,350))
       obs_plt   = obs_plot.isel(x=slice(450, 900), y=slice(130,350))
       bias_plt  = bias.isel(x=slice(450, 900), y=slice(130,350))
       # Little helper function to help cartopy with landmasking
       def mask_land(nemo_mesh, file_var):
          lon_plot = np.ma.masked_where(mesh_sub.tmask.isel(nav_lev=0) == 0, mesh_sub.nav_lon.values)
          lat_plot = np.ma.masked_where(mesh_sub.tmask.isel(nav_lev=0) == 0, mesh_sub.nav_lat.values)
          plot_var = np.ma.masked_where(mesh_sub.tmask.isel(nav_lev=0) == 0, file_var.values)
          return lon_plot, lat_plot, plot_var 
      
       data_plot  = [nemo_plt, obs_plt, bias_plt]
       var_titles = ['Bottom temperature ('+deg_string+'C)', 'Bottom salinity ('+gkg_string+')']
       vmin = [-2, -2, -1, 34.2, 34.2, -0.4]
       vmax = [2, 2, 1, 35, 35, 0.4]

       # fig, ax = plt.subplots(2,3, figsize=(20,8), subplot_kw={'projection': ccrs.Mercator(latitude_true_scale=-70)})
       fig, ax = plt.subplots(2,3, figsize=(15,6), subplot_kw={'projection': ccrs.Mercator(latitude_true_scale=-70)}, dpi=dpi)

       for axis in ax.ravel():
          axis.set_extent([-95, -135, -76, -68], ccrs.PlateCarree())
          # axis.set_extent([-95, -160, -78, -67], ccrs.PlateCarree())
          gl = axis.gridlines(draw_labels=True);
          gl.xlines=None; gl.ylines=None; gl.top_labels=None; gl.right_labels=None;

       i=0
       for v, var in enumerate(['temp', 'salt']):
          for n, name in enumerate(['Model', 'Observations', 'Model bias']):
             lon_plt, lat_plt, var_plt = mask_land(mesh_sub, data_plot[n][var])
             img = ax[v,n].pcolormesh(lon_plt, lat_plt, var_plt, transform=ccrs.PlateCarree(), rasterized=True, cmap='RdBu_r', vmin=vmin[i], vmax=vmax[i])
             #ax[v,n].set_title(name)
             i+=1
             if n != 1:
                cax = fig.add_axes([0.04+0.44*n, 0.56-0.41*v, 0.02, 0.3])
                plt.colorbar(img, cax=cax, extend='both', label=var_titles[v])
       if return_fig:
           return fig, ax
       else:
           finished_plot(fig, fig_name=fig_name, dpi=dpi)
    else:
       # Make the plot
       fig = plt.figure(figsize=(10,7))
       gs = plt.GridSpec(2,3)
       gs.update(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.2, wspace=0.1)
       data_plot = [nemo_plot, obs_plot, bias]
       var_plot = ['temp', 'salt']
       var_titles = ['Bottom temperature ('+deg_string+'C)', 'Bottom salinity ('+gkg_string+')']
       alt_titles = [None, 'Observations', 'Model bias']
       vmin = [-2, -2, -0.5, 34.5, 34.5, -0.2]
       vmax = [2, 2, 0.5, 35, 35, 0.2]
       ctype = ['RdBu_r', 'RdBu_r', 'plusminus']
       i=0
       for v in range(2):
           for n in range(3):
               ax = plt.subplot(gs[v,n])
               ax.axis('equal')
               img = circumpolar_plot(data_plot[n][var_plot[v]], nemo, ax=ax, masked=True, make_cbar=False, 
                                      title=(var_titles[v] if n==0 else alt_titles[n]), 
                                      vmin=vmin[i], vmax=vmax[i], ctype=ctype[n], shade_land=False)
               i+=1
               if n != 1:
                   cax = fig.add_axes([0.01+0.46*n, 0.58-0.48*v, 0.02, 0.3])
                   plt.colorbar(img, cax=cax, extend='both' if n==0 else 'neither')
       finished_plot(fig, fig_name=fig_name, dpi=dpi)
          
# 4-panel evaluation plot of Barotropic streamfunction, winter mixed-layer depth, bottom T, bottom S as in Fig.1, Holland et al. 2014
def circumpolar_Holland_tetraptych(run_folder, nemo_domain='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/domain_cfg-20250715.nc',
                                   nemo_mesh='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20250715.nc',
                                   fig_name=None, dpi=None):

    from ..diagnostics import barotropic_streamfunction

    # Load NEMO gridT files for MLD and bottom T, S
    gridT_files = glob.glob(f'{run_folder}*grid_T*')
    nemo_ds     = xr.open_mfdataset(gridT_files) # load all the gridT files in the run folder
    nemo_grid   = xr.open_dataset(gridT_files[0]) # for plotting for later 
    nemo_ds = nemo_ds.rename({'e3t':'thkcello', 'x_grid_T':'x', 'y_grid_T':'y', 'e3t':'thkcello',
                              'nav_lon_grid_T':'nav_lon', 'nav_lat_grid_T':'nav_lat'})

    # Calculate the average winter (June, July, August) mixed layer depth    
    dates_month = nemo_ds.time_counter.dt.month
    nemo_winter = nemo_ds.isel(time_counter=((dates_month==6) | (dates_month==7) | (dates_month==8)))
    MLD_winter  = nemo_winter['mldr10_1'].mean(dim='time_counter')

    # Calculate the average of bottom temperature and salinity over the full time series and mask land
    nemo_plot = xr.Dataset({'temp':nemo_ds['sbt'], 'salt':nemo_ds['sbs']}).mean(dim='time_counter')
    nemo_plot = nemo_plot.assign({'MLD':MLD_winter}).rename({'x_grid_T_inner':'x', 'y_grid_T_inner':'y'})
    nemo_plot = nemo_plot.where(nemo_plot['temp']!=0)

    # Mask out anything beyond region of interest, plus ice shelf cavities for the barotropic streamfunction
    def apply_mask(data, nemo_mesh, mask_shallow=False):
       mesh_file = xr.open_dataset(nemo_mesh).isel(time_counter=0)
    
       data = data.where(mesh_file.misf!=0) # mask ice shelf
       if mask_shallow:
           # Also mask anything shallower than 500m
           data = data.where(mesh_file.bathy_metry >= 500)
       return data

    # Load velocity files for barotropic streamfunction calculation
    gridU_files = glob.glob(f'{run_folder}*grid_U*')
    gridV_files = glob.glob(f'{run_folder}*grid_V*')
    ds_u = xr.open_mfdataset(gridU_files, chunks='auto').squeeze().rename({'e3u':'thkcello'})[['uo','thkcello']]
    ds_v = xr.open_mfdataset(gridV_files, chunks='auto').squeeze().rename({'e3v':'thkcello'})[['vo','thkcello']]
    # Calculate barotropic streamfunction and average over the full time series
    ds_domcfg = xr.open_dataset(nemo_domain).isel(time_counter=0)
    strf = barotropic_streamfunction(ds_u, ds_v, ds_domcfg, periodic=True, halo=True)
    strf_masked = apply_mask(strf, nemo_mesh, mask_shallow=False)
    strf_mean   = strf_masked.mean(dim='time_counter')

    # create figure
    plot_vars = [strf_mean, nemo_plot['MLD'], nemo_plot['temp'], nemo_plot['salt']]
    titles    = ['Barotropic streamfunction (Sv)', 'Winter mixed-layer depth (m)', 'Bottom Temp. (C)', 'Bottom Sal. (g/kg)']
    vlims     = [(-50,150), (0,300), (-2,2), (34.5, 35.0)]

    fig, ax = plt.subplots(2,2, figsize=(10,10), dpi=dpi)
    args = {'masked':True, 'make_cbar':False, 'ctype':cmocean.cm.balance, 'shade_land':False}
    for i, axis in enumerate(ax.ravel()):
       img = circumpolar_plot(plot_vars[i], nemo_grid, ax=axis, title=titles[i], contour=0, vmin=vlims[i][0], vmax=vlims[i][1], **args)
       if i<2:
          cax = fig.add_axes([0.49+0.43*i, 0.55, 0.02, 0.3])
       else:
          cax = fig.add_axes([0.49+0.43*(i-2), 0.14, 0.02, 0.3])
       plt.colorbar(img, cax=cax, extend='both')

    finished_plot(fig, fig_name=fig_name, dpi=dpi)
    
    return


# Compare temperature and salinity in a depth range in NEMO (time-averaged over the given xarray Dataset) to observations: 
# Specifically, Shenji Zhou's 2024 dataset
def circumpolar_TS_vs_obs (nemo, depth_min, depth_max, nemo_mesh='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20250715.nc', fig_name=None, dpi=None):

    # depth is the depth to look at (could be a slice):
    obs      = read_zhou()
    obs      = obs.where((abs(obs.depth) > depth_min) * (abs(obs.depth) <= depth_max)).mean(dim='z')
    obs_zhou = obs.drop_vars(['lat', 'lon', 'pressure', 'depth']).rename_dims({'x':'lon', 'y':'lat'})
    obs_zhou = obs_zhou.assign_coords({'lon':obs.lon.isel(y=0).values,'lat':obs.lat.isel(x=0).values}).transpose() 
    del obs
    # Regrid to the NEMO grid
    print('Interpolating Zhou 2024 dataset to grid')
    nemo_mesh = xr.open_dataset(nemo_mesh)
    obs_zhou_interp    = interp_latlon_cf_blocks(obs_zhou, nemo_mesh, method='bilinear', pster_src=False, periodic_nemo=False)
    # Now combine them, giving precedence to Shenji's dataset, then Schmidtko
    obs_plot = obs_zhou

    # Select the NEMO variables we need and time-average
    nemo_plot = xr.Dataset({'ConsTemp':nemo['thetao'], 'AbsSal':nemo['so']})
    nemo_plot = nemo_plot.where((nemo_plot.deptht > depth_min) * (nemo_plot.deptht <= depth_max)).mean(dim='deptht')
    nemo_plot = nemo_plot.where(nemo_plot['ConsTemp']!=0) # Apply NEMO land mask to both

    obs_plot
    obs_plot  = obs_plot.where(nemo_plot['ConsTemp'].notnull()*obs_plot.notnull())
    obs_plot  = obs_plot.where(nemo_plot['ConsTemp']!=0)
    nemo_plot = nemo_plot.where(nemo_plot['ConsTemp']!=0)
    # Get difference from obs
    bias = nemo_plot - obs_plot

    
    print('Creating figure')
   # Make the plot
    fig = plt.figure(figsize=(10,7))
    gs = plt.GridSpec(2,3)
    gs.update(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.2, wspace=0.1)
    data_plot = [nemo_plot, obs_plot, bias]
    var_plot = ['ConsTemp', 'AbsSal']
    var_titles = ['Bottom temperature ('+deg_string+'C)', 'Bottom salinity ('+gkg_string+')']
    alt_titles = [None, 'Observations', 'Model bias']
    vmin = [-2, -2, -0.5, 34.5, 34.5, -0.2]
    vmax = [2, 2, 0.5, 35, 35, 0.2]
    ctype = ['RdBu_r', 'RdBu_r', 'plusminus']
    i=0
    for v in range(2):
        for n in range(3):
            ax = plt.subplot(gs[v,n])
            ax.axis('equal')
            img = circumpolar_plot(data_plot[n][var_plot[v]], nemo, ax=ax, masked=True, make_cbar=False, 
                                   title=(var_titles[v] if n==0 else alt_titles[n]), 
                                   vmin=vmin[i], vmax=vmax[i], ctype=ctype[n], shade_land=False)
            i+=1
            if n != 1:
                cax = fig.add_axes([0.01+0.46*n, 0.58-0.48*v, 0.02, 0.3])
                plt.colorbar(img, cax=cax, extend='both' if n==0 else 'neither')
    
    finished_plot(fig, fig_name=fig_name, dpi=dpi)

# Helper function to mask the temperature and salinity from the simulation outputs for regional_profile_TS_std
# Function is very similar to extract_var_region in grid.py and could probably be replaced by it with a bit of adjustment
# Inputs
# gridT_files : list of NEMO simulation gridT files
# mask        : xarray dataset containing a mask to extract the specified region 
# Returns xarray DataArrays of temperature and salinity with NaNs everywhere except the region of
def mask_sim_region(gridT_files, mask, region_subsetx=slice(0,None), region_subsety=slice(0,None)):

    # load all the gridT files in the run folder
    nemo_ds     = xr.open_mfdataset(gridT_files)
    nemo_ds = nemo_ds.rename({'x_grid_T':'x', 'y_grid_T':'y', 'nav_lon_grid_T':'nav_lon', 'nav_lat_grid_T':'nav_lat'})
    dates_month  = nemo_ds.time_counter.dt.month
    nemo_ds      = nemo_ds.isel(time_counter=((dates_month==1) | (dates_month==2))) # select only January and February

    # Average full time series: ## average only over January and February of each year
    nemo_T = nemo_ds.thetao.isel(x=region_subsetx, y=region_subsety)
    nemo_S = nemo_ds.so.isel(x=region_subsetx, y=region_subsety)

    # region masked: fill regions outside of the mask with NaN and replace zeros with NaN for averaging
    nemo_T_masked = xr.where((mask!=0)*(nemo_T!=0), nemo_T, np.nan)
    nemo_S_masked = xr.where((mask!=0)*(nemo_S!=0), nemo_S, np.nan)

    return nemo_T_masked, nemo_S_masked

# Helper function to mask temperature and salinity from observations for regional_profile_TS_std
# Inputs
# fileT : list of temperature observation files
# fileS : list of salinity observation files
# mask  : xarray dataset containing a mask to extract the specified region 
# nemo_domcfg : string of path to NEMO domain_cfg file
# Returns xarray DataArrays of temperature and salinity with NaNs everywhere except the region of interest
def mask_obs_region(fileT, fileS, mask, nemo_domcfg='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/domain_cfg-20250715.nc'):

    nemo_file = xr.open_dataset(nemo_domcfg).squeeze()

    #  set region limits based on min and max value in mask (not the exact solution but ok for now):
    mask_lon_min = (mask*nemo_file.nav_lon).min().values
    mask_lon_max = xr.where(mask*nemo_file.nav_lon !=0, mask*nemo_file.nav_lon, np.nan).max().values
    mask_lat_min = (mask*nemo_file.nav_lat).min().values
    mask_lat_max = xr.where(mask*nemo_file.nav_lat !=0, mask*nemo_file.nav_lat, np.nan).max().values

    #  load observations
    if type(fileT) == list:
        i=0
        for fT, fS in zip(fileT, fileS):
            if i==0:
                obs = read_dutrieux(eos='teos10', fileT=fT, fileS=fS)
            else:
                obs_new = read_dutrieux(eos='teos10', fileT=fT, fileS=fS)
                obs = xr.concat([obs, obs_new], 'year')
            i+=1
    else:
        obs = read_dutrieux(eos='teos10', fileT=fileT, fileS=fileS)
    array_mask   = (obs.lon >= mask_lon_min)*(obs.lon <= mask_lon_max)*(obs.lat >= mask_lat_min)*(obs.lat <= mask_lat_max)
    # mask observations:
    obs_T_masked = xr.where(array_mask, obs.ConsTemp, np.nan)
    obs_S_masked = xr.where(array_mask, obs.AbsSal, np.nan)

    return obs_T_masked, obs_S_masked
    
# Function to plot vertical profiles of regional mean annual temperature, salinity, and their standard deviations from observations and simulations
# Inputs
# run_folder             : string path to NEMO simulation run folder with grid_T files
# region                 : string of the region to calculate the profiles for (from one of the region_names in constants.py)
# option      (optional) : string specifying whether to calculate averages over continental shelf region, 'shelf', 'cavity', or 'all'
# conf        (optional) : string of name of configuration; used to subset grid to the Amundsen region when specifying eANT025 (messy)
# fig_name    (optional) : string of path to save figure to
# dir_obs     (optional) : string of path to observation directory
# nemo_domcfg (optional) : string of path to NEMO domain_cfg file 
# years_show             : which years to show, defaults to only show simulated profiles during overlap with obs, otherwise pass a list or range of the years to include
def regional_profile_TS_std(run_folder, region, option='shelf', fig_name=None, dpi=None, conf=None,
                            dir_obs='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/observations/pierre-dutrieux/',
                            nemo_domcfg='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/domain_cfg-20250715.nc',
                            years_show=np.arange(2000,2020)):
    from datetime import datetime

    # Get NEMO domain grid and mask for the specified region
    nemo_file = xr.open_dataset(nemo_domcfg).squeeze()
    mask, _, region_name = region_mask(region, nemo_file, option=option, return_name=True)
    # not the most elegant, but to speed up the calculations, subset x and y grid to the Amundsen Sea region. Specific to configuration/region:
    if conf=='eANT025':
        region_subsetx = slice(450,850); region_subsety = slice(140,300);
    else:
    	region_subsetx = slice(0,None); region_subsety = slice(0,None);  
    mask_subset = mask.isel(x=region_subsetx, y=region_subsety)

    # Find list of observations and simulation files
    # observations span 1994-2019, with mostly 2000-2019, so only take simulation files between 2000-2019 onwards
    yearly_Tobs  = glob.glob(f'{dir_obs}ASEctd_griddedMean????_PT.nc')
    yearly_Sobs  = glob.glob(f'{dir_obs}ASEctd_griddedMean????_S.nc')
    file_list    = glob.glob(f'{run_folder}*1m*grid_T*')
    yearly_TSsim = []
    for file in file_list:
        year = datetime.strptime(file.split('1m_')[1].split('_')[0], '%Y%m%d').year
        if year in years_show:
            yearly_TSsim = yearly_TSsim + [file]

    #----------- Figure ----------
    fig, ax = plt.subplots(1,4, figsize=(12,6), dpi=dpi, gridspec_kw={'width_ratios': [2, 1, 2, 1]})

    fig.suptitle(f'{region_name}', fontweight='bold')
    ax[0].set_ylabel('Depth (m)')
    titles = ['Conservative Temperature (C)', 'std', 'Absolute Salinity (g/kg)', 'std']
    for i, axis in enumerate(ax.ravel()):
        axis.set_ylim(1000, 0)
        if i!=0:
            axis.yaxis.set_ticklabels([])
        axis.set_title(titles[i])
        axis.xaxis.grid(True, which='major', linestyle='dotted')
        axis.yaxis.grid(True, which='major', linestyle='dotted')

    # Yearly profiles of model simulations:
    for file in yearly_TSsim:
        sim_T, sim_S = mask_sim_region(file, mask_subset, region_subsetx=region_subsetx, region_subsety=region_subsety)
        ax[0].plot(sim_T.mean(dim=['x','y','time_counter']), sim_T.deptht, '-k', linewidth=0.3)
        ax[2].plot(sim_S.mean(dim=['x','y','time_counter']), sim_S.deptht, '-k', linewidth=0.3)
    # mean over all the years:
    sim_T, sim_S = mask_sim_region(yearly_TSsim, mask_subset, region_subsetx=region_subsetx, region_subsety=region_subsety)
    ax[0].plot(sim_T.mean(dim=['x','y','time_counter']), sim_T.deptht, '-k', linewidth=2.5, label='Model')
    ax[2].plot(sim_S.mean(dim=['x','y','time_counter']), sim_T.deptht, '-k', linewidth=2.5)
    # standard deviation
    ax[1].plot(sim_T.mean(dim=['x','y']).std(dim='time_counter'), sim_T.deptht, '-k')
    ax[3].plot(sim_S.mean(dim=['x','y']).std(dim='time_counter'), sim_S.deptht, '-k')

    # Yearly profiles of observations:
    for obsT, obsS in zip(yearly_Tobs, yearly_Sobs):
        obs_T, obs_S = mask_obs_region(obsT, obsS, mask, nemo_domcfg=nemo_domcfg)
        ax[0].plot(obs_T.mean(dim=['lon','lat']), abs(obs_T.depth), '--c', linewidth=0.5)
        ax[2].plot(obs_S.mean(dim=['lon','lat']), abs(obs_S.depth), '--c', linewidth=0.5)
    # mean over all the years 
    obs_T, obs_S = mask_obs_region(f'{dir_obs}ASEctd_griddedMean_PT.nc', f'{dir_obs}ASEctd_griddedMean_S.nc', mask, nemo_domcfg=nemo_domcfg)
    ax[0].plot(obs_T.mean(dim=['lon','lat']), abs(obs_T.depth), '--c', linewidth=2.5, label='Observations')
    ax[2].plot(obs_S.mean(dim=['lon','lat']), abs(obs_S.depth), '--c', linewidth=2.5)
    # standard deviation
    print('Calculating standard dev. obs')
    obs_T_yearly, obs_S_yearly = mask_obs_region(yearly_Tobs, yearly_Sobs, mask, nemo_domcfg=nemo_domcfg)
    ax[1].plot(obs_T_yearly.mean(dim=['lon','lat']).std(dim='year'), abs(obs_T.depth), '-c')
    ax[3].plot(obs_S_yearly.mean(dim=['lon','lat']).std(dim='year'), abs(obs_S.depth), '-c')

    ax[0].legend(frameon=False)

    finished_plot(fig, fig_name=fig_name, dpi=dpi)

    return

# Function creates a figure with T-S diagram for simulations and for Pierre's observations in the Amundsen Sea
# Inputs:
# run_folder : string path to simulation folder
# show_obs   : (optional) boolean for whether to plot observations as well
# file_ind   : (optional) index of file to read
# time_slice : (optional) slice to subset time_counter for averaging simulation
# depth_slice: (optional) slice to subset deptht from simulation
# fig_name   : (optional) string for path to save figure if you want to save it
# return_fig : (optional) boolean for returning fig and ax
def TS_diagrams_Amundsen (run_folder, show_obs=True, file_ind=None, time_slice=None, depth_slice=None, fig_name=None, return_fig=False, smin=30, smax=35.25, tmin=-3, tmax=2.25, nbins=150):
    # --- get data ----
    
    if show_obs:
        # load observations
        obs = read_dutrieux(eos='teos10')
    
    # load nemo simulations
    gridT_files = glob.glob(f'{run_folder}*grid_T*') # load all the gridT files in the run folder
    if file_ind:
        nemo_ds = xr.open_dataset(gridT_files[file_ind]).rename({'x_grid_T':'x','y_grid_T':'y'})
    else: 
        nemo_ds = xr.open_mfdataset(gridT_files).rename({'x_grid_T':'x','y_grid_T':'y'}) 
    if time_slice:
        nemo_average = nemo_ds.isel(time_counter=time_slice).mean(dim='time_counter') 
    else:
        nemo_average = nemo_ds.mean(dim='time_counter')
    # extract specific region
    amundsen_so = extract_var_region(nemo_average, 'so'    , 'amundsen_sea')
    amundsen_to = extract_var_region(nemo_average, 'thetao', 'amundsen_sea')
    if depth_slice:
        amundsen_so = amundsen_so.isel(deptht=depth_slice)
        amundsen_to = amundsen_to.isel(deptht=depth_slice)

    # --- plot distributions -----
    if not show_obs:
        fig, ax = plt.subplots(1,1,figsize=(9,7), dpi=300)
        axis = ax
    else:
        fig, ax = plt.subplots(1,2,figsize=(18,7), dpi=300)
        axis = ax[0]
        ax[1].set_title('Amundsen Sea observations Pierre')
        plot_ts_distribution(ax[1], obs.AbsSal.values.flatten(), obs.ConsTemp.values.flatten(), plot_density=True, plot_freeze=True, smin=smin, smax=smax, tmin=tmin, tmax=tmax)

    axis.set_title('Amundsen Sea simulations')
    plot_ts_distribution(axis, amundsen_so.values.flatten(), amundsen_to.values.flatten(), plot_density=True, plot_freeze=True, bins=nbins, smin=smin, smax=smax, tmin=tmin, tmax=tmax)

    if fig_name:
        finished_plot(fig, fig_name=fig_name)
    if return_fig:
        return fig, ax
    else:
        return

# Function produces animation of transects of the Amundsen Sea shelf (with constant observation panels)
def animate_transect(run_folder, loc='shelf_west'):
    
    import tqdm

    gridT_files  = glob.glob(f'{run_folder}*grid_T*')
    nemo_ds      = xr.open_mfdataset(gridT_files)
    for t, time in enumerate(nemo_ds.time_counter):
        print(t)
        year  = time.dt.year.values
        month = time.dt.month.values
        transects_Amundsen(run_folder, transect_locations=[loc], time_slice=t, savefig=True, 
                           fig_name=f'{run_folder}animations/frames/transect_{loc}_y{year}m{month:02}.jpg')

    return

# not yet generalized for other domains
def frames_transect_Amundsen_sims(run_folder, savefig=False, transect_location='shelf_west', add_rho=False, clevels=10, 
                                  smin=33.8, smax=34.9, tmin=0.2, tmax=1.0, fig_name='',
                                  nemo_mesh='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20250715.nc'):
    import warnings
    import gsw

    gridT_files  = glob.glob(f'{run_folder}*grid_T*')
    nemo_ds      = xr.open_mfdataset(gridT_files, engine='netcdf4').isel(x_grid_T=slice(580, 790), y_grid_T=slice(200,300), time_counter=slice(0,365))
    nemo_ds      = nemo_ds.rename({'x_grid_T':'x', 'y_grid_T':'y', 'nav_lon_grid_T':'nav_lon', 'nav_lat_grid_T':'nav_lat', 'deptht':'depth'}) 
    if add_rho:
        sigma        = gsw.density.sigma0(nemo_ds.so, nemo_ds.thetao)
        nemo_ds      = nemo_ds.assign({'sigma0':sigma})
        contour_var  = 'sigma0'
    else:
        contour_var  = '' 

    print(transect_amundsen[transect_location])
    print(nemo_ds.isel(time_counter=0))
    x_sim, y_sim = transect_coords_from_latlon_waypoints(nemo_ds.isel(time_counter=0), transect_amundsen[transect_location], opt_float=False)
    print(x_sim)
    sim_transect = nemo_ds.isel(x=xr.DataArray(x_sim, dims='n'), y=xr.DataArray(y_sim, dims='n'), time_counter=0)
    nemo_mesh_ds = xr.open_dataset(nemo_mesh).isel(time_counter=0,x=slice(580, 790),y=slice(200,300))
    nemomesh_tr  = nemo_mesh_ds.isel(x=xr.DataArray(x_sim, dims='n'), y=xr.DataArray(y_sim, dims='n')).rename({'nav_lev':'depth'})
    print('sim', sim_transect)
    # add tmask, iceshelfmask and depths to the simulation dataset
    sim_transect = sim_transect.assign({'gdept_0':nemomesh_tr.gdept_0, 'tmask':nemomesh_tr.tmask, 'isfdraft':nemomesh_tr.isfdraft})
    sim_distance = distance_along_transect(sim_transect)
    print('distance', sim_distance)
    
    for time in nemo_ds.time_counter:
        sim_transect = nemo_ds.isel(x=xr.DataArray(x_sim, dims='n'), y=xr.DataArray(y_sim, dims='n')).sel(time_counter=time)
        sim_transect = sim_transect.assign({'gdept_0':nemomesh_tr.gdept_0, 'tmask':nemomesh_tr.tmask, 'isfdraft':nemomesh_tr.isfdraft})

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                "The input coordinates to pcolormesh are interpreted as cell centers, but "
                "are not monotonically increasing or decreasing. This may lead to "
                "incorrectly calculated cell edges, in which case, please supply explicit "
                "cell edges to pcolormesh.",
                UserWarning,
            )

            fig, ax = plt.subplots(1,2, figsize=(14,4), dpi=125)
            kwagsT    ={'vmin':tmin,'vmax':tmax,'cmap':cmocean.cm.dense,'label':'Conservative Temp.','ylim':(1300, -10)}
            kwagsS    ={'vmin':smin,'vmax':smax,'cmap':cmocean.cm.haline,'label':'Absolute Salinity','ylim':(1300, -10)}
            kwagsrho  = {'clevels':clevels, 'contour_var':contour_var}
            kwags_mask={'mask_land':True, 'mask_iceshelf':True}
            plot_transect(ax[0], sim_distance, sim_transect, 'thetao', **kwagsT, **kwagsrho, **kwags_mask)
            plot_transect(ax[1], sim_distance, sim_transect, 'so', **kwagsS, **kwagsrho, **kwags_mask)   
            ax[0].set_xlabel('Distance (km)')
            ax[1].set_xlabel('Distance (km)')
    
            fig.suptitle(f"{time.dt.strftime('%Y-%m-%d').values}")
            #plt.close()

            if savefig:
                year=time.dt.year.values
                month=time.dt.month.values
                day=time.dt.day.values
                if fig_name:
                    finished_plot(fig, fig_name=f'{run_folder}animations/frames/transect_{transect_location}_{fig_name}_rho_y{year}m{month:02}d{day:02}.jpg')
                else:
                    finished_plot(fig, fig_name=f'{run_folder}animations/frames/transect_{transect_location}_rho_y{year}m{month:02}d{day:02}.jpg')
    return 
    
# Function produces figures of transects of observations on the Amundsen Sea shelf and simulation results    
# Inputs:
# run_folder : string path to folder containing NEMO simulations (gridT files)
# savefig    : (optional) boolean whether to save figure within figures sub-directory in run_folder
def transects_Amundsen(run_folder, transect_locations=['Getz_left','Getz_right','Dotson','PI_trough','shelf_west','shelf_mid','shelf_east','shelf_edge'], 
                       time_slice=("2000-01-01", "2015-12-31"), tmin=-2, tmax=0.5, smin=33, smax=35, savefig=False, ylim=(1300, -20), 
                       nemo_mesh='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20250715.nc', fig_dir='', fig_name=''):
    
    # load nemo simulations
    gridT_files  = glob.glob(f'{run_folder}*grid_T*')
    nemo_ds      = xr.open_mfdataset(gridT_files).sel(time_counter=time_slice) # load all the gridT files in the run folder
    nemo_ds      = nemo_ds.rename({'x_grid_T':'x', 'y_grid_T':'y', 'nav_lon_grid_T':'nav_lon', 'nav_lat_grid_T':'nav_lat', 'deptht':'depth'})
    nemo_results = nemo_ds.mean(dim='time_counter')
    nemo_mesh_ds = xr.open_dataset(nemo_mesh).squeeze()
    
    # load observations:
    obs          = read_dutrieux(eos='teos10')
    dutrieux_obs = obs.assign({'nav_lon':obs.lon, 'nav_lat':obs.lat}).rename_dims({'lat':'y', 'lon':'x'})
    
    # calculate transects and plot:
    for transect in transect_locations:
        # get coordinates for the transect:
        x_obs, y_obs = transect_coords_from_latlon_waypoints(dutrieux_obs, transect_amundsen[transect], opt_float=False)
        x_sim, y_sim = transect_coords_from_latlon_waypoints(nemo_mesh_ds, transect_amundsen[transect], opt_float=False)

        # subset the datasets and nemo_mesh to the coordinates of the transect:
        obs_transect = dutrieux_obs.isel(x=xr.DataArray(x_obs, dims='n'), y=xr.DataArray(y_obs, dims='n'))
        sim_transect = nemo_results.isel(x=xr.DataArray(x_sim, dims='n'), y=xr.DataArray(y_sim, dims='n'))
        nemo_mesh_transect  = nemo_mesh_ds.isel(x=xr.DataArray(x_sim, dims='n'), y=xr.DataArray(y_sim, dims='n')).rename({'nav_lev':'depth'})

        # add tmask, iceshelfmask and depths to the simulation dataset
        sim_transect = sim_transect.assign({'gdept_0':nemo_mesh_transect.gdept_0, 'tmask':nemo_mesh_transect.tmask, 'isfdraft':nemo_mesh_transect.isfdraft})

        # calculate the distance of each point along the transect relative to the start of the transect:
        obs_distance = distance_along_transect(obs_transect)
        sim_distance = distance_along_transect(nemo_mesh_transect)

        # visualize the transect:
        fig, ax = plt.subplots(2,2, figsize=(15,6), dpi=300)
        kwagsT    ={'vmin':tmin,'vmax':tmax,'cmap':cmocean.cm.dense,'label':'Conservative Temp.', 'ylim':ylim}
        kwagsS    ={'vmin':smin,'vmax':smax,'cmap':cmocean.cm.haline,'label':'Absolute Salinity', 'ylim':ylim}
        kwags_mask={'mask_land':True, 'mask_iceshelf':True}
        ax[0,0].set_title('Observations Dutrieux')
        ax[0,1].set_title('Observations Dutrieux')
        ax[1,0].set_title('Model simulations')
        ax[1,1].set_title('Model simulations')
        plot_transect(ax[0,0], obs_distance, obs_transect, 'ConsTemp', **kwagsT)
        plot_transect(ax[1,0], sim_distance, sim_transect, 'thetao', **kwagsT, **kwags_mask) 
        plot_transect(ax[0,1], obs_distance, obs_transect, 'AbsSal', **kwagsS)
        plot_transect(ax[1,1], sim_distance, sim_transect, 'so', **kwagsS, **kwags_mask) 
        ax[1,0].set_xlabel('Distance (km)')
        ax[1,1].set_xlabel('Distance (km)')

        if savefig:
            if fig_name:
                finished_plot(fig, fig_name=f'{fig_dir}evaluation_transect_{transect}_{fig_name}.jpg')
            else:
                finished_plot(fig, fig_name=f'{fig_dir}evaluation_transect_{transect}.jpg')

    return


# Set up list of timeseries for evaluation deck
def timeseries_types_evaluation ():

    regions = ['all', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross', 'west_antarctica', 'dotson_cosgrove']    
    var_names = ['massloss', 'shelf_bwtemp', 'shelf_bwsalt']
    var_names_ASE = ['massloss', 'shelf_temp_btw_200_700m', 'shelf_salt_btw_200_700m']
    timeseries_types_T = []
    for region in regions:
        if region == 'dotson_cosgrove':
            var_names_use = var_names_ASE
        else:
            var_names_use = var_names
        for var in var_names_use:
            timeseries_types_T.append(region+'_'+var)
    timeseries_types_U = ['drake_passage_transport', 'weddell_gyre_transport', 'ross_gyre_transport']
    timeseries_types = {'T' : timeseries_types_T,
                        'U' : timeseries_types_U}
    return timeseries_types


# Precompute timeseries for evaluation deck from Birgit's NEMO config
# eg for latest 'best' ERA5 case, uncompressed: in_dir='/gws/nopw/j04/terrafirma/kaight/NEMO_AIS/birgit_baseline/"
def update_timeseries_evaluation_NEMO_AIS (in_dir, out_dir='./'):

    domain_cfg = '/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/domain_cfg-20250715.nc'
    timeseries_types = timeseries_types_evaluation()

    for gtype in timeseries_types:
        update_simulation_timeseries('L121', timeseries_types[gtype], timeseries_file='timeseries_'+gtype+'.nc', timeseries_dir=out_dir, config='eANT025', sim_dir=in_dir, halo=False, gtype=gtype, domain_cfg=domain_cfg)


# As above, for UKESM1 suites
def update_timeseries_evaluation_UKESM1 (suite_id, base_dir='./', out_dir=None, compute_u=False):

    domain_cfg = '/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'
    timeseries_types = timeseries_types_evaluation()
    sim_dir = base_dir+'/'+suite_id+'/'
    if out_dir is None:
        out_dir = sim_dir        

    gtypes = ['T']
    if compute_u:
        gtypes += ['U']
    for gtype in gtypes:
        update_simulation_timeseries(suite_id, timeseries_types[gtype], timeseries_file='timeseries_'+gtype+'.nc', timeseries_dir=out_dir, sim_dir=sim_dir, halo=True, gtype=gtype, domain_cfg=domain_cfg)   


# Bug with Dotson-Cosgrove mask definition means we need to redo those timeseries variables only
def redo_dotson_cosgrove_timeseries (in_dir, out_dir='./'):

    var_names_ASE = ['massloss', 'shelf_temp_btw_200_700m', 'shelf_salt_btw_200_700m']
    timeseries_types = ['dotson_cosgrove_'+var for var in var_names_ASE]
    timeseries_file_old = 'timeseries_T.nc'
    timeseries_file_new = 'timeseries_redo.nc'
    update_simulation_timeseries('L121', timeseries_types, timeseries_file=timeseries_file_new, timeseries_dir=out_dir, config='eANT025', sim_dir=in_dir, halo=False, gtype='T')
    # Merge with old file
    print('Merging')
    ds_old = xr.open_dataset(out_dir+'/'+timeseries_file_old)
    ds_new = xr.open_dataset(out_dir+'/'+timeseries_file_new)
    for var in ds_new:
        ds_old[var] = ds_new[var]
    overwrite_file(ds_old, out_dir+'/'+timeseries_file_old)    


def plot_evaluation_timeseries_shelf (timeseries_file='timeseries_T.nc', fig_name=None):

    regions = ['all', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross', 'west_antarctica', 'dotson_cosgrove']    
    var_names = ['massloss', 'shelf_bwtemp', 'shelf_bwsalt']
    var_names_ASE = ['massloss', 'shelf_temp_btw_200_700m', 'shelf_salt_btw_200_700m']
    var_titles = ['Basal mass loss\n(Gt/y)', 'Temperature ('+deg_string+')\n on continental shelf', 'Salinity on\n continental shelf']
    num_regions = len(regions)
    num_var = len(var_names)
    smooth = 2*months_per_year

    ds = xr.open_dataset(timeseries_file)
    
    fig = plt.figure(figsize=(14,7))
    rows = num_regions//2
    columns = num_var*4*2 + 2
    gs = plt.GridSpec(rows, columns)
    gs.update(left=0.01, right=0.99, bottom=0.08, top=0.92, hspace=0.2, wspace=3)
    for n in range(num_regions):
        for v in range(num_var):
            y0 = 1 + (4*num_var+1)*(n//rows) + v*4
            ax = plt.subplot(gs[n%rows, y0:y0+4])
            var = regions[n]+'_'
            if regions[n] == 'dotson_cosgrove':
                var += var_names_ASE[v]
            else:
                var += var_names[v]
            # Plot data; monthly in thin grey, 2-year running mean in thicker black
            time = ds['time_centered']
            try:
                ax.plot(time, ds[var], color='DarkGrey', linewidth=1)
            except(TypeError):
                ax.plot_date(time, ds[var], '-', color='DarkGrey', linewidth=1)
            data_smoothed = moving_average(ds[var], smooth)
            try:
                ax.plot(data_smoothed.time_centered, data_smoothed,  color='black', linewidth=1.5)
            except(TypeError):
                ax.plot_date(data_smoothed.time_centered, data_smoothed, '-', color='black', linewidth=1.5)
            # Plot obs; central estimate in dashed blue, uncertainty range in shaded blue
            if 'massloss' in var:
                obs_mean = adusumilli_melt[regions[n]]
                obs_std = adusumilli_std[regions[n]]                
            else:
                if 'temp' in var:
                    m = 0
                elif 'salt' in var:
                    m = 1
                obs_mean = zhou_TS[regions[n]][m]
                obs_std = zhou_TS_std[regions[n]][m]
            ax.axhline(obs_mean, color='blue', linestyle='dashed', linewidth=1)
            ax.axhspan(obs_mean-obs_std, obs_mean+obs_std, color='DodgerBlue', alpha=0.1)
            ax.set_xlim([time[0], time[-1]])
            if n%rows == rows-1:
                ax.tick_params(axis='x', labelrotation=90)
            else:
                ax.set_xticklabels([])
            ax.grid(linestyle='dotted')
            ax.tick_params(axis='both', labelsize=7)
            if n%(rows) == 0:
                # Variable title and units on top
                ax.set_title(var_titles[v], fontsize=11)
            if v == 0:
                # Label region on the left
                plt.text(-0.3, 0.5, region_names[regions[n]], fontsize=10, ha='center', va='center', transform=ax.transAxes, rotation=90)
            else:
                # Label depth
                if n == 0:
                    depth_label = '(bottom)'
                elif n == num_regions - 1:
                    depth_label = '(200-700m)'
                else:
                    depth_label = ''
                plt.text(0.05, 0.95, depth_label, fontsize=8, ha='left', va='top', transform=ax.transAxes)
    finished_plot(fig, fig_name=fig_name)


def plot_evaluation_timeseries_transport (timeseries_file='timeseries_U.nc', fig_name=None):

    var_names = ['drake_passage_transport', 'weddell_gyre_transport', 'ross_gyre_transport']
    var_titles = ['Drake Passage', 'Weddell Gyre', 'Ross Gyre']
    num_var = len(var_names)
    smooth = 2*months_per_year

    ds = xr.open_dataset(timeseries_file)

    fig = plt.figure(figsize=(10,3))
    gs = plt.GridSpec(1, num_var)
    gs.update(left=0.06, right=0.99, bottom=0.2, top=0.9, wspace=0.15)
    for v in range(num_var):
        ax = plt.subplot(gs[0,v])
        time = ds['time_centered']
        ax.plot(time, ds[var_names[v]], color='DarkGrey', linewidth=1)
        data_smoothed = moving_average(ds[var_names[v]], smooth)
        ax.plot(data_smoothed.time_centered, data_smoothed, color='black', linewidth=1.5)
        region = var_names[v][:var_names[v].index('_transport')]
        obs_mean = transport_obs[region]
        obs_std = transport_std[region]
        ax.axhline(obs_mean, color='blue', linestyle='dashed', linewidth=1)
        ax.axhspan(obs_mean-obs_std, obs_mean+obs_std, color='DodgerBlue', alpha=0.1)
        ax.set_xlim([time[0], time[-1]])
        ax.set_title(var_titles[v])
        if v == 0:
            ax.set_ylabel('Transport (Sv)')
        ax.grid(linestyle='dotted')
        ax.tick_params(axis='x', labelrotation=90)
    finished_plot(fig, fig_name=fig_name)
        

# Calculate the observed mean and uncertainty of T and S on the shelf averaged over each region, from the Zhou 2025 climatology.
def preproc_shenjie (obs_file='/gws/nopw/j04/terrafirma/kaight/input_data/OI_climatology.nc', bathy_file='shenjie_climatology_bottom_TS.nc', out_file='OI_climatology_2D.nc'):

    regions_bottom = ['all', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross', 'west_antarctica']
    regions_mid = ['dotson_cosgrove']
    depth_bounds = [200, 700]
    bottom_thickness = 150 

    ds = xr.open_dataset(obs_file).squeeze().transpose('nz', 'ny', 'nx')

    # Read bathymetry from alternate file
    ds_bathy = xr.open_dataset(bathy_file).rename_dims({'NB_x':'nx', 'NB_y':'ny'}).drop_vars(['shelf_mask']).transpose()
    # Swap sign on bathymetry and mask land
    ds_bathy['bathymetry'] = -1*ds_bathy['bathymetry']    
    # Build shelf mask
    ds_bathy = build_shelf_mask(ds_bathy)[1]
    # Now mask land (and cavities) - crucially after shelf mask constructed otherwise the Brunt overhang disconnects it
    ds_bathy['shelf_mask'] = xr.where(ds['ct'].isel(nz=0).notnull(), ds_bathy['shelf_mask'], 0)
    # Copy the two variables we need over to the main dataset
    ds = ds.assign({'bathymetry':ds_bathy['bathymetry'], 'shelf_mask':ds_bathy['shelf_mask']})
    ds_bathy.close()
    
    # Precompute the region masks
    for region in regions_bottom+regions_mid:
        if region == 'east_antarctica':
            # Do this one at the end
            continue
        mask, ds = region_mask(region, ds, option='shelf')
    # Now do East Antarctica: what's left when you remove the others
    mask = ds['all_shelf_mask'].copy()
    for region in regions_bottom[1:]:
        if region not in ['east_antarctica', 'amery']:
            mask -= ds[region+'_shelf_mask']*mask
    # Now slice out disconnected regions by imposing bounds
    region = 'east_antarctica'
    mask = xr.where((region_edges[region][0][0] < ds['longitude'])*(ds['longitude'] < region_edges[region][1][0]), mask, 0).transpose()
    ds = ds.assign({'east_antarctica_shelf_mask':mask})

    # Get area integrands
    lon = ds['longitude'].data
    lon_mid = 0.5*(lon[:-1] + lon[1:])
    lon_edges = np.concatenate(([0.5*(lon_mid[0] + lon_mid[-1] - 360)], lon_mid, [0.5*(lon_mid[0] + 360 + lon_mid[-1])]))
    lat = ds['latitude'].data
    lat_mid = 0.5*(lat[:-1] + lat[1:])
    lat_edges = np.concatenate(([2*lat_mid[0] - lat_mid[1]], lat_mid, [2*lat_mid[-1] - lat_mid[-2]]))
    lon_edges, lat_edges = np.meshgrid(lon_edges, lat_edges)
    lon, lat = np.meshgrid(lon, lat)
    dlon = lon_edges[1:,1:] - lon_edges[1:,:-1]
    dlat = lat_edges[1:,1:] - lat_edges[:-1,1:]
    dx = rEarth*np.cos(lat*deg2rad)*dlon*deg2rad
    dy = rEarth*dlat*deg2rad
    dA = xr.DataArray(dx*dy, coords={'ny':ds['ny'], 'nx':ds['nx']})
    # Get depth integrand
    z = ds['pressure']
    z_mid = 0.5*(z[:-1] + z[1:])
    z_edges = np.concatenate(([2*z_mid[0] - z_mid[1]], z_mid, [2*z_mid[-1] - z_mid[-2]]))
    dz = z_edges[1:] - z_edges[:-1]
    dz = xr.DataArray(dz, coords={'nz':ds['nz']})

    # Mask for bottom layer: within 150 m of bathymetry (assume pressure in dbar = depth in m)
    bathy = ds['bathymetry']
    bottom_mask = xr.where((z<bathy)*(z>bathy-bottom_thickness)*ds['ct'].notnull(), 1, 0)
    # Mask for 200-700m depth range
    mid_mask = xr.where((z>depth_bounds[0])*(z<depth_bounds[1])*ds['ct'].notnull(), 1, 0)

    # Loop over variables we care about
    ds_out = None
    for var in ['ct', 'ct_mse', 'sa', 'sa_mse']:
        print('\n'+var)
        ds[var].load()
        for depth_mask, regions, depth_name in zip([bottom_mask, mid_mask], [regions_bottom, regions_mid], ['bottom', '200-700m']):
            if 'mse' in var:
                # Extra mask on uncertainty variables
                depth_mask_tmp = depth_mask.where(ds[var]!=1e10)
            else:
                depth_mask_tmp = depth_mask
            # Vertically average over depth range
            var_2D = (ds[var]*dz*depth_mask_tmp).sum(dim='nz')/(dz*depth_mask_tmp).sum(dim='nz')
            if ds_out is None:
                ds_out = xr.Dataset({var+'_'+depth_name:var_2D})
                for varg in ['latitude', 'longitude']:
                    ds_out = ds_out.assign({varg:ds[varg].squeeze()})
            else:
                ds_out = ds_out.assign({var+'_'+depth_name:var_2D})
            for region in regions:
                mask = ds[region+'_shelf_mask']
                area = (dA*mask).where(var_2D.notnull())
                var_avg = (var_2D*area).sum()/area.sum()
                print(region+' ('+depth_name+'): '+str(var_avg.item()))
    ds_out.to_netcdf(out_file)


# Precompute bottom T and S averaged over the last part of the simulation (default 20 years). Convert to TEOS-10 if it's not already.
# config can be NEMO_AIS or UKESM1
def precompute_bottom_TS (config='NEMO_AIS', suite_id=None, in_dir=None, num_years=20, out_file='bottom_TS_avg.nc'):

    var_names_1 = ['tob', 'sob']
    var_names_2 = ['sbt', 'sbs']

    if config == 'NEMO_AIS':
        if suite_id is None:
            suite_id = 'eANT025.L121'
        if in_dir is None:
            in_dir = './'
        file_head = suite_id+'_1m_'
        file_tail = '_grid_T.nc'
        eos = 'teos10'
    elif config == 'UKESM1':
        if suite_id is None:
            raise Exception('Must set suite_id')
        if in_dir is None:
            in_dir = suite_id + '/'
        file_head = 'nemo_'+suite_id+'o_1m_'
        file_tail = 'grid-T.nc'
        eos = 'eos80'

    # Find all the output filenames
    nemo_files = []
    months_per_file = None
    for f in os.listdir(in_dir):
        if f.startswith(file_head) and f.endswith(file_tail):
            nemo_files.append(in_dir+'/'+f)
            if months_per_file is None:
                ds = xr.open_dataset(f)
                months_per_file = ds.sizes['time_counter']
                if months_per_file not in [1, months_per_year]:
                    raise Exception('Invalid months_per_file = '+str(months_per_file))
                ds.close()
    # Sort chronologically
    nemo_files.sort()
    # Select the last num_years
    num_t = int(num_years*months_per_year/months_per_file)
    nemo_files = nemo_files[-num_t:]

    # Now read one file at a time
    ds_accum = None
    depth_bottom = None
    for file_path in nemo_files:
        print('Processing '+file_path)
        ds = xr.open_dataset(file_path)
        if eos == 'eos80' and depth_bottom is None:
            # Will need depth of bottom cell for converting to TEOS-10 later
            
            depth_3d_masked = xr.broadcast(ds['deptht'], ds['so'])[0].where(ds['so']!=0)
            depth_bottom =  depth_3d_masked.max(dim='deptht')
        # Only keep T and S
        if var_names_1[0] in ds:
            # Bottom variables already exist
            var_names = var_names_1
        else:
            var_names = var_names_2
        # Select only variables we want, and mask where identically zero
        ds = ds[var_names].where(ds[var_names[0]]!=0)
        if eos == 'eos80':
            # Convert to TEOS-10
            pot_temp = ds[var_names[0]]
            prac_salt = ds[var_names[1]]
            abs_salt = gsw.SA_from_SP(prac_salt, depth_bottom, ds['nav_lon'], ds['nav_lat'])
            con_temp = gsw.CT_from_pt(abs_salt, pot_temp)
            ds[var_names[0]] = con_temp.assign_attrs(long_name='conservative temperature, TEOS-10')
            ds[var_names[1]] = abs_salt.assign_attrs(long_name='absolute salinity, TEOS-10')
            
        if months_per_file == months_per_year:
            # Annual average
            ndays = ds.time_centered.dt.days_in_month
            weights = ndays/ndays.sum()
            # This is more memory efficient than the built in functions, if not more code efficient!
            for t in range(months_per_file):
                ds_tmp = ds.isel(time_counter=t)
                for var in var_names:
                    ds_tmp[var] = ds_tmp[var]*weights[t]
                ds_tmp = ds_tmp.drop_vars({'time_counter', 'time_centered'})
                if ds_accum is None:
                    ds_accum = ds_tmp
                else:
                    ds_accum += ds_tmp
        elif config == 'UKESM1':
            # UKESM1 has 30-day months so don't need to worry about weights
            ds = ds.squeeze().drop_vars({'time_counter', 'time_centered'})
            if ds_accum is None:
                ds_accum = ds
            else:
                ds_accum += ds
        else:
            raise Exception('Unsure how to handle monthly files for config='+config)
        ds.close()
    ds_avg = ds_accum/num_t

    print('Writing '+out_file)
    ds_avg.to_netcdf(out_file)
    ds_avg.close()


# Plot bottom T and S compared to Shenjie's obs.
def plot_bottom_TS (in_file='bottom_TS_avg.nc', obs_file='/gws/nopw/j04/terrafirma/kaight/input_data/OI_climatology_2D.nc', fig_name=None):

    var_names_1 = ['tob', 'sob']
    var_names_2 = ['sbt', 'sbs']
    var_names_obs = ['ct_bottom', 'sa_bottom']
    var_titles = ['Conservative temperature ('+deg_string+'C)', 'Absolute salinity']
    vmin = [-2, 34]
    vmax = [5, 35]
    vdiff = [1, 0.5]
    subtitles = ['Model', 'Observations', 'Model bias']
    ctype = ['RdBu_r', 'RdBu_r', 'plusminus']

    # Read precomputed model fields
    ds_model = xr.open_dataset(in_file)
    if var_names_1[0] in ds_model:
        var_names = var_names_1
    else:
        var_names = var_names_2

    # Read observations and interpolate to model grid
    ds_obs = xr.open_dataset(obs_file)
    def set_var (var_name):
        return xr.DataArray(ds_obs[var_name].data, coords=[ds_obs['latitude'].data, ds_obs['longitude'].data], dims=['lat', 'lon'])
    [temp, salt] = [set_var(var_names_obs[v]) for v in range(2)]
    ds_obs_rename = xr.Dataset({var_names[0]:temp, var_names[1]:salt})
    ds_obs_interp = interp_latlon_cf(ds_obs_rename, ds_model, method='bilinear')
    
    # Plot
    fig = plt.figure(figsize=(8,6))
    gs = plt.GridSpec(2,3)
    gs.update(left=0.1, right=0.9, bottom=0.05, top=0.9, wspace=0.1, hspace=0.3)
    for v in range(2):
        model_plot = ds_model[var_names[v]]
        obs_plot = ds_obs_interp[var_names[v]]
        data_plot = [model_plot, obs_plot, model_plot-obs_plot]
        vmin_tmp = [vmin[v], vmin[v], -1*vdiff[v]]
        vmax_tmp = [vmax[v], vmax[v], vdiff[v]]
        for n in range(3):
            ax = plt.subplot(gs[v,n])
            ax.axis('equal')
            img = circumpolar_plot(data_plot[n], ds_model, ax=ax, masked=True, make_cbar=False, title=subtitles[n], vmin=vmin_tmp[n], vmax=vmax_tmp[n], ctype=ctype[n], lat_max=-63)
            if n != 1:
                cax = fig.add_axes([0.01+0.45*n, 0.05+0.5*v, 0.02, 0.3])
                plt.colorbar(img, cax=cax, extend='both')
        plt.text(0.5, 0.99-0.5*v, var_titles[v], ha='center', va='top', transform=fig.transFigure, fontsize=14)
    finished_plot(fig, fig_name=fig_name)

    

    

    
    
