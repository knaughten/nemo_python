import matplotlib.pyplot as plt
import matplotlib.colors as cl
import xarray as xr
import socket
import numpy as np
import cmocean
from .utils import polar_stereo, extend_grid_edges, moving_average
from .grid import build_ocean_mask, build_ice_mask
from .plot_utils import set_colours
from .constants import line_colours, region_names, land_colour, iceshelf_colour

# If a figure name is defined, save the figure to that file. Otherwise, display the figure on screen.
def finished_plot (fig, fig_name=None, dpi=None, print_out=True):

    if fig_name is not None:
        if print_out: print(('Saving ' + fig_name))
        fig.savefig(fig_name, dpi=dpi)
    else:
        fig.show()
        

# Plot a 2D field (lat-lon) on a polar stereographic projection of Antarctica (default) or Arctic. Assumes it's a periodic grid covering all longitudes.
# Arguments:
# data: an xarray DataArray of a 2D field (lat-lon)
# grid: an xarray Dataset containing the fields:
#    1. nav_lon_grid_T, nav_lat_grid_T, bounds_nav_lon_grid_T, bounds_nav_lat_grid_T; or
#    2. glamt, gphit, glamf, gphif; or
#    3. nav_lon, nav_lat, bounds_lon, bounds_lat; or
#    4. (if cice=True): TLON, TLAT, lont_bounds, latt_bounds
# Optional keyword arguments:
# pole: default 'S' (Antarctica), or 'N' (Arctic)
# cice: whether to plot CICE data insetad of NEMO (default False)
# ax: handle to Axes object to make the plot in
# make_cbar: whether to make a colourbar (default True)
# masked: whether data is already masked; if False (default) it will be masked wherever it's identically zero
# title: optional string for the title, if None, will use the DataArray name
# titlesize: fontsize for the title (default 16)
# fig_name: optional filename for figure (otherwise show interactively)
# return_fig: if True, return the figure and axes handles
# vmin, vmax: optional bounds on colour map
# ctype: colourmap type (see set_colours in plot_utils.py)
# change_points: arguments to ismr colourmap (see above)
# contour: list of levels to contour
# contour_colour: colour to contour (default black)
# shade_land: whether to shade the land mask in grey (default True unless cice=True)
# contour_ice: whether to contour the ice front (default False)
# icefront_colour: colour to contour ice front (default black)
# lognorm: logarithmic colormap normalization
# zoom_amundsen: boolean to activate a zoom on the Amundsen sea region
# lon0: longitude at the top of the plot (default 0)

def circumpolar_plot (data, grid, pole='S', cice=False, ax=None, make_cbar=True, masked=False, title=None, titlesize=16, fig_name=None, return_fig=False, vmin=None, vmax=None, ctype='viridis', change_points=None, periodic=True, lat_max=None, contour=None, contour_colour='black', shade_land=None, lognorm=False, cbar_kwags={}, zoom_amundsen=False, contour_ice=False, icefront_colour='black', lon0=0):

    import cf_xarray as cfxr

    # shade_land only works if cice=False
    if shade_land is None:
        shade_land = not cice
    if shade_land and cice:
        raise Exception('shade_land can only be True if cice=False')

    if pole == 'N':
        lat_c = 71
    elif pole == 'S':
        lat_c = -71
    else:
        raise Exception('Invalid value for pole: '+str(pole))

    new_fig = ax is None
    if title is None:
        title = data.name

    if not masked:
        # Mask where identically zero
        data = data.where(data!=0)
    # Remove any singleton time dimension
    data = data.squeeze()

    if 'nav_lat_grid_T' in grid:
        lat_name = 'nav_lat_grid_T'
        lon_name = 'nav_lon_grid_T'
    elif 'gphit' in grid:
        lat_name = 'gphit'
        lon_name = 'glamt'
    elif 'nav_lat' in grid:
        lat_name = 'nav_lat'
        lon_name = 'nav_lon'
    elif 'TLAT' in grid:
        lat_name = 'TLAT'
        lon_name = 'TLON'
    # Choose a northern/southern boundary if needed
    # lat_max actually also works as a southern boundary if pole='N', despite the name
    if lat_max is None:
        if cice:
            if pole == 'N':
                lat_max = 50
                data = data.where(grid[lat_name]>0)
            elif pole == 'S':
                lat_max = -50
                data = data.where(grid[lat_name]<0)
        else:
            if pole == 'N':
                if grid[lat_name].min() < 0:
                    print('Warning: this grid includes the southern hemisphere. Can cause weirdness in plotting')
                # Manually find southern boundary - careful with -1 used as missing values
                lat_max = grid[lat_name].where(grid[lat_name]!=-1).min().item()
            elif pole == 'S':   
                if grid[lat_name].max() > 0:
                    print('Warning: this grid includes the northern hemisphere. Can cause weirdness in plotting')
                lat_max = grid[lat_name].where(grid[lat_name]!=-1).max().item()

    # Get cell edges in polar stereographic coordinates
    if lat_name == 'nav_lat_grid_T':
        
        lon_edges = cfxr.bounds_to_vertices(grid['bounds_nav_lon_grid_T'], 'nvertex_grid_T')
        lat_edges = cfxr.bounds_to_vertices(grid['bounds_nav_lat_grid_T'], 'nvertex_grid_T')
    elif lat_name == 'gphit':
        lon_edges = extend_grid_edges(grid['glamf'], 'f', periodic=True)
        lat_edges = extend_grid_edges(grid['gphif'], 'f', periodic=True)
    elif lat_name == 'nav_lat':
        try:
            lon_edges = cfxr.bounds_to_vertices(grid['bounds_lon'], 'nvertex')
            lat_edges = cfxr.bounds_to_vertices(grid['bounds_lat'], 'nvertex')
        except(KeyError):
            lon_edges = cfxr.bounds_to_vertices(grid['bounds_nav_lon'], 'nvertex')
            lat_edges = cfxr.bounds_to_vertices(grid['bounds_nav_lat'], 'nvertex')
    elif lat_name == 'TLAT':
        lon_edges = cfxr.bounds_to_vertices(grid['lont_bounds'], 'nvertices')
        lat_edges = cfxr.bounds_to_vertices(grid['latt_bounds'], 'nvertices')
    x_edges, y_edges = polar_stereo(lon_edges, lat_edges, lat_c=lat_c, lon0=lon0)

    # Get axes bounds
    if zoom_amundsen:
        lon_bounds = np.array([-87, -145, -145, -87])
        lat_bounds = np.array([-70, -77, -77, -70])
    else:
        lon_bounds = np.array([0, 90, 180, -90])
        lat_bounds = np.array([lat_max]*4)
    x_bounds, y_bounds = polar_stereo(lon_bounds, lat_bounds, lat_c=lat_c, lon0=lon0)
    xlim = [x_bounds[3], x_bounds[1]]
    ylim = [y_bounds[2], y_bounds[0]]

    # Set up colour map
    cmap, vmin, vmax = set_colours(data, ctype=ctype, vmin=vmin, vmax=vmax, change_points=change_points)

    if shade_land or contour_ice:
        ocean_mask = build_ocean_mask(grid)[0]
    if contour_ice:
        ice_mask = build_ice_mask(grid)[0]
        mask_sum = ocean_mask + ice_mask  # 0 in grounded ice, 1 in open ocean, 2 in cavity
        ice_front_mask = ice_mask.where(mask_sum!=0)
    if shade_land:
        ocean_mask = ocean_mask.where(ocean_mask)
        x_bg, y_bg = np.meshgrid(np.linspace(x_edges.min().item(), x_edges.max().item()), np.linspace(y_edges.min().item(), y_edges.max().item()))
        mask_bg = np.ones(x_bg.shape)

    if new_fig:
        fig, ax = plt.subplots()
    if shade_land:
        # Shade background in grey
        ax.pcolormesh(x_bg, y_bg, mask_bg, cmap=cl.ListedColormap(['DarkGrey']))
        # Clear ocean back to white
        ax.pcolormesh(x_edges, y_edges, ocean_mask, cmap=cl.ListedColormap(['white']))
    # Now plot the data
    if lognorm:
       if (vmin <= 0 or vmax <= 0):
          print('vmin and vmax must be positive and non-zero when using a logarithmic colormap (lognorm=True), so allowing pcolormesh to choose limits instead')
          vmin=None; vmax=None;

       img = ax.pcolormesh(x_edges, y_edges, data, cmap=cmap, norm=cl.LogNorm(vmin=vmin, vmax=vmax)) # note that vmin can't be zero when logarithmic
    else:   
       img = ax.pcolormesh(x_edges, y_edges, data, cmap=cmap, vmin=vmin, vmax=vmax)
    if contour is not None or contour_ice:
        x, y = polar_stereo(grid[lon_name], grid[lat_name], lon0=lon0)
        if contour is not None:
            ax.contour(x, y, data, levels=contour, colors=(contour_colour), linewidths=1, linestyles='solid')
        if contour_ice:
            ax.contour(x, y, ice_front_mask, levels=[0.5], colors=(icefront_colour), linewidths=0.5, linestyles='solid')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    ax.set_title(title, fontsize=titlesize)
    if make_cbar:
        plt.colorbar(img, **cbar_kwags)
    if return_fig:
        return fig, ax
    elif fig_name is not None:
        fig.savefig(fig_name)
    elif new_fig:
        fig.show()
    else:
        return img
        


# Plot one or more timeseries on the same axis, with different colours and a legend if needed.
def timeseries_plot (datas, labels=None, colours=None, title='', units='', ax=None, fig_name=None, linewidth=None):

    new_ax = ax is None
    
    # Check if multiple lines to plot
    multi_data = isinstance(datas, list)
    if multi_data:
        if labels is None or colours is None:
            raise Exception('Need to set labels and colours')
        if len(labels) != len(datas) or len(colours) != len(datas):
            raise Exception('Wrong length of labels or colours')
    else:
        datas = [datas]
        labels = [None]
        colours = [None]

    if new_ax:
        if multi_data:
            figsize = (11,6)
        else:
            figsize = (6,4)
        fig, ax = plt.subplots(figsize=figsize)
    for data, label, colour in zip(datas, labels, colours):
        ax.plot_date(data.time_centered, data, '-', color=colour, label=label, linewidth=linewidth)
    ax.grid(linestyle='dotted')
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(units, fontsize=16)
    if multi_data and new_ax:
        # Make legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    if new_ax:
        finished_plot(fig, fig_name=fig_name)


# Plot timeseries of the same variable in different regions. Can either do for a single simulation (sim_dir is a string) or an initial conditions ensemble (sim_dir is a list of strings). 
def timeseries_by_region (var_name, sim_dir, regions=['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross'], colours=None, timeseries_file='timeseries.nc', smooth=0, fig_name=None, linewidth=None, title=None):

    if isinstance(sim_dir, str):
        sim_dir = [sim_dir]
    if colours is None:
        if len(regions) <= len(line_colours):
            colours = line_colours[:len(regions)]
        else:
            raise Exception('Too many regions to use default line_colours: set colours instead')

    all_ds = []
    for d in sim_dir:
        all_ds.append(xr.open_dataset(d+'/'+timeseries_file))
    num_ens = len(all_ds)

    datas = []
    labels = []
    colours_plot = []
    units = None
    for region, colour in zip(regions, colours):
        labels.append(region_names[region])
        labels += [None]*(num_ens-1)
        colours_plot += [colour]*num_ens
        var_full = region+'_'+var_name
        for ds in all_ds:
            datas.append(moving_average(ds[var_full], smooth))
            if title is None:
                long_name = ds[var_full].long_name
                title = long_name.replace(region_names[region]+' ','')
                units = ds[var_full].units

    timeseries_plot(datas, labels=labels, colours=colours_plot, title=title, units=units, fig_name=fig_name, linewidth=linewidth)
        

# Plot timeseries of the same variable in different experiments. Each experiment can be a single simulation (sim_dirs=list of strings) or an ensemble (sim_dirs=list of lists of strings).
# For ensembles, if sim_names is set, it can be the name of every member (list of lists of strings) or one name for every ensemble (list of strings).
# If sim_names is not set, lines will be labelled with the suite IDs (extracted from sim_dirs)
def timeseries_by_expt (var_name, sim_dirs, sim_names=None, colours=None, timeseries_file='timeseries.nc', smooth=0, fig_name=None, linewidth=None):

    num_expt = len(sim_dirs)
    if colours is None:
        if num_expt <= len(line_colours):
            colours = line_colours[:num_expt]
        else:
            raise Exception('Too many experiments to use default line_colours: set colours instead')

    # Inner function to extract the name of the current directory (with trailing slashes and parent directories stripped out)
    def make_sim_name (dir_name):
        # First strip out trailing slash(es), if it's there
        while True:
            if dir_name[-1] != '/':
                break
            dir_name = dir_name[:-1]
        # Now strip out parent directories
        if '/' in dir_name:
            dir_name = dir_name[dir_name.rfind('/')+1:]
        return dir_name
        
    if sim_names is None:
        # Generate simulation names from directory names (hopefully suite IDs), with same structure as sim_dirs
        sim_names = []
        for sim_dir in sim_dirs:
            if isinstance(sim_dir, str):
                # Single simulation
                sim_names.append(make_sim_name(sim_dir))
            elif isinstance(sim_dir, list):
                # Ensemble
                ens_names = []
                for d in sim_dir:
                    ens_names.append(make_sim_name(d))
                sim_names.append(ens_names)

    datas = []
    labels = []
    colours_plot = []
    title = None
    units = None
    for sim_dir, sim_name, colour in zip(sim_dirs, sim_names, colours):
        if isinstance(sim_dir, str):
            # Generalise to ensemble of 1
            sim_dir = [sim_dir]
            sim_name = [sim_name]
        elif isinstance(sim_dir, list) and isinstance(sim_name, str):
            # Just one name for the whole ensemble; only label the first member
            sim_name = [sim_name] + [None]*(len(sim_dir)-1)
        num_ens = len(sim_dir)
        colours_plot += [colour]*num_ens
        for d, n in zip(sim_dir, sim_name):
            ds = xr.open_dataset(d+'/'+timeseries_file)
            datas.append(moving_average(ds[var_name],smooth))
            labels.append(n)
            if title is None:
                title = ds[var_name].long_name
                units = ds[var_name].units

    timeseries_plot(datas, labels=labels, colours=colours_plot, title=title, units=units, fig_name=fig_name, linewidth=linewidth)


# Function to create mp4 animation from image files (jpg or png)
# Input:
# filenames : list of strings of image file names (in sorted order)
# out_file  : (optional) string of name and path for animation
def create_animation (filenames, out_file='test.mp4'):
    import imageio
    
    # filenames is a list of the names/locations of image files to combine into animation (mp4 in this case)
    with imageio.get_writer(f'{out_file}', fps=2, mode='I') as writer: 
        for filename in filenames:  
            image = imageio.imread(filename)  
            writer.append_data(image)
    return

### Function that produces an animation of a 2D circumpolar field
# Inputs:
# run_folder : string of run directory path
# var        : string of the variable name to visualize from the NetCDF files 
# stub       : string of the end of name of run file (for ex: grid_T, icemod etc.)
# vlim       : (optional) colorbar lower and upper limits
# cmap       : (optional) colormap
# nemo_mesh  : (optional) string of location of NEMO mesh mask file for grid
# Output: jpg figures in animations/ directory within the run directory and an animation within an animations/ sub-directory
def animate_2D_circumpolar(run_folder, var, stub, vlim=(0,100), cmap='viridis', out_folder='',
                           nemo_mesh='/gws/ssde/j25b/terrafirma/birgal/NEMO_AIS/bathymetry/mesh_mask-20260121.nc'):
    import glob

    if not out_folder:
        out_folder = f'{run_folder}animations/'

    # Load necessary NetCDF files:
    nemo_mesh_ds  = xr.open_dataset(nemo_mesh).isel(time_counter=0)
    file_list     = glob.glob(f'{run_folder}/*{stub}*.nc')
    animate_ds    = xr.open_mfdataset(file_list)

    # Create figure for each timestep:
    for t, time in enumerate(animate_ds.time_counter):
        fig, ax = plt.subplots(1,1, figsize=(10,8))
        circumpolar_plot(animate_ds[var].isel(time_counter=t), nemo_mesh_ds, ax=ax, make_cbar=True, 
                         return_fig=False, ctype=cmap, lat_max=-50, vmin=vlim[0], vmax=vlim[1],
                         title=f'{animate_ds[var].long_name} \n {time.dt.year.values}-{time.dt.month.values:02}-{time.dt.day.values:02}')
        fig.tight_layout()
        finished_plot(fig, print_out=False,
                      fig_name=f'{out_folder}frames/{var}-y{time.dt.year.values}m{time.dt.month.values:02}d{time.dt.day.values:02}.jpg')
        plt.close()
    
    # Create animation from frames:
    filenames =np.sort(glob.glob(f'{out_folder}frames/{var}-y????m??d??.jpg'))
    create_animation(filenames, out_file=f'{out_folder}animation_{var}.mp4')

    return
    
### Function that produces a hovmoeller plot of the specified xarray datarray
# Inputs:
# datarray   : xarray dataarray of the variable to create the Hovmoeller plot from (with time_centered and deptht coordinates)
# varname    : (optional) string of the variable name to add as colorbar label
# title      : (optional) string of title for plot
# fig_size   : (optional) tuple dimensions of figure
# return_fig : (optional) boolean to return the figure and axis
# fig_name   : (optional) string of the name to save the figure as
# ylim       : (optional) tuple of the limits of the y-axis
# cmap       : (optional) colormap
# dpi        : (optional) resolution for saved figure
# vlim       : (optional) colorbar lower and upper limits
# Output: jpg figures in animations/ directory within the run directory and an animation within an animations/ sub-directory
def plot_hovmoeller(datarray, varname='', title=None, fig_size=(8,5), return_fig=False, fig_name=None, ylim=(5500,0), 
                    cmap='viridis', dpi=None, vlim=(-1.5,0.8)):

    fig, ax = plt.subplots(1,1, figsize=fig_size)
    cm1 = ax.pcolormesh([np.datetime64(date) for date in datarray.time_centered.values],
                        datarray.deptht.values, datarray.values,
                        rasterized=True, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    fig.colorbar(cm1, ax=ax, label=varname, extend='both')
    ax.invert_yaxis()
    ax.set_ylabel('Depth (m)')
    ax.set_ylim(ylim[0],ylim[1])

    if title is not None:
        fig.suptitle(title)
    if fig_name is not None:
        finished_plot(fig, fig_name=fig_name, dpi=dpi)
    if return_fig:
        return fig, ax
    else:
        return

# Function to plot a 2D histogram of temperature and salinity values on an axis you pass it
# Inputs:
# ax           : matplotlib axis
# salt         : absolute salinity
# temp         : conservative temperature
# smin, smax, tmin, tmax : (optional) bounds for the salinity and temperature axes
# plot_density : (optional) boolean specifying whether to plot dashed lines of constant potential density in the background
# plot_freeze  : (optional) boolean specifying whether to plot dashed line of surface freezing temperature in the background
# lognorm      : (optional) boolean specifying whether to log normalize the histogram counts
# bins         : (optional) number of bins of histogram
def plot_ts_distribution(ax, salt, temp, smin=30, smax=35.25, tmin=-3, tmax=2.25, plot_density=False, plot_freeze=False, lognorm=True, bins=400):

    import seaborn as sns
    import gsw
    import gsw.freezing as fr

    # Plot surface freezing temperature line
    if plot_freeze:
        tfreeze_sfc = fr.CT_freezing(np.linspace(smin, smax), 0, 0) # saturation_fraction=0
        ax.plot(np.linspace(smin, smax), tfreeze_sfc, color='black', linestyle='dashed', zorder=1)

    # Plot contours of potential density
    if plot_density:
        salt_2d, temp_2d = np.meshgrid(np.linspace(smin, smax), np.linspace(tmin, tmax))
        density = gsw.density.sigma0(salt_2d, temp_2d)
        ax.contour(salt_2d, temp_2d, density, colors='DarkGrey', linestyles='dotted', zorder=2)

    # Choose whether histogram count is logarithmic or normal
    if lognorm:
        kwags={'cbar':True,'ax':ax,'bins':bins,'norm':cl.LogNorm(),'vmin':None,'vmax':None,'zorder':3}
    else:
        kwags={'cbar':True,'ax':ax,'bins':bins,'zorder':3}
    
    # Plot histogram
    sns.histplot(x=salt, y=temp, **kwags)
    
    ax.set_xlim(smin, smax)
    ax.set_ylim(tmin, tmax)
    ax.set_xlabel('Absolute Salinity')
    ax.set_ylabel('Conservative Temperature')

    return 

# Function plots a transect of a variable
# Inputs:
# ax            : axis to plot the transect on
# distance_vec  : array of distance points along transect (could also be some other x-axis like lons or lats)
# data_transect : xarray dataset containing the depth and variable of the dataset to visualize
# var_name      : string name of variable to visualize
# ylim          : (optional) vertical axis range (as tuple)
# label         : (optional) string of name of colorbar label, otherwise defaults to the units of the variable
# contour_var   : (optional) string of name of variable to plot contour levels from
# clevels       : (optional) integer of number of contour levels or list of specific levels to draw
def plot_transect(ax, distance_vec, data_transect, var_name, ylim=(1600, -20), vmin=33, vmax=34.8, cmap=cmocean.cm.haline, 
                  label=None, mask_land=False, mask_iceshelf=False, contour_var='', clevels=10):
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_ylabel('Depth (m)')

    distance, __ = np.meshgrid(distance_vec, np.abs(data_transect.depth))
    kwags = {'vmin':vmin, 'vmax':vmax, 'cmap':cmap}

    # either create a meshgrid of depths or read in the depth array from the simulations (contains the partial steps)
    if 'gdept_0' in list(data_transect.keys()):
        depth = data_transect.gdept_0
    else:
        __, depth = np.meshgrid(distance_vec, np.abs(data_transect.depth))

    # Plot the transects
    try:
        cm1 = ax.pcolormesh(distance, depth, data_transect[var_name].values, **kwags)
        if contour_var:
            plot_contour = xr.where(data_transect[contour_var].values==0, np.nan, data_transect[contour_var].values)
            ax.contour(distance, depth, plot_contour, levels=clevels, colors='w', linewidths=0.5)
    except: # in case the shapes are flipped
        cm1 = ax.pcolormesh(distance.transpose(), depth.transpose(), data_transect[var_name].values, **kwags)
        if contour_var:
            plot_contour = xr.where(data_transect[contour_var].values==0, np.nan, data_transect[contour_var].values)
            ax.contour(distance.transpose(), depth.transpose(), plot_contour, levels=clevels, colors='w', linewidths=0.5)

    # mask land:
    if mask_land:
        if 'tmask' not in list(data_transect.keys()):
            raise Exception('Dataset must contain tmask if you use mask_land')
        land_mask = xr.where(data_transect.tmask==1, np.nan, data_transect.tmask.values)
        ax.pcolormesh(distance, depth, land_mask, rasterized=True, cmap=cl.ListedColormap(land_colour), zorder=3)

    # mask iceshelf:
    if mask_iceshelf:
        if ('gdept_0' or 'isfdraft') not in list(data_transect.keys()):
            raise Exception('Dataset must contain gdept_0 and isfdraft if you use mask_iceshelf')
        iceshelf_mask = xr.where(data_transect.gdept_0 < data_transect.isfdraft, 1, np.nan) 
        ax.pcolormesh(distance, depth, iceshelf_mask, rasterized=True, cmap=cl.ListedColormap(iceshelf_colour), zorder=4)
        
    if label:
        plt.colorbar(cm1, ax=ax, extend='both', label=label)
    else:
        plt.colorbar(cm1, ax=ax, extend='both', label=data_transect[var_name].unit)
    
    return

