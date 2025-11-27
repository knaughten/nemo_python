import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime
from .constants import region_bounds, region_bathy_bounds
from .interpolation import interp_latlon_cf_blocks
from .utils import polar_stereo, remove_disconnected, distance_to_bdry
from .plots import circumpolar_plot, finished_plot

# Steps to make new bathymetry on a given cut-out of a global grid (eg eORCA025):
# 1. coordinates_from_global (to set the coordinates)
# 2. interp_topo (to interpolate the bathymetry and ice draft)
# 3. fill_missing_topo (only if the dataset in interp_topo didn't cover your entire domain)
# 4. process_topo (to get everything in NEMO format)
# 5. splice_topo (only if you want to copy the Antarctic bathymetry back into a larger grid without changing the rest of the domain)
# 6. Copy the final file over to ARCHER2 and run it through DOMAINcfg (for NEMO 4.2) or just plug it straight into NEMO (for earlier versions).
# NB, NEMO 4.2 doesn't want the periodic domain to have a 2-cell halo, whereas earlier versions do.

# Given a pre-existing global domain (eg eORCA025), slice out a regional domain.
# Inputs:
# global_file = path to a NetCDF file for the global domain. The domain_cfg file is ideal, but any file that includes all of these variables will work: nav_lon, nav_lat, e2f, e2v, e2u, e2t, e1f, e1v, e1u, e1t, gphif, gphiv, gphiu, gphit, glamf, glamv, glamu, glamt
# out_file: path to desired output NetCDF file
# imin, imax, jmin, jmax: optional bounds on the i and j indicies to slice. Uses the python convention of 0-indexing, and selecting the indices up to but not including the last value. So, jmin=0, jmax=10 will select the southernmost 10 rows.
# nbdry: optional latitude of the desired northern boundary. The code will generate the value of jmax that corresponds to this on the zonal mean.
def coordinates_from_global (global_file='/gws/ssde/j25b/terrafirma/kaight/input_data/grids/domcfg_eORCA025_v3.nc', out_file='coordinates_AIS.nc', imin=None, imax=None, jmin=None, jmax=None, nbdry=None, remove_halo=True):

    ds = xr.open_dataset(global_file)

    if nbdry is not None:
        # Choose the value of jmax based on the supplied latitude
        if jmax is not None:
            raise Exception('nbdry and jmax are both defined')
        # Average over longitude to get a 1D field of "typical" latitude
        lat_1d = ds['nav_lat'].mean(dim='x')
        # Find the first index north of the given boundary
        jmax = np.where(lat_1d > nbdry)[0][0]

    # Now set default values for i and j slicing
    if imin is None:
        if remove_halo:
            # Remove the periodic halo (for NEMO 4.2): slice off first index
            imin = 1
        else:
            imin = 0
    if imax is None:
        if remove_halo:
            # also last index
            imax = ds.sizes['x'] - 1
        else:
            imax = ds.sizes['x']
    if jmin is None:
        jmin = 0
    if jmax is None:
        jmax = ds.sizes['y']

    # Now slice to these bounds
    ds_regional = ds.isel(x=slice(imin,imax), y=slice(jmin, jmax))

    # Select only the variables we need and write to file
    var_names = ['nav_lon', 'nav_lat', 'e2f', 'e2v', 'e2u', 'e2t', 'e1f', 'e1v', 'e1u', 'e1t', 'gphif', 'gphiv', 'gphiu', 'gphit', 'glamf', 'glamv', 'glamu', 'glamt']
    ds_regional[var_names].to_netcdf(out_file)


# Interpolate topography from a dataset (default BedMachine3) to the given NEMO coordinates.
# Inputs:
# dataset: name of source dataset, only 'BedMachine3' supported now
# topo_file: path to file containing dataset
# coordinates_file: path to file containing NEMO coordinates (could be created by coordinates_from_global above)
# out_file: desired path to output file for interpolated dataset
# periodic: whether the NEMO grid is periodic in longitude
# blocks_x, blocks_y: number of subdomains in x and y to split the domain into: iterating over smaller domains prevents memory overflowing when the entirety of BedMachine3 is being used. If you have a little domain which won't overflow, set them both to 1.
def interp_topo (dataset='BedMachine3', topo_file='/gws/ssde/j25b/terrafirma/kaight/input_data/topo/BedMachineAntarctica-v3.nc', coordinates_file='coordinates_AIS.nc', out_file='eORCA025_BedMachine3_AIS.nc', periodic=True, blocks_x=10, blocks_y=10):

    print('Processing input data')
    if dataset == 'BedMachine3':        
        source = xr.open_dataset(topo_file)
        # x and y coordinates are ints which can overflow later; cast to floats
        x = source['x'].astype('float32')
        y = source['y'].astype('float32')
        # Bathymetry is the variable "bed"
        bathy = source['bed']
        print('...calculating ice draft')
        # Ice draft is the surface minus thickness
        draft = source['surface'] - source['thickness']
        print('...combining masks')
        # Ocean mask includes open ocean (0) and floating ice (3)
        omask = xr.where((source['mask']==0)+(source['mask']==3), 1, 0)
        # Ice sheet mask includes everything except open ocean (1=rock, 2=grounded ice, 3=floating ice, 4=subglacial lake)
        imask = xr.where(source['mask']!=0, 1, 0)
        pster_src = True
        periodic_src = False
        # Now make a new Dataset containing only the variables we need
        source = xr.Dataset({'x':x, 'y':y, 'bathy':bathy, 'draft':draft, 'omask':omask, 'imask':imask})
    else:
        raise Exception('source dataset not yet supported')

    print('Reading NEMO coordinates')
    nemo = xr.open_dataset(coordinates_file).squeeze()
    
    print('Interpolating')
    #data_interp = interp_cell_binning(source, nemo, pster=pster_src, periodic=periodic, tmp_file=tmp_file)
    #data_interp = interp_latlon_cf(source, nemo, pster_src=pster_src, periodic_src=periodic_src, periodic_nemo=periodic, method='conservative')
    data_interp = interp_latlon_cf_blocks(source, nemo, pster_src=pster_src, periodic_src=periodic_src, periodic_nemo=periodic, method='conservative', blocks_x=blocks_x, blocks_y=blocks_y)
    data_interp.to_netcdf(out_file)


# Fill any missing cells from interp_topo near the northern boundary with another dataset (default GEBCO).
def fill_missing_topo (dataset='IBCSO', topo_file='/gws/ssde/j25b/terrafirma/kaight/input_data/topo/IBCSO_v2_bed_WGS84.nc', coordinates_file='coordinates_AIS.nc', interp_file='eORCA025_BedMachine3_AIS.nc', out_file='eORCA025_BedMachine3_IBCSO_AIS.nc', periodic=True, blocks_x=10, blocks_y=2):

    print('Processing input data')
    if dataset in ['GEBCO', 'IBCSO']:
        if dataset == 'GEBCO':
            z_name = 'elevation'
        else:
            z_name = 'z'
        source = xr.open_dataset(topo_file)
        bathy = source[z_name]
        omask = xr.where(bathy<0, 1, 0)
        draft = source[z_name]*0  # No ice shelves in the region to interpolate
        imask = omask*0
        pster_src = False
        periodic_src = True
        source = xr.Dataset({'lon':source['lon'], 'lat':source['lat'], 'bathy':bathy, 'draft':draft, 'omask':omask, 'imask':imask})
    else:
        raise Exception('source dataset not yet supported')

    print('Reading NEMO coordinates')
    nemo = xr.open_dataset(coordinates_file).squeeze()
    print('Selecting missing regions')
    nemo_interp1 = xr.open_dataset(interp_file)
    # Find the southernmost row with missing data, and give it a few buffer cells to the south
    missing_rows = nemo_interp1['bathy'].isnull().sum(dim='x')
    jmin = np.where(missing_rows > 0)[0][0] - 2
    # Now slice the NEMO datasets
    nemo_N = nemo.isel(y=slice(jmin,None))
    nemo_N_interp1 = nemo_interp1.isel(y=slice(jmin,None))

    print('Interpolating')
    nemo_N_interp2 = interp_latlon_cf_blocks(source, nemo_N, pster_src=pster_src, periodic_src=periodic_src, periodic_nemo=periodic, method='conservative', blocks_x=blocks_x, blocks_y=blocks_y)

    # Merge this new data into the missing regions
    nemo_N_interp = xr.where(nemo_N_interp1.isnull(), nemo_N_interp2, nemo_N_interp1)
    nemo_interp = xr.concat([nemo_interp1.isel(y=slice(0,jmin)), nemo_N_interp], dim='y')

    # Save to file
    nemo_interp.to_netcdf(out_file)
    


# Following interpolation of topography, get everything in the right format for the NEMO tool DOMAINcfg, make diagnostic plots (optional), and save to a file.
# Inputs:
# in_file: path to file created by interp_topo above
# coordinates_file: path to file containing NEMO coordinates (could be created by coordinates_from_global above)
# out_file: desired path to output file
# will_splice: will this new topography be spliced into an existing domain (like eORCA1 for UKESM)? If so, turn errors about missing points into warnings.
# plot: whether to make diagnostic plots
# pster_src: whether the source dataset (eg BedMachine3) was polar stereographic (and hence the x2d, y2d variables in in_file); only matters if plot=True
def process_topo (in_file='eORCA025_BedMachine3_IBCSO_AIS.nc', coordinates_file='coordinates_AIS.nc', out_file='bathy_meter_eORCA025_BedMachine3_IBCSO_AIS.nc', bear_ridge=False, will_splice=False, plot=True, pster_src=True):

    if plot:
        import matplotlib.pyplot as plt

    topo = xr.open_dataset(in_file)
    if np.count_nonzero(topo['bathy'].isnull()):
        # There are missing points
        if will_splice:
            print('Warning: there are missing points. Hopefully this will be dealt with by your splicing later - check to make sure.')
        else:
            raise Exception('Missing points')
    nemo = xr.open_dataset(coordinates_file).squeeze()

    # Set masks where they fall between 0 and 1
    topo['omask'] = np.round(topo['omask'])
    topo['imask'] = np.round(topo['imask'])
    # Make bathymetry and ice draft positive
    # Don't need to mask grounded ice with zeros as NEMO will do that for us (and if coupled ice sheet is active need to know bathymetry of grounded ice)
    topo['bathy'] = -topo['bathy']
    topo['draft'] = xr.where(topo['imask']==1, -topo['draft'], 0)
    if not will_splice:
        # Turn negative values (above sea level) to 0 - they'll never be ocean cells
        topo['bathy'] = xr.where(topo['bathy']>0, topo['bathy'], 0)
        topo['draft'] = xr.where(topo['draft']>0, topo['draft'], 0)

    if bear_ridge:
       print('Masking grounded Bear Ridge icebergs as land')
       topo['bathy'], topo['omask'] = add_bear_ridge_bergs(nemo['nav_lon'], nemo['nav_lat'], topo['bathy'], topo['omask'])
    else:
       print('Not including grounded Bear Ridge icebergs')
    
    # Now get bathymetry outside of cavities, masked with zeros where there's grounded or floating ice
    topo['Bathymetry'] = xr.where(topo['imask']==0, topo['bathy'], 0)
    # Make a new dataset with all the variables we need
    output = xr.Dataset({'nav_lon':nemo['nav_lon'], 'nav_lat':nemo['nav_lat'], 'Bathymetry':topo['Bathymetry'], 'Bathymetry_isf':topo['bathy'], 'isf_draft':topo['draft']})
    output.to_netcdf(out_file)

    if plot:
        if pster_src:
            # Convert nemo lat and lon into polar stereographic for direct comparison with x2d and y2d
            x, y = polar_stereo(nemo['nav_lon'], nemo['nav_lat'])
        else:
            x = nemo['nav_lon']
            y = nemo['nav_lat']
        # Make a bunch of plots
        #circumpolar_plot((x-topo['x2d']).where(topo['omask']==1), nemo, title='Error in x-coordinate (m)', ctype='plusminus', masked=True)
        #circumpolar_plot(y-topo['y2d'].where(topo['omask']==1), nemo, title='Error in y-coordinate (m)', ctype='plusminus', masked=True)
        #circumpolar_plot(topo['num_points'].where(topo['omask']==1), nemo, title='num_points', masked=True)
        for var in ['Bathymetry', 'Bathymetry_isf', 'isf_draft']:
            circumpolar_plot(output[var], nemo, title=var, masked=True, shade_land=False)


# Helper function to add grounded icebergs to Bear Ridge.
# Inputs: as xarray variables
# lon: NEMO grid longitudes (from -180 to 180 degrees east)
# lat: NEMO grid latitudes
# bathy: sea floor bathymetry
# omask: ocean mask (contains 1s and 0s)
def add_bear_ridge_bergs (lon, lat, bathy, omask):

    # Southern part: north-south wall
    [x_wall, x_wall, ymin, ymax] = region_bounds['bear_ridge_S']
    # Find longitude indices closest to the target
    i_wall   = abs(lon-x_wall).argmin(axis=1)
    wall_pts = (lat >= ymin)*(lat <= ymax)*(lon == lon[:,i_wall])
    bathy = xr.where(wall_pts, 0, bathy)
    omask = xr.where(wall_pts, 0, omask)

    # Northern part: select easternmost point in area where bathymetry is shallower than 300m
    [xmin, xmax, ymin, ymax] = region_bounds['bear_ridge_N']
    z_deep = region_bathy_bounds['bear_ridge_N'][1]
    # Find area of Bear Ridge with points shallower than 300 m
    bear_ridge = (lat <= ymax)*(lat >= ymin)*(lon >= xmin)*(lon <= xmax)*(bathy < z_deep)
    # Identify maximum longitude grid point for each latitude that meets these conditions:
    bathy = xr.where((lon == lon.where(bear_ridge).max(axis=1)), 0, bathy)
    omask = xr.where((lon == lon.where(bear_ridge).max(axis=1)), 0, omask)

    return bathy, omask


# Given a global bathy_meter topography file, update the region around Antarctica (south of 57S and the 2500m isobath, seamounts excluded) using the supplied regional file.
# Inputs:
# topo_regional: path to regional bathy_meter file, created by process_topo above.
# topo_global: path to global bathy_meter file. The domain covered by topo_regional has to be the southernmost N rows of this global domain, but they don't both have to have a halo.
# out_file: path to desired output merged bathy_meter file.
# halo: whether to keep the halo on the periodic boundary - only matters if it exists in the global file but not the regional file.
# lat0: latitude bound to search for isobath (negative, degrees)
# depth0: isobath to define Antarctica (positive, metres)
# smooth: whether to smooth over the transition
# scale_dist: if smooth=True, distance to apply smoothing on either side of transition, in km
def splice_topo (topo_regional='bathy_meter_AIS.nc', topo_global='/gws/ssde/j25b/terrafirma/kaight/input_data/grids/bathy_eORCA1_noclosea_from_GEBCO2021_PlusCaspian_FillZero_S21TT_editsJuly2024.nc', out_file='bathy_meter_eORCA1_Storkey_spliceBedMachine3_smooth_nohalo.nc', halo=False, lat0=-57, depth0=2500, smooth=True, scale_dist=100):

    xr.set_options(keep_attrs=True)

    ds_regional = xr.open_dataset(topo_regional)
    ds_global = xr.open_dataset(topo_global)

    if ds_regional.sizes['x'] == ds_global.sizes['x']-2:
        # The global domain has a halo, but the regional one doesn't.
        if halo:
            # Need to create the halo in ds_regional
            ds_regional = xr.concat([ds_regional.isel(x=-1), ds_regional, ds_regional.isel(x=0)], dim='x')
            ds_regional['x'] = ds_global['x']
            ds_regional = ds_regional.drop_vars('x')
        else:
            # Need to remove the halo from global
            ds_global = ds_global.isel(x=slice(1,-1))
            ds_global['x'] = ds_regional['x']

    # Modify ds_global if needed
    # Change any mask to zeros 
    ds_global['Bathymetry'] = xr.where(ds_global['Bathymetry'].isnull(), 0, ds_global['Bathymetry'])
    if 'Bathymetry_isf' not in ds_global:
        # Placeholder: same as Bathymetry
        ds_global = ds_global.assign({'Bathymetry_isf':ds_global['Bathymetry']})
    if 'isf_draft' not in ds_global:
        # Placeholder: all 0s
        ds_global = ds_global.assign({'isf_draft':0*ds_global['Bathymetry']})

    # Make sure the regional dataset has the same coordinates and variables as the global one
    ds_regional = ds_regional.assign_coords(nav_lon=ds_regional['nav_lon'], nav_lat=ds_regional['nav_lat'])
    ds_regional = ds_regional[[var for var in ds_global]]
    # Extend the regional dataset to cover the global domain (just copy the rest of the global domain over)
    ny = ds_regional.sizes['y']
    ds_regional = xr.concat([ds_regional, ds_global.isel(y=slice(ny,None))], dim='y')

    # Now choose the points we want to update
    mask = (ds_regional['nav_lat']<lat0)*(ds_regional['Bathymetry']<depth0)
    # Remove seamounts
    connected = remove_disconnected(mask, (1,1))
    mask = xr.where(connected, mask, 0)

    # Calculate weights: 1 means entirely ds_regional, 0 means entirely ds_global
    if smooth:
        # Calculate distance to shelf edge
        dist = distance_to_bdry(ds_global['nav_lon'], ds_global['nav_lat'], mask)
        # Taper over scale_dist on either side using cosine function
        taper = (dist < scale_dist)*np.cos(np.pi/2*dist/scale_dist)
        # Want weights of 1 on shelf, 0.5 at boundary, 0 off shelf
        weights = xr.where(mask, 1-0.5*taper, 0.5*taper)
    else:
        weights = xr.where(mask, 1, 0)

    # Replace the global values with regional ones in this mask
    # Bathymetry is 0 in cavities, Bathymetry_isf includes cavities, isf_draft is 0 outside cavities.
    # Loop over variables rather than doing entire dataset at once, because if so some variables like nav_lat, nav_lon get lost
    missing = ds_regional['Bathymetry_isf'].isnull()
    for var in ds_global:
        # Fill missing regions in the regional dataset, otherwise the mask is carried through in the next line
        ds_regional[var] = xr.where(missing, ds_global[var], ds_regional[var])
        ds_global[var] = weights*ds_regional[var] + (1-weights)*ds_global[var]        # Take the inverse of the mask so we can put ds_global first and keep its attributes.
        #ds_global[var] = xr.where(mask==0, ds_global[var], ds_regional[var], keep_attrs=True)
    ds_global.attrs['history'] = ds_global.attrs['history'] + 'Antarctic topography updated to BedMachine3 by Kaitlin Naughten ('+str(datetime.date.today())+')'
    ds_global.to_netcdf(out_file)


# Plot up the differences resulting from splicing. Do this once you have a mesh_mask created by NEMO.
# old_file, new_file: old and new files for mesh_mask
# grid_file: file with all the coordinates for this grid, to be used in plotting
# old_halo, new_halo: whether the old and new files respectively contain the halo (True if used for NEMO 3.6)
# nbdry: northern boundary to plot
# fig_dir: directory in which to save figures (if not set, will show them on the screen instead)
def plot_splice_diff (old_file='mesh_mask_old.nc', new_file='mesh_mask.nc', old_halo=True, new_halo=True, nbdry=-56, fig_dir=None):

    old = xr.open_dataset(old_file).squeeze()
    new = xr.open_dataset(new_file).squeeze()
    
    # Trim to the northern boundary for plotting
    jmax = np.where(old['nav_lat'].mean(dim='x') > nbdry)[0][0]
    old = old.isel(y=slice(0,jmax))
    new = new.isel(y=slice(0,jmax))
    if old_halo:
        # Have to remove the halo for plotting
        old = old.isel(x=slice(1,-1))
    if new_halo:
        new = new.isel(x=slice(1,-1))
    diff = new-old

    # Make plots
    for var in ['tmaskutil', 'bathy', 'isfdraft']:
        fig = plt.figure(figsize=(10,4))
        gs = plt.GridSpec(1,3)
        gs.update(left=0.1, right=0.9, bottom=0.05, top=0.8, wspace=0.1)
        ds = [old, new, diff]
        titles = ['Old', 'New', 'Difference']
        vmin = min(np.amin(old[var].values), np.amin(new[var].values))
        vmax = max(np.amax(old[var].values), np.amax(new[var].values))
        for n in range(3):
            ax = plt.subplot(gs[0,n])
            ax.axis('equal')
            img = circumpolar_plot(ds[n][var], old, ax=ax, masked=True, make_cbar=False, title=titles[n], vmin=(vmin if n<2 else None), vmax=(vmax if n<2 else None), ctype=('viridis' if n<2 else 'plusminus'), titlesize=14)
            if n != 1:
                cax = fig.add_axes([0.01+0.46*n, 0.15, 0.02, 0.6])
                plt.colorbar(img, cax=cax)
        plt.suptitle(var, fontsize=16)
        fig_name = None
        if fig_dir is not None:
            fig_name = fig_dir+var+'compare.png'
        finished_plot(fig, fig_name=fig_name)
                                         
        
    


def interp_ics_TS (dataset='WOA18', source_files='/gws/ssde/j25b/terrafirma/kaight/input_data/WOA18/woa18_decav_*01_04.nc', nemo_dom='/gws/ssde/j25b/terrafirma/kaight/input_data/grids/domain_cfg_eANT025.L121.nc'):

    import gsw

    if dataset == 'WOA':
        # Read temperature and salinity from January as a single dataset
        woa = xr.open_mfdataset(in_files, decode_times=False).squeeze()
        # Convert to TEOS10
        # Need pressure in dbar at every 3D point: approx depth in m
        woa_press = np.abs(xr.broadcast(woa['depth'], woa['t_an'])[0])
        # Also need 3D lat and lon
        woa_lon = xr.broadcast(woa['lon'], woa['t_an'])[0]
        woa_lat = xr.broadcast(woa['lat'], woa['t_an'])[0]
        # Get absolute salinity
        woa_salt = gsw.SA_from_SP(woa['s_an'], woa_press, woa_lon, woa_lat)
        # Get conservative temperature
        woa_temp = gsw.CT_from_t(woa_salt, woa['t_an'], woa_press)
        # Now wrap up into a new Dataset
        source = xr.Dataset({'temp':woa_temp, 'salt':woa_salt})
        
        
    
        

