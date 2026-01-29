import netCDF4 as nc
import numpy as np
import xarray as xr

from .constants import cesm2_ensemble_members
from .utils import select_bottom, convert_to_teos10
from .interpolation import interp_latlon_cf

# NEMO 4.2 mesh_mask files are written with 2D variables x and y instead of nav_lon and nav_lat - at the same time as 1D dimensions x and y. This causes all manner of problems with xarray so the best thing is just to create a new file from scratch and copy over the variables one at a time, renaming as needed.
def fix_mesh_mask (file_in, file_out):

    id_in = nc.Dataset(file_in, 'r')
    id_out = nc.Dataset(file_out, 'w')

    print('Setting up dimensions')
    for dim in ['x', 'y', 'nav_lev']:
        id_out.createDimension(dim, id_in.dimensions[dim].size)
        id_out.createVariable(dim, 'f8', (dim))
        id_out.variables[dim][:] = np.arange(id_in.dimensions[dim].size)
    id_out.createDimension('time_counter', None)

    for var in id_in.variables:
        if var in ['nav_lev', 'time_counter']:
            continue
        print('Writing '+var)
        if var == 'x':
            var_new = 'nav_lon'
        elif var == 'y':
            var_new = 'nav_lat'
        else:
            var_new = var
        id_out.createVariable(var_new, id_in.variables[var].dtype, id_in.variables[var].dimensions)
        id_out.variables[var_new][:] = id_in.variables[var][:]

    id_in.close()
    id_out.close()


# Read bottom temperature and salinity from the Schmidtko dataset.
def read_schmidtko (schmidtko_file='/gws/ssde/j25b/terrafirma/kaight/input_data/schmidtko_TS.txt', eos='teos10'):

    import gsw
    # Read Schmidtko data on continental shelf
    obs = np.loadtxt(schmidtko_file, dtype=str)[1:,:]
    obs_lon_vals = obs[:,0].astype(float)
    obs_lat_vals = obs[:,1].astype(float)
    obs_depth_vals = obs[:,2].astype(float)
    obs_temp_vals = obs[:,3].astype(float)
    obs_salt_vals = obs[:,5].astype(float)
    num_obs = obs_temp_vals.size
    # Grid it
    obs_lon = np.unique(obs_lon_vals)
    obs_lat = np.unique(obs_lat_vals)
    obs_temp = np.zeros([obs_lat.size, obs_lon.size]) - 999
    obs_salt = np.zeros([obs_lat.size, obs_lon.size]) - 999
    obs_depth = np.zeros([obs_lat.size, obs_lon.size]) - 999
    for n in range(num_obs):
        j = np.argwhere(obs_lat==obs_lat_vals[n])[0][0]
        i = np.argwhere(obs_lon==obs_lon_vals[n])[0][0]
        if obs_temp[j,i] != -999:
            raise Exception('Multiple values at same point')
        obs_temp[j,i] = obs_temp_vals[n]
        obs_salt[j,i] = obs_salt_vals[n]
        obs_depth[j,i] = obs_depth_vals[n]
    obs_temp = xr.DataArray(obs_temp, coords=[obs_lat, obs_lon], dims=['lat', 'lon'])
    obs_temp = obs_temp.where(obs_temp!=-999)
    obs_salt = xr.DataArray(obs_salt, coords=[obs_lat, obs_lon], dims=['lat', 'lon'])
    obs_salt = obs_salt.where(obs_salt!=-999)
    obs_depth = xr.DataArray(obs_depth, coords=[obs_lat, obs_lon], dims=['lat', 'lon'])
    obs_depth = obs_depth.where(obs_depth!=-999)
    if eos == 'eos80':
        # Convert from TEOS10 to EOS80
        # Have conservative temperature and absolute salinity; want potential temperature and practical salinity
        # Pressure in dbar is approx depth in m
        obs_press = np.abs(obs_depth)
        lon_2d, lat_2d = np.meshgrid(obs_lon, obs_lat)
        obs_temp = gsw.pt_from_CT(obs_salt, obs_temp)
        obs_salt = gsw.SP_from_SA(obs_salt, obs_press, lon_2d, lat_2d)
    obs = xr.Dataset({'temp':obs_temp, 'salt':obs_salt})
    return obs


# Read World Ocean Atlas 2018 data for the deep ocean.
def read_woa (woa_files='/gws/ssde/j25b/terrafirma/kaight/input_data/WOA18/woa18_decav_*00_04.nc', eos='teos10'):

    import gsw
    woa = xr.open_mfdataset(woa_files, decode_times=False)
    # Find seafloor depth
    woa_bathy = -1*woa['t_an'].coords['depth'].where(woa['t_an'].notnull()).max(dim='depth')
    # Pressure in dbar is approx depth in m
    woa_press = np.abs(woa_bathy)
    # Mask shallow regions in the Amundsen and Bellingshausen Seas where weird things happen
    mask = (woa['lon'] >= -130)*(woa['lon'] <= -60)*(woa_bathy >= -500)
    # Now get bottom temperature and salinity    
    woa_temp = select_bottom(woa['t_an'], 'depth').where(~mask)
    woa_salt = select_bottom(woa['s_an'], 'depth').where(~mask)
    # Regardless of EOS will need absolute salinity at least temporarily
    abs_salt = gsw.SA_from_SP(woa_salt, woa_press, woa['lon'], woa['lat'])
    if eos == 'eos80':
        # Convert to EOS80
        # Have in-situ temperature and practical salinity; want potential temperature        
        woa_temp = gsw.pt0_from_t(abs_salt, woa_temp, woa_press)
    elif eos == 'teos10':
        # Convert to TEOS10
        # Want conservative temperature and absolute salinity
        woa_salt = abs_salt
        woa_temp = gsw.CT_from_t(woa_salt, woa_temp, woa_press)
    # Now wrap up into a new Dataset
    woa = xr.Dataset({'temp':woa_temp, 'salt':woa_salt}).drop_vars('depth').squeeze()
    return woa

def read_dutrieux(fileT='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/observations/pierre-dutrieux/ASEctd_griddedMean_PT.nc',
                fileS='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/observations/pierre-dutrieux/ASEctd_griddedMean_S.nc', 
                eos='teos10'):
    import gsw 
    # Load observations on Amundsen Shelf from Pierre Dutrieux
    obs    = xr.open_mfdataset([fileT, fileS])
    obs_ds = obs.rename({'PTmean':'PotTemp', 'Smean':'PracSal', 'longrid':'lon', 'latgrid':'lat', 
                         'pvec':'pressure', 'depthvec':'depth'})

    # Convert units to TEOS10
    if eos=='teos10':
        obs_AS   = convert_to_teos10(obs_ds, var='PracSal')
        obs_CT   = convert_to_teos10(obs_ds, var='PotTemp')
        obs_conv = xr.Dataset({'ConsTemp':(('lat','lon','depth'), obs_CT.values), 'AbsSal':(('lat','lon','depth'), obs_AS.values)})
        obs_conv = obs_conv.assign_coords({'lon':obs_ds.lon.isel(indexlat=0).values,
                                           'lat':obs_ds.lat.isel(indexlon=0).values,
                                           'depth':obs_ds.depth.values})
    else:
        obs_conv = obs_ds.copy()
        
    return obs_conv

def read_zhou(fileT='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/observations/shenjie-zhou/SO_CT_monthly/Merge_all_SO_CT_10dbar_monthly_1.nc',
              fileS='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/observations/shenjie-zhou/SO_SA_monthly/Merge_all_SO_SA_10dbar_monthly_1.nc', 
              eos='teos10'):
    
    import gsw 
    # Load observations on Amundsen Shelf from Pierre Dutrieux
    obs    = xr.open_mfdataset([fileT, fileS], chunks='auto', engine='netcdf4')
    obs_ds = obs.rename({'ct':'ConsTemp', 'sa':'AbsSal', 'pres':'pressure', 'NB_X':'x', 'NB_Y':'y', 'NB_LEV':'z'})

    # does not provide depth, so calculate:
    depth  = gsw.z_from_p(obs_ds.pressure, obs_ds.lat)
    obs_ds = obs_ds.assign({'depth':abs(depth)}).squeeze()

    # Convert units to TEOS10
    if eos!='teos10':
        raise Exception('Observations are in TEOS-10 units, will need to convert')
        
    return obs_ds


def read_zhou_bottom_climatology (in_file='/gws/ssde/j25b/terrafirma/kaight/input_data/shenjie_climatology_bottom_TS.nc', eos='teos10'):

    import gsw
    ds = xr.open_dataset(in_file)
    def set_var (var_name):
        return xr.DataArray(np.transpose(ds[var_name].data), coords=[ds['lat'].data, ds['lon'].data], dims=['lat', 'lon'])
    temp = set_var('bottom_temperature')
    salt = set_var('bottom_salinity')
    depth = set_var('bathymetry')
    if eos == 'eos80':
        press = np.abs(depth)
        lon_2d, lat_2d = np.meshgrid(ds['lon'], ds['lat'])
        temp = gsw.pt_from_CT(salt, temp)
        salt = gsw.SP_from_SA(salt, press, lon_2d, lat_2d)
    obs = xr.Dataset({'temp':temp, 'salt':salt})
    return obs    
    

# Generate the file name and starting/ending index for a CESM variable for the given experiment, year and ensemble member.
# for example, expt = 'LE2', ensemble_member='1011.001', domain ='atm', freq = 'daily'
def find_cesm2_file(expt, var_name, domain, freq, ensemble_member, year,
                    base_dir='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/climate-forcing/CESM2/'):

    import glob
    from datetime import datetime

    if expt not in ['LE2', 'piControl', 'SF-AAER', 'SF-BMB', 'SF-GHG', 'SF-EE', 'SF-xAER']:
        raise Exception(f'Invalid experiment {expt}')
    if freq not in ['daily', 'monthly', '3-hourly']:
            raise Exception(f'Frequency can be either daily, monthly, or 3-hourly, but is specified as {freq}')
    if expt == 'LE2':
        if ensemble_member not in cesm2_ensemble_members:
            raise Exception(f'Ensemble member {ensemble_member} is not available')

    # list of single-forcing experiment names
    SF_expts = ['SF-AAER', 'SF-BMB', 'SF-GHG', 'SF-EE', 'SF-xAER']

    # define start of filename stub
    if expt == 'LE2':
        if (year <= 2014) and (year >= 1850):
            start_stub = 'b.e21.BHISTsmbb.f09_g17.'
        elif year <= 2100 and (year >=2015):
            start_stub = 'b.e21.BSSP370smbb.f09_g17.'
        else:
            raise Exception('Not a valid year for the specified experiment and ensemble member')
    elif expt=='piControl':
        start_stub = 'b.e21.B1850.f09_g17.CMIP6'
    elif expt in SF_expts:
        if expt=='SF-xAER':
            if (year <= 2014):
                start_stub = 'b.e21.BHISTcmip6.f09_g17.CESM2'
            else:
                start_stub = 'b.e21.BSSP370cmip6.f09_g17.CESM2'
        else:
            start_stub = 'b.e21.B1850cmip6.f09_g17.CESM2'

    if domain == 'atm':
        if freq == 'monthly':
            domain_stub = '.cam.h0.'
        elif freq == 'daily':
            domain_stub = '.cam.h1.'
        elif freq == '3-hourly':
            domain_stub = '.cam.h3.'
    elif domain in ['oce', 'ocn']:
        domain_stub = '.pop.h.'
    elif domain == 'ice':
        domain_stub = '.cice.h.'

    # find the file that contains the requested year
    if freq == 'daily':
        str_format = '%Y%m%d'
    elif freq == 'monthly':
        str_format = '%Y%m'
    elif freq == '3-hourly':
        str_format = '%Y%m%d%H'
 
    if expt=='LE2':
        file_list  = glob.glob(f'{base_dir}{expt}/raw/{start_stub}{expt}-{ensemble_member}{domain_stub}{var_name}*')
    elif expt=='piControl':
        file_list  = glob.glob(f'{base_dir}{expt}/raw/{start_stub}-{expt}.001{domain_stub}{var_name}*')    
    elif expt in SF_expts:
        if (year < 2015):
            file_list  = glob.glob(f'{base_dir}{expt}/raw/{start_stub}-{expt}.{ensemble_member}{domain_stub}{var_name}*')
        else:
            if expt=='SF-xAER':
                file_list = glob.glob(f'{base_dir}{expt}/raw/{start_stub}-{expt}.{ensemble_member}{domain_stub}{var_name}*')
            else:
                file_list  = glob.glob(f'{base_dir}{expt}/raw/{start_stub}-{expt}-SSP370.{ensemble_member}{domain_stub}{var_name}*') 

    found_date = False
    for file in file_list:
        date_range = (file.split(f'.{var_name}.')[1]).split('.nc')[0]
        start_date = datetime.strptime(date_range.split('-')[0], str_format)
        end_date   = datetime.strptime(date_range.split('-')[1], str_format)
        if freq=='monthly':
            if (year <= end_date.year) and (year >= start_date.year):
                found_date=True
                break
        else:
            if (datetime(year,12,31) <= end_date) and (datetime(year,1,1) >= start_date): # found the file we're looking for
                found_date = True
                break

    if not found_date:
        raise Exception(f'File for {var_name}, {ensemble_member} requested year ({year}) not found, double-check that it exists?')

    if expt=='LE2':
        file_path = f'{base_dir}{expt}/raw/{start_stub}{expt}-{ensemble_member}{domain_stub}{var_name}.{date_range}.nc'
    elif expt=='piControl':
        file_path = f'{base_dir}{expt}/raw/{start_stub}-{expt}.001{domain_stub}{var_name}.{date_range}.nc'
    elif expt in SF_expts:
        if (year < 2015):
            file_path = f'{base_dir}{expt}/raw/{start_stub}-{expt}.{ensemble_member}{domain_stub}{var_name}.{date_range}.nc'
        else:
            if expt=='SF-xAER':
                file_path = f'{base_dir}{expt}/raw/{start_stub}-{expt}.{ensemble_member}{domain_stub}{var_name}.{date_range}.nc'
            else:
                file_path = f'{base_dir}{expt}/raw/{start_stub}-{expt}-SSP370.{ensemble_member}{domain_stub}{var_name}.{date_range}.nc'

    return file_path

# Generate the postprocessing file name for a CESM variable for the given experiment, year and ensemble member.
# for example, expt = 'LE2', ensemble_member='1011.001'
# Inputs:
# bias_corr (optional) : boolean indicating whether you are looking for the bias corrected files
def find_processed_cesm2_file(expt, var_name, ensemble_member, year, freq='daily', highres=False,
                              base_dir='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/climate-forcing/CESM2/'):

    import glob
    from datetime import datetime

    if expt not in ['LE2', 'piControl', 'SF-AAER', 'SF-BMB', 'SF-GHG', 'SF-EE', 'SF-xAER']:
        raise Exception(f'Invalid experiment {expt}')
    if expt=='LE2':
        if ensemble_member not in cesm2_ensemble_members:
            raise Exception(f'Ensemble member {ensemble_member} is not available')
        if (year > 2100) or (year < 1850):
            raise Exception('Not a valid year for the specified experiment and ensemble member')
    elif expt == 'piControl':
        if year > 2000:
            raise Exception('Not a valid year for the specified experiment and ensemble member')

    # set the experiment directory
    if highres:
        file_dir = f'{base_dir}{expt}/processed_highres/ens{ensemble_member}/'
        file_list = glob.glob(f'{file_dir}CESM2-{expt}_ens{ensemble_member}_eANT025_{freq}_{var_name}_y*')
    else:
        file_dir = f'{base_dir}{expt}/processed/ens{ensemble_member}/'
        file_list = glob.glob(f'{file_dir}CESM2-{expt}_ens{ensemble_member}_{freq}_{var_name}_y*')

    found_date = False
    file_path  = []
    for file in file_list:
        file_year = datetime.strptime((file.split(f'_{var_name}_y')[1])[0:4], '%Y').year
        if (year == file_year): # found the file we're looking for
            found_date = True
            file_path.append(file)

    if not found_date:
        raise Exception(f'File for {var_name}, {ensemble_member} requested year ({year}) not found, double-check that it exists?')
    
    if len(file_path)==1:
        return file_path[0]
    else:
        return file_path
