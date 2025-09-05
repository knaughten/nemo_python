
# Analysing TerraFIRMA overshoot simulations with UKESM1.1-ice (NEMO 3.6)

import xarray as xr
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import os
import shutil
import subprocess
import numpy as np
import cf_xarray as cfxr
import re
import datetime
from scipy.stats import ttest_ind, linregress, ttest_1samp
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..timeseries import update_simulation_timeseries, update_simulation_timeseries_um, check_nans, fix_missing_months, calc_timeseries, overwrite_file
from ..plots import timeseries_by_region, timeseries_by_expt, finished_plot, timeseries_plot, circumpolar_plot
from ..plot_utils import truncate_colourmap
from ..utils import moving_average, add_months, rotate_vector, polar_stereo, convert_ismr, bwsalt_abs
from ..grid import region_mask, calc_geometry, build_ice_mask
from ..constants import line_colours, region_names, deg_string, gkg_string, months_per_year, rho_fw, rho_ice, sec_per_year, vaf_to_gmslr
from ..file_io import read_schmidtko, read_woa, read_zhou_bottom_climatology
from ..interpolation import interp_latlon_cf, interp_grid
from ..diagnostics import barotropic_streamfunction
from ..plot_utils import set_colours, latlon_axes
from ..bisicles_utils import read_bisicles

# Global dictionaries of suites - update these as more suites become available!

# Dictionary of which suites correspond to which scenario
suites_by_scenario = {'piControl_static_ice' : ['cs495'],
                      'piControl' : ['cs568'],
                      'ramp_up' : ['cx209', 'cw988', 'cw989', 'cw990'],
                      'ramp_up_static_ice': ['cz826'],
                      '1.5K_stabilise': ['cy837','cz834','da087'],
                      '1.5K_ramp_down': ['da697', 'dc052', 'dc248'],
                      '2K_stabilise': ['cy838','cz855','da266'],
                      '2K_ramp_down': ['di335','dc051', 'da800', 'dc565', 'df025', 'df027'],
                      '2.5K_stabilise' : ['cz374','cz859'],
                      '3K_stabilise' : ['cz375','db587','db597'],
                      '3K_ramp_down' : ['dc032', 'dc249', 'df453', 'df028', 'do136', 'df021'],  # do136 branched off df023 before first geometry problem
                      '4K_stabilise' : ['cz376','db723','db733'],
                      '4K_ramp_down' : ['da892', 'do135', 'dh859', 'dd210', 'dh541'],  # do135 branched off dc123 before first geometry problem
                      '5K_stabilise' : ['cz377','db731','dc324'],
                      '5K_ramp_down' : ['dc251', 'dc130', 'dg095', 'dg093', 'dg094'],
                      '6K_stabilise' : ['cz378'],
                      '6K_ramp_down' : ['de943', 'de962', 'de963', 'dm357', 'dm358', 'dm359'],
                      '2K_restabilise': ['dn822']}
# Choose one ensemble member of each main scenario type for plotting a less-messy timeseries.
suites_by_scenario_1ens = {'ramp_up': 'cx209',  # First ensemble member for ramp-up and all stabilisation
                           '1.5K_stabilise': 'cy837',
                           '2K_stabilise': 'cy838',
                           '3K_stabilise': 'cz375',
                           '4K_stabilise': 'cz376',
                           '5K_stabilise': 'cz377',
                           '6K_stabilise': 'cz378',
                           '1.5K_ramp_down': 'da697',  # 50y overshoot, -8 Gt/y for all ramp-downs
                           '2K_ramp_down': 'di335',
                           '3K_ramp_down': 'df453',
                           '4K_ramp_down': 'da892',
                           '5K_ramp_down': 'dc251',
                           '6K_ramp_down': 'de943'}
# Dictionary of ramp-down rates
suites_ramp_down_rates = {'8 Gt/y' : ['di335', 'da800', 'da697', 'da892', 'df453', 'dc251', 'de943', 'dg093', 'dm357'],
                          '4 Gt/y' : ['dc051', 'dc052', 'dc248', 'dc249', 'dc565', 'dd210', 'dc032', 'df028', 'do135', 'dc130', 'de962', 'dg094', 'dm358'],
                          '2 Gt/y' : ['df025', 'df027', 'df021', 'do136', 'dh541', 'dh859', 'de963', 'dg095', 'dm359']}
# Dictionary of overshoot lengths
suites_overshoot_lengths = {'50 years': ['da697', 'dc052', 'di335', 'dc051', 'df025', 'df453', 'df028', 'df021', 'da892', 'do135', 'dh541', 'dc251', 'dc130', 'de943', 'de962', 'de963'],
                            '200 years': ['dc248', 'da800', 'dc565', 'df027', 'dc249', 'do136', 'dh859', 'dd210', 'dg093', 'dg094', 'dg095', 'dm357', 'dm358', 'dm359'],
                            '30 years': ['dc032']}

# Dictionary of which suites branch from which. None means it's a ramp-up suite (so branched from a piControl run, but we don't care about that for the purposes of integrated GW)
suites_branched = {'cx209':None, 'cw988':None, 'cw989':None, 'cw990':None, 'cz826':None, 'cy837':'cx209', 'cy838':'cx209', 'cz374':'cx209', 'cz375':'cx209', 'cz376':'cx209', 'cz377':'cx209', 'cz378':'cx209', 'cz855':'cw988', 'cz859':'cw988', 'db587':'cw988', 'db723':'cw988', 'db731':'cw988', 'da087':'cw989', 'da266':'cw989', 'db597':'cw989', 'db733':'cw989', 'dc324':'cw989', 'da800':'cy838', 'da697':'cy837', 'da892':'cz376', 'dc051':'cy838', 'dc052':'cy837', 'dc248':'cy837', 'dc249':'cz375', 'dc251':'cz377', 'dc032':'cz375', 'do135':'cz376', 'dc130':'cz377', 'di335':'cy838', 'df453':'cz375', 'dc565':'cy838', 'dd210':'cz376', 'df028':'cz375', 'df025':'cy838', 'df027':'cy838', 'df021':'cz375', 'do136':'cz375', 'dh541':'cz376', 'dh859':'cz376', 'de943':'cz378', 'de962':'cz378', 'de963':'cz378', 'dg093':'cz377', 'dg094':'cz377', 'dg095':'cz377', 'dm357':'cz378', 'dm358':'cz378', 'dm359':'cz378', 'dn822':'da892'}

tipping_threshold = -1.9  # If cavity mean temp is warmer than surface freezing point, it's tipped
temp_correction = 1.5450286133565283 # Precomputed by warming_implied_by_salinity_bias()
temp_correction_lower = 0.5716454706352944
temp_correction_upper = 2.394992374100425  # 10-90% bounds precomputed by temp_correction_uncertainty
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

# End global vars


# Call update_simulation_timeseries for the given suite ID
def update_overshoot_timeseries (suite_id, base_dir='./', domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'):

    # Construct list of timeseries types for T-grid
    regions = ['all', 'ross', 'filchner_ronne', 'west_antarctica', 'east_antarctica']
    var_names = ['massloss', 'draft', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_bwsalt', 'shelf_bwSA']
    timeseries_types = []
    # All combinations of region and variable
    for region in regions:
        for var in var_names:
            timeseries_types.append(region+'_'+var)

    update_simulation_timeseries(suite_id, timeseries_types, timeseries_file='timeseries.nc', sim_dir=base_dir+'/'+suite_id+'/', freq='m', halo=True, gtype='T')

    # Now for UM
    update_simulation_timeseries_um(suite_id, ['global_mean_sat'], timeseries_file='timeseries_um.nc', sim_dir=base_dir+'/'+suite_id+'/', stream='p5')


# Call for all simulations (add to the list of suite IDs as needed)
def update_overshoot_timeseries_all (base_dir='./', domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'):

    for scenario in suites_by_scenario:
        for suite_id in suites_by_scenario[scenario]:
            update_overshoot_timeseries(suite_id, base_dir=base_dir, domain_cfg=domain_cfg)


# Calculate a new timeseries variable(s) for the given suite, and then concatenate it with the existing corresponding timeseries file. After running this, add the variable(s) to the list in update_overshoot_timeseries.
def new_timeseries_var (suite_id, timeseries_types, timeseries_file_new, timeseries_file='timeseries.nc', base_dir='./', domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'):

    # Calculate a new file
    if timeseries_file == 'timeseries_um.nc':
        # Atmospheric variable
        update_simulation_timeseries_um(suite_id, timeseries_types, timeseries_file=timeseries_file_new, sim_dir=base_dir+'/'+suite_id+'/', stream='p5')
    else:
        # Ocean variable
        if timeseries_file == 'timeseries.nc':
            gtype = 'T'
        elif timeseries_file == 'timeseries_u.nc':
            gtype = 'U'
        else:
            raise Exception('unknown gtype - add another case to the code?')
        update_simulation_timeseries(suite_id, timeseries_types, timeseries_file=timeseries_file_new, sim_dir=base_dir+'/'+suite_id+'/', freq='m', halo=True, gtype=gtype, domain_cfg=domain_cfg)

    # Now concatenate with existing file
    print('Merging with '+timeseries_file)
    os.rename(suite_id+'/'+timeseries_file, suite_id+'/tmp_'+timeseries_file)
    ds = xr.open_mfdataset([suite_id+'/tmp_'+timeseries_file, suite_id+'/'+timeseries_file_new], decode_times=time_coder)
    ds.to_netcdf(suite_id+'/'+timeseries_file)
    os.remove(suite_id+'/tmp_'+timeseries_file)
    os.remove(suite_id+'/'+timeseries_file_new)
    ds.close()    


# Plot timeseries by region for all variables in one simulation.
def plot_all_timeseries_by_region (suite_id, regions=['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross'], var_names=['massloss', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_temp', 'shelf_salt', 'temp_btw_200_700m', 'salt_btw_200_700m'], colours=None, timeseries_file='timeseries.nc', base_dir='./', smooth=24, fig_dir=None):

    while suite_id.endswith('/'):
        suite_id = suite_id[:-1]

    for var in var_names:
        # Special treatment of 200-700m variables as these are only valid for certain regions
        if var.endswith('200_700m'):
            regions_use = []
            for region in ['amundsen_sea', 'bellingshausen_sea']:
                if region in regions:
                    regions_use.append(region)
        else:
            regions_use = regions
        if len(regions) == 0:
            continue                    
        timeseries_by_region(var, base_dir+'/'+suite_id+'/', regions=regions_use, colours=colours, timeseries_file=timeseries_file, smooth=smooth, fig_name=None if fig_dir is None else (fig_dir+'/'+var+'_'+suite_id+'.png'))


# Set the list of experiments, names for the legend, and colours to use for timeseries etc.
# If separate_stages=True, the ramp down and re-stabilise stages will be separated out from the initial stabilisations.
# If only_up=True, only the ramp-up and stabilise suites will be considered.
# If only_down=True, only the ramp-down and re-stabilise suites will be considered.
def set_expt_list (separate_stages=False, only_up=False, only_down=False):

    sim_names = []
    colours = []
    sim_dirs = []

    # Inner function to add a new suite or ensemble to the lists
    def add_ens (name, colour, dirs):
        sim_names.append(name)
        colours.append(colour)
        sim_dirs.append(dirs)
        
    # Inner function to combine two strings and/or lists into lists
    def combine_ens (ens1, ens2):
        if isinstance(ens1, str):
            all_ens = [ens1]
        else:
            all_ens = ens1
        if isinstance(ens2, str):
            all_ens += [ens2]
        else:
            all_ens += ens2
        return all_ens    

    # Inner function to process stabilisation targets. 7 arguments:
    # gw: string with the global warming level (eg '2K')
    # stabilise, ramp_down, restabilise: strings or lists of strings containing suite(s) corresponding to stabilisation, ramp-down, and restabilisation (at 0K) respectively. They can be None if no such suites exist.
    # colour1, colour2, colour3: colour names corresponding to stabilise, ramp_down, and restabilise. Can be None if the corresponding suite lists are None.
    def add_gw_level (gw, stabilise, ramp_down, restabilise, colour1, colour2, colour3):
        if separate_stages:
            # Keep the distinct colours regardless of values of only_up, only_down
            if stabilise is not None and not only_down:
                add_ens(gw+' stabilise', colour1, stabilise)
            if ramp_down is not None and not only_up:
                add_ens(gw+' ramp down', colour2, ramp_down)
            if restabilise is not None and not only_up:
                add_ens(gw+' re-stabilise', colour3, restabilise)
        else:
            # Only use colour1
            if only_up and stabilise is not None:
                add_ens(gw+' stabilise', colour1, stabilise)
            if only_down and ramp_down is not None:
                if restabilise is None:
                    add_ens(gw+' ramp down', colour1, ramp_down)
                else:
                    add_ens(gw+' ramp down & re-stabilise', colour1, combine_ens(ramp_down, restabilise))
            if not only_up and not only_down:
                name = gw+' stabilise'
                ens = stabilise
                if ramp_down is not None:
                    ens = combine_ens(stabilise, ramp_down)
                    name += ' & ramp_down'
                    if restabilise is not None:
                        ens = combine_ens(ens, restabilise)
                        name += ' & re-stabilise'
                add_ens(name, colour1, ens)

    # Now add the suites
    if not only_down:
        add_ens('ramp up', 'Black', suites_by_scenario['ramp_up'])
        add_ens('ramp up static ice', 'DarkGrey', suites_by_scenario['ramp_up_static_ice'])
    add_gw_level('1.5K', suites_by_scenario['1.5K_stabilise'], suites_by_scenario['1.5K_ramp_down'], None, 'DarkMagenta', 'MediumOrchid', None)
    add_gw_level('2K', suites_by_scenario['2K_stabilise'], suites_by_scenario['2K_ramp_down'], suites_by_scenario['2K_restabilise'], 'Blue', 'CornflowerBlue', 'LightBlue')
    add_gw_level('2.5K', suites_by_scenario['2.5K_stabilise'], None, None, 'DarkCyan', None, None)
    add_gw_level('3K', suites_by_scenario['3K_stabilise'], suites_by_scenario['3K_ramp_down'], None, 'DarkGreen', 'DarkSeaGreen', None)
    add_gw_level('4K', suites_by_scenario['4K_stabilise'], suites_by_scenario['4K_ramp_down'], None, 'GoldenRod', 'Gold', None)
    add_gw_level('5K', suites_by_scenario['5K_stabilise'], suites_by_scenario['5K_ramp_down'], None, 'Chocolate', 'LightSalmon', None)
    add_gw_level('6K', suites_by_scenario['6K_stabilise'], None, None, 'Crimson', None, None)

    return sim_names, colours, sim_dirs


# Set the list of experiments, names for the legend, and timeseries to use - but only separating (by names/colours) the ramp-up, stabilise, and ramp-down scenarios, rather than showing all the different stabilisation targets individually. Also, do not include static ice cases.
def minimal_expt_list (one_ens=False):

    if one_ens:
        suite_list = suites_by_scenario_1ens
    else:
        suite_list = suites_by_scenario

    keys = ['ramp_up', '_stabilise', '_ramp_down']
    sim_names = ['ramp-up', 'zero-emission', 'ramp-down']
    colours = ['Crimson', 'Grey', 'DodgerBlue']
    sim_dirs = []
    for key in keys:
        dirs = []
        for scenario in suite_list:
            if 'static_ice' in scenario:
                continue
            if key in scenario:
                if isinstance(suite_list[scenario], str):
                    dirs += [suite_list[scenario]]
                else:
                    dirs += suite_list[scenario]
        sim_dirs.append(dirs)

    return sim_names, colours, sim_dirs


# Plot timeseries by experiment for all variables and regions, in all experiments.
def plot_all_timeseries_by_expt (base_dir='./', regions=['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross'], var_names=['massloss', 'draft', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_temp', 'shelf_salt', 'temp_btw_200_700m', 'salt_btw_200_700m', 'drake_passage_transport', 'global_mean_sat'], timeseries_file='timeseries.nc', timeseries_file_u='timeseries_u.nc', timeseries_file_um='timeseries_um.nc', smooth=24, fig_dir=None):

    sim_names, colours, sim_dirs = set_expt_list(separate_stages=True)

    # Now construct master list of variables
    var_names_all = []
    for region in regions:
        for var in var_names:
            if var in ['drake_passage_transport', 'global_mean_sat']:
                # Special cases with no region
                if var not in var_names_all:
                    var_names_all.append(var)
            elif var.endswith('200_700m'):
                # Special cases where only some regions defined
                if region in ['amundsen_sea', 'bellingshausen_sea']:
                    var_names_all.append(region+'_'+var)
            else:
                # Every combination of region and variable
                var_names_all.append(region+'_'+var)

    for var in var_names_all:
        if var == 'drake_passage_transport':
            fname = timeseries_file_u
        elif var == 'global_mean_sat':
            fname = timeseries_file_um
        else:
            fname = timeseries_file
        timeseries_by_expt(var, sim_dirs, sim_names=sim_names, colours=colours, timeseries_file=fname, smooth=smooth, linewidth=1, fig_name=None if fig_dir is None else (fig_dir+'/'+var+'_master.png'))


# Helper function to read global mean SAT
def global_temp (suite, base_dir='./', timeseries_file_um='timeseries_um.nc'):
    ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file_um, decode_times=time_coder)
    sat = ds['global_mean_sat']
    ds.close()
    return sat
# Helper function to get baseline PI global temp
def pi_baseline_temp (pi_suite='cs495', base_dir='./'):
    return global_temp(pi_suite, base_dir=base_dir).mean()
# Helper function to get global warming relative to preindustrial
def global_warming (suite, pi_suite='cs495', base_dir='./'):
    baseline = pi_baseline_temp(pi_suite=pi_suite, base_dir=base_dir)
    return global_temp(suite, base_dir=base_dir) - baseline    


# Calculate the integrated global warming relative to preindustrial mean, in Kelvin-years, for the given suite (starting from the beginning of the relevant ramp-up simulation). Returns a timeseries over the given experiment, with the first value being the sum of all branched-from experiments before then.
def integrated_gw (suite, pi_suite='cs495', timeseries_file_um='timeseries_um.nc', base_dir='./'):

    # Inner function to read global mean SAT
    def global_mean_sat (suite):
        ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file_um, decode_times=time_coder)
        sat = ds['global_mean_sat']
        ds.close()
        return sat

    # Get timeseries of global warming relative to preindustrial, in this suite
    gw_level = global_warming(suite, pi_suite=pi_suite, base_dir=base_dir)
    # Indefinite integral over time
    integrated_gw = (gw_level/months_per_year).cumsum(dim='time_centered')

    # Now add on definite integrals of all branched-from suites before that
    prev_suite = suites_branched[suite]
    while prev_suite is not None:
        # Find the starting date of the current suite
        start_date = gw_level.time_centered[0]
        # Now calculate global warming timeseries over the previous suite
        gw_level = global_warming(prev_suite, pi_suite=pi_suite, base_dir=base_dir)
        # Find the time index at the branch point
        time_branch = np.argwhere(gw_level.time_centered.data == start_date.data)[0][0]
        # Integrate from the beginning to just before that point
        integrated_gw += (gw_level.isel(time_centered=slice(0,time_branch))/months_per_year).sum(dim='time_centered')
        # Prepare for next iteration of loop
        suite = prev_suite
        prev_suite = suites_branched[suite]

    return integrated_gw


# Plot timeseries of integrated global warming in every experiment.
def plot_integrated_gw (base_dir='./', timeseries_file_um='timeseries_um.nc', pi_suite='cs495', fig_name=None):

    sim_names, colours, sim_dirs = set_expt_list(separate_stages=True)
    datas = []
    labels_plot = []
    colours_plot = []
    for expt, label, colour in zip(sim_dirs, sim_names, colours):
        if isinstance(expt, str):
            expt = [expt]
        num_ens = len(expt)
        for suite in expt:
            datas.append(integrated_gw(suite, pi_suite=pi_suite, timeseries_file_um=timeseries_file_um, base_dir=base_dir))
        labels_plot += [label] + [None]*(num_ens-1)
        colours_plot += [colour]*num_ens
    timeseries_plot(datas, labels=labels_plot, colours=colours_plot, title='Integrated global warming relative to preindustrial', units='Kelvin-years', linewidth=1, fig_name=fig_name)


# Given two timeseries, trim them to line up along the time axis and be the same length.
def align_timeseries (data1, data2, time_coord='time_centered'):

    # Find most restrictive endpoints
    date_start = max(data1[time_coord][0], data2[time_coord][0])
    date_end = min(data1[time_coord][-1], data2[time_coord][-1])
    # Inner function to trim
    def trim_timeseries (A):
        t_start = np.argwhere(A[time_coord].data == date_start.data)[0][0]
        t_end = np.argwhere(A[time_coord].data == date_end.data)[0][0]
        return A.isel({time_coord:slice(t_start, t_end+1)})
    data1 = trim_timeseries(data1)
    data2 = trim_timeseries(data2)
    return data1, data2
    

# Plot the timeseries of one or more experiments/ensembles (expts can be a string, a list of strings, or a list of lists of string) and one variable against global warming level (relative to preindustrial mean in the given PI suite, unless offsets is not None).
# Can also use an ocean variable to plot against instead of GW level (eg cavity temperature) with alternate_var='something' (assumed to be within timeseries_file).
# Can also set offsets as a list of the same shape as expts, with different global warming baselines for each - this is still on top of PI mean, so eg pass 3 for 3K above PI.
# Can also set highlight as a suite-string showing a trajectory which should be plotted twice as thick. If highlight_arrows is True, will also plot some arrows along this trajectory: must set arrow_loc = list of lists of global temperatures to plot at for each suite in trajectory.
def plot_by_gw_level (expts, var_name, pi_suite='cs495', base_dir='./', fig_name=None, timeseries_file='timeseries.nc', timeseries_file_um='timeseries_um.nc', smooth=24, labels=None, colours=None, linewidth=1, title=None, units=None, ax=None, offsets=None, alternate_var=None, temp_correct=0, highlight=None, highlight_arrows=True, arrow_loc=None):

    new_ax = ax is None

    if isinstance(expts, str):
        # Just one suite - generalise
        expts = [expts]
    num_expt = len(expts)
    if colours is None:
        if num_expt <= len(line_colours):
            colours = line_colours[:num_expt]
    if labels is None:
        labels = [None]*num_expt            

    if alternate_var is None:
        # Get baseline global mean SAT
        baseline_temp = pi_baseline_temp(pi_suite=pi_suite, base_dir=base_dir)

    if offsets is None:
        # Set up dummy list of 0s, same shape as expts
        offsets = []
        for expt in expts:
            if isinstance(expt, str):
                offsets.append([0])
            else:
                offsets.append([0]*len(expt))

    if highlight is None:
        highlight_suites = []
    else:
        highlight_suites = highlight.split('-')
        if highlight_arrows and arrow_loc is None:
            raise Exception('Must set arrow_loc')
    num_highlight = len(highlight_suites)
    
    gw_levels = []
    datas = []
    labels_plot = []
    colours_plot = []
    lw_plot = []
    highlight_stype = [None]*num_highlight
    highlight_colours = [None]*num_highlight
    for expt, label, colour, expt_offsets in zip(expts, labels, colours, offsets):
        if isinstance(expt, str):
            # Generalise to ensemble of 1
            expt = [expt]
        num_ens = len(expt)
        for suite, offset in zip(expt, expt_offsets):
            if np.isnan(offset):
                # Flag to skip this suite
                continue            
            # Join with the parent suite so there isn't a gap when simulations branch and smoothing is applied.
            if suite in suites_branched and suites_branched[suite] is not None:
                suite_list = [suites_branched[suite], suite]
            else:
                suite_list = [suite]
            # Read global mean SAT in this suite and convert to GW level
            if alternate_var is None:
                gw_level = build_timeseries_trajectory(suite_list, 'global_mean_sat', base_dir=base_dir, timeseries_file=timeseries_file_um, offset=-baseline_temp-offset)
            else:
                gw_level = build_timeseries_trajectory(suite_list, alternate_var, base_dir=base_dir, timeseries_file=timeseries_file)
            # Read the variable
            data = build_timeseries_trajectory(suite_list, var_name, base_dir=base_dir, timeseries_file=timeseries_file)
            # Trim the two timeseries to line up and be the same length
            data, gw_level = align_timeseries(data, gw_level)
            if data.size != gw_level.size:
                print('Warning: timeseries do not align for suite '+suite+'. Removing suite from plot. Try running fix_missing_months()')
                num_ens -= 1
                continue
            # Figure out the scenario type we're actually trying to plot
            stype = data.scenario_type[-1]
            if suite in highlight_suites:
                # Save the stype and colour at the right place in the highlight list
                i = highlight_suites.index(suite)
                highlight_stype[i] = stype
                highlight_colours[i] = colour
            # Smooth in time            
            gw_level = moving_average(gw_level, smooth)
            data = moving_average(data, smooth)
            # Now trim off the parent suite, so we only keep the bit needed to smooth over the gap
            t_start = np.where(data.scenario_type==stype)[0][0]
            gw_level = gw_level.isel(time_centered=slice(t_start,None))
            data = data.isel(time_centered=slice(t_start,None))
            gw_levels.append(gw_level)
            datas.append(data)
        if num_ens > 0:
            labels_plot += [label] + [None]*(num_ens-1)
            colours_plot += [colour]*num_ens
            lw_plot += [linewidth]*num_ens

    if highlight is not None:
        # Plot a specific trajectory again in a thicker line
        if highlight_arrows:
            arrow_x = []
            arrow_y = []
            arrow_dx = []
            arrow_dy = []
            arrow_colours = []
        if alternate_var is not None:
            raise Exception('Can only highlight trajectories when alternate_var is None')
        if None in highlight_stype or None in highlight_colours:
            raise Exception('One or more highlight suites is not in main suite list')
        gw_level = build_timeseries_trajectory(highlight_suites, 'global_mean_sat', base_dir=base_dir, timeseries_file=timeseries_file_um, offset=-baseline_temp-offset)
        data = build_timeseries_trajectory(highlight_suites, var_name, base_dir=base_dir, timeseries_file=timeseries_file)
        data, gw_level = align_timeseries(data, gw_level)
        gw_level = gw_level.interpolate_na(dim='time_centered')
        data = data.interpolate_na(dim='time_centered')
        gw_level = moving_average(gw_level, smooth)
        data = moving_average(data, smooth)
        if data.size != gw_level.size:
            print('Warning: timeseries do not align for highlight trajectory '+highlight+'. Removing from plot.')
        else:
            # Select each stage from trajectory and add to plotting list
            for i in range(num_highlight):
                gw_level_stage = gw_level.where(data.scenario_type==highlight_stype[i], drop=True)
                data_stage = data.where(data.scenario_type==highlight_stype[i], drop=True)
                gw_levels.append(gw_level_stage)
                datas.append(data_stage)
                labels_plot += [None]
                colours_plot += [highlight_colours[i]]
                lw_plot += [linewidth*3]
                if highlight_arrows:
                    arrow_loc_stage = arrow_loc[i]
                    for target_temp in arrow_loc_stage:
                        # Find closest data point to this global temperature
                        dtemp = np.abs(gw_level_stage + temp_correct - target_temp)
                        t0 = np.argwhere((dtemp == np.amin(dtemp)).data)[0][0]
                        arrow_x.append(gw_level_stage.data[t0]+temp_correct)
                        arrow_y.append(data_stage.data[t0])
                        # Find tangent to the curve over 1-year timescale
                        arrow_dx.append(gw_level_stage.data[t0+months_per_year] - gw_level_stage.data[t0-months_per_year])
                        arrow_dy.append(data_stage.data[t0+months_per_year] - data_stage.data[t0-months_per_year])
                        arrow_colours.append(highlight_colours[i])                    

    # Plot
    if new_ax:
        if labels is None:
            figsize = (6,4)
        else:
            # Need a bigger plot to make room for a legend
            figsize = (8,5)
        fig, ax = plt.subplots(figsize=figsize)
    for gw_level, data, colour, label, lw in zip(gw_levels, datas, colours_plot, labels_plot, lw_plot):
        ax.plot(gw_level+temp_correct, data, '-', color=colour, label=label, linewidth=lw)
    if highlight_arrows:
        for x, y, dx, dy, colour in zip(arrow_x, arrow_y, arrow_dx, arrow_dy, arrow_colours):
            ax.annotate("", xytext=(x,y), xy=(x+dx,y+dy), arrowprops=dict(arrowstyle='fancy', mutation_scale=15, color=colour))
    ax.grid(linestyle='dotted')
    if title is None:
        title = datas[0].long_name
    if units is None:
        units = datas[0].units
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(units, fontsize=14)
    if alternate_var is None:
        ax.set_xlabel('Global warming relative to preindustrial (K)', fontsize=14)
    else:
        ax.set_xlabel(gw_levels[0].long_name+' ('+gw_levels[0].units+')', fontsize=14)
    if labels is not None and new_ax:
        # Make legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    if new_ax:
        finished_plot(fig, fig_name=fig_name)


# Plot timeseries by global warming level for all variables in all experiments.
def plot_all_by_gw_level (base_dir='./', regions=['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross'], var_names=['massloss', 'draft', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_temp', 'shelf_salt', 'temp_btw_200_700m', 'salt_btw_200_700m', 'drake_passage_transport'], timeseries_file='timeseries.nc', timeseries_file_u='timeseries_u.nc', timeseries_file_um='timeseries_um.nc', smooth=24, fig_dir=None, pi_suite='cs495', integrate=False):

    sim_names, colours, sim_dirs = set_expt_list(separate_stages=True)
    
    # Now construct master list of variables as above - modularise this if I do it a third time!
    var_names_all = []
    for region in regions:
        for var in var_names:
            if var == 'drake_passage_transport':
                # Special case with no region
                if var not in var_names_all:
                    var_names_all.append(var)
            elif var.endswith('200_700m'):
                # Special cases where only some regions defined
                if region in ['amundsen_sea', 'bellingshausen_sea']:
                    var_names_all.append(region+'_'+var)
            else:
                # Every combination of region and variable
                var_names_all.append(region+'_'+var)

    for var in var_names_all:
        if var == 'drake_passage_transport':
            fname = timeseries_file_u
        else:
            fname = timeseries_file
        plot_by_gw_level(sim_dirs, var, pi_suite=pi_suite, base_dir=base_dir, timeseries_file=fname, timeseries_file_um=timeseries_file_um, smooth=smooth, labels=sim_names, colours=colours, linewidth=1, fig_name=None if fig_dir is None else (fig_dir+'/'+var+'_gw.png'), integrate=integrate)


# Synthesise all this into a set of 5-panel plots for 3 different variables showing timeseries by GW level.
def gw_level_panel_plots (base_dir='./', pi_suite='cs495', fig_dir=None, integrate=False):

    regions = ['ross', 'filchner_ronne', 'west_antarctica', 'east_antarctica', 'all']
    var_names = ['bwtemp', 'bwsalt', 'massloss']
    var_titles = ['Bottom temperature on continental shelf and cavities', 'Bottom salinity on continental shelf and cavities', 'Basal mass loss']
    units = [deg_string+'C', gkg_string, 'Gt/y']
    sim_names, colours, sim_dirs = set_expt_list(separate_stages=True)
    timeseries_file = 'timeseries.nc'
    smooth = 24

    for v in range(len(var_names)):
        fig = plt.figure(figsize=(9,10))
        gs = plt.GridSpec(3,2)
        gs.update(left=0.09, right=0.98, bottom=0.07, top=0.9, hspace=0.3, wspace=0.15)
        for n in range(len(regions)):
            ax = plt.subplot(gs[n//2,n%2])
            plot_by_gw_level(sim_dirs, regions[n]+'_'+var_names[v], pi_suite=pi_suite, base_dir=base_dir, timeseries_file=timeseries_file, smooth=smooth, labels=sim_names, colours=colours, linewidth=0.5, ax=ax, integrate=integrate)
            if n == len(regions)-1:
                title = 'Antarctica mean'
            else:
                title = region_names[regions[n]]
            ax.set_title(title, fontsize=14)
            if n%2 == 0:
                ax.set_ylabel(units[v], fontsize=12)
            else:
                ax.set_ylabel('')
            if n//2 == 2:
                ax.set_xlabel('Global warming relative to preindustrial (K)', fontsize=12)
            else:
                ax.set_xlabel('')
        plt.suptitle(var_titles[v], fontsize=16)
        # Make legend in the last box
        ax.legend(loc='center left', bbox_to_anchor=(1.2,0.5), fontsize=11)
        finished_plot(fig, fig_name=None if fig_dir is None else fig_dir+'/'+var_names[v]+'_gw_panels.png')


# Plot timeseries of the given variable (massloss or cavity_temp) against bottom salinity in the corresponding region for the three big cold cavities (FRIS, Ross, and Amery).
def cold_cavities_by_bwsalt (var_name, base_dir='./', fig_name=None):

    regions = ['filchner_ronne', 'ross', 'amery']
    var_x = [region+'_shelf_bwsalt' for region in regions] 
    var_y = [region+'_'+var_name for region in regions]
    sim_names, colours, sim_dirs = set_expt_list()
    timeseries_file = 'timeseries.nc'
    smooth = 24
    if var_name == 'massloss':
        units = 'Gt/y'
    elif 'temp' in var_name:
        units = deg_string+'C'

    fig = plt.figure(figsize=(10,6))
    gs = plt.GridSpec(1,3)
    gs.update(left=0.08, right=0.98, bottom=0.3, top=0.92, wspace=0.2)
    for n in range(len(regions)):
        ax = plt.subplot(gs[0,n])
        # Read timeseries from every experiment
        data_x = []
        data_y = []
        labels_plot = []
        colours_plot = []
        for expt, label, colour in zip(sim_dirs, sim_names, colours):
            if isinstance(expt, str):
                expt = [expt]
            num_ens = len(expt)
            labels_plot += [label] + [None]*(num_ens-1)
            colours_plot += [colour]*num_ens
            for suite in expt:
                if not os.path.isfile(base_dir+'/'+suite+'/'+timeseries_file):
                    data_x.append(None)
                    data_y.append(None)
                    continue
                ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file, decode_times=time_coder)
                data_x.append(moving_average(ds[var_x[n]], smooth))
                data_y.append(moving_average(ds[var_y[n]], smooth))
                ds.close()
        # Plot
        for x, y, label, colour in zip(data_x, data_y, labels_plot, colours_plot):
            if x is not None and y is not None:
                ax.plot(x, y, '-', color=colour, label=label, linewidth=1)
        ax.grid(linestyle='dotted')
        ax.set_title(region_names[regions[n]], fontsize=14)
        if n==0:
            ax.set_xlabel('Shelf bottom salinity ('+gkg_string+')', fontsize=10)
            ax.set_ylabel(var_name+' ('+units+')', fontsize=10)
    # Legend at bottom
    ax.legend(loc='lower center', bbox_to_anchor=(-0.7, -0.42), fontsize=10, ncol=3)
    finished_plot(fig, fig_name=fig_name)


# Before running this on Jasmin, do "source ~/pyenv/bin/activate" so we can use gsw
def plot_bwsalt_vs_obs (suite='cy691', schmidtko_file='/gws/nopw/j04/terrafirma/kaight/input_data/schmidtko_TS.txt', woa_files='/gws/nopw/j04/terrafirma/kaight/input_data/WOA18/woa18_decav_*00_04.nc', precomputed_file='bwsalt_1995_2014.nc', fig_name=None, base_dir='./'):

    start_year = 1995
    end_year = 2014
    eos = 'eos80'

    sim_dir = base_dir + '/' + suite + '/'
    if os.path.isfile(sim_dir+precomputed_file):
        nemo = xr.open_dataset(sim_dir+precomputed_file, decode_times=time_coder)
    else:
        # Identify NEMO output files in the suite directory within the given date range
        file_head = 'nemo_'+suite+'o_1m_'
        file_tail = '_grid-T.nc'
        nemo_files = []
        for f in os.listdir(sim_dir):
            if f.startswith(file_head) and f.endswith(file_tail):
                year = int(f[len(file_head):len(file_head)+4])
                if year >= start_year and year <= end_year:
                    nemo_files.append(sim_dir+f)
        # Read and time-average
        nemo = xr.open_mfdataset(nemo_files, concat_dim='time_counter', combine='nested', decode_times=time_coder)
        nemo = nemo.mean(dim='time_counter').squeeze()
        nemo.load()
        # Save to NetCDF for next time
        nemo.to_netcdf(sim_dir+precomputed_file)
    # Trim halo
    nemo = nemo.isel(x=slice(1,-1))

    # Read observations
    schmidtko = read_schmidtko(schmidtko_file=schmidtko_file, eos=eos)
    woa = read_woa(woa_files=woa_files, eos=eos)
    # Regrid to the NEMO grid, giving precedence to Schmidtko where both datasets exist
    schmidtko_interp = interp_latlon_cf(schmidtko, nemo, method='bilinear')
    woa_interp = interp_latlon_cf(woa, nemo, method='bilinear')
    obs = xr.where(schmidtko_interp.isnull(), woa_interp, schmidtko_interp)
    # Apply NEMO land mask to both
    nemo_plot = nemo['sob'].where(nemo['sob']!=0)
    obs_plot = obs['salt'].where(nemo_plot.notnull()*obs['salt'].notnull())
    obs_plot = obs_plot.where(nemo['sob']!=0)

    # Make the plot
    fig = plt.figure(figsize=(8,3))
    gs = plt.GridSpec(1,3)
    gs.update(left=0.1, right=0.9, bottom=0.05, top=0.8, wspace=0.1)
    data_plot = [nemo_plot, obs_plot, nemo_plot-obs_plot]
    titles = ['UKESM', 'Observations', 'Model bias']
    vmin = [34, 34, -0.5]
    vmax = [34.85, 34.85, 0.5]
    ctype = ['RdBu_r', 'RdBu_r', 'plusminus']
    for n in range(3):
        ax = plt.subplot(gs[0,n])
        ax.axis('equal')
        img = circumpolar_plot(data_plot[n], nemo, ax=ax, masked=True, make_cbar=False, title=titles[n], titlesize=14, vmin=vmin[n], vmax=vmax[n], ctype=ctype[n], lat_max=-63)
        if n != 1:
            cax = cax = fig.add_axes([0.01+0.45*n, 0.1, 0.02, 0.6])
            plt.colorbar(img, cax=cax, extend='both')
    plt.suptitle('Bottom salinity (psu), historical ('+str(start_year)+'-'+str(end_year)+')', fontsize=18)
    finished_plot(fig, fig_name=fig_name)


# Time-average each stabilisation scenario (all years and all ensemble members) for the given file type (grid-T, isf-T, grid-U).
def calc_stabilisation_means (base_dir='./', file_type='grid-T', out_dir='time_averaged/'):

    scenarios = ['piControl', '1.5K', '2K', '2.5K', '3K', '4K', '5K', '6K']
    
    for scenario in scenarios:
        print('Processing '+scenario)
        if scenario == 'piControl':
            scenario_full = scenario
        else:
            scenario_full = scenario + '_stabilise'
        out_file = base_dir+'/'+out_dir+'/'+scenario+'_'+file_type+'.nc'
        log_file = base_dir+'/'+out_dir+'/'+scenario+'_'+file_type+'.log'
        update_file = os.path.isfile(out_file)
        if update_file:
            # Read the log file to find out which files have already been processed
            old_files = np.loadtxt(log_file, dtype=str)
            num_old_files = old_files.size
        # Open the log file (may or may not already exist) to append to it
        log = open(log_file, 'a')
        nemo_files = []
        for suite in suites_by_scenario[scenario_full]:
            sim_dir = base_dir+'/'+suite+'/'
            file_head = 'nemo_'+suite+'o_1m_'
            file_tail = '_'+file_type+'.nc'
            for f in os.listdir(sim_dir):
                if f.startswith(file_head) and f.endswith(file_tail):
                    if update_file and f in old_files:
                        # Skip it; already processed
                        continue
                    nemo_files.append(sim_dir+f)
                    log.write(f+'\n')        
        log.close()
        ds_accum = None
        num_files = len(nemo_files)
        if num_files == 0:
            continue
        for n in tqdm(range(num_files), desc=' files'):
            ds = xr.open_dataset(nemo_files[n], decode_times=time_coder).squeeze()
            if ds_accum is None:
                ds_accum = ds
            else:
                ds_accum += ds
            ds.close()
        ds_mean = ds_accum/num_files
        if update_file:
            # Now combine with existing mean as weighted average
            num_files_total = num_files + num_old_files
            ds_mean_old = xr.open_dataset(out_file, decode_times=time_coder)
            ds_mean = ds_mean*num_files/num_files_total + ds_mean_old*num_old_files/num_files_total
        ds_mean.to_netcdf(out_file+'_tmp', mode='w')
        ds_mean.close()
        if update_file:
            ds_mean_old.close()
        os.rename(out_file+'_tmp',out_file)
        

# Plot maps of the time-mean of the given variable in each stabilisation scenario
def plot_stabilisation_maps (var_name, fig_name=None):

    scenarios = ['piControl', '1.5K', '2K', '2.5K', '3K', '4K', '5K', '6K']
    in_dir = 'time_averaged/' 
    num_scenarios = len(scenarios)
    domain_cfg = '/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'
    if var_name == 'barotropic_streamfunction':
        title = 'Barotropic streamfunction (Sv)'
        file_type = 'grid-U'
        file_type_2 = 'grid-V'
        contour = [-15, 0]
        vmin = -60
        vmax = 60
        lat_max = None
        ctype = 'plusminus'
    elif var_name == 'tob':
        title = 'Bottom temperature ('+deg_string+'C)'
        file_type = 'grid-T'
        contour = None
        vmin = -2
        vmax = 4.5
        lat_max = -63
        ctype = 'RdBu_r'
    elif var_name == 'sob':
        title = 'Bottom salinity ('+gkg_string+')'
        file_type = 'grid-T'
        contour = None
        vmin = 33.5
        vmax = 34.8
        lat_max = -63
        ctype = 'RdBu_r'
    elif var_name == 'temp500m':
        title = 'Temperature at 500m ('+deg_string+'C)'
        file_type = 'grid-T'
        contour = None
        vmin = -2
        vmax = 7
        lat_max = None
        ctype = 'RdBu_r'
    else:
        raise Exception('invalid var_name')
    
    fig = plt.figure(figsize=(4,8))
    gs = plt.GridSpec(num_scenarios//2, 2)
    gs.update(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.2, wspace=0.2)
    for n in range(num_scenarios):
        ds = xr.open_dataset(in_dir+scenarios[n]+'_'+file_type+'.nc', decode_times=time_coder).squeeze()
        if var_name=='barotropic_streamfunction':
            if n==0:
                # Grab e2u from domain_cfg
                ds_domcfg = xr.open_dataset(domain_cfg, decode_times=time_coder).squeeze()
                ds_domcfg = ds_domcfg.isel(y=slice(0, ds.sizes['y']))
            ds_v = xr.open_dataset(in_dir+scenarios[n]+'_'+file_type_2+'.nc', decode_times=time_coder).squeeze()
        if var_name == 'barotropic_streamfunction':
            data_plot = barotropic_streamfunction(ds, ds_v, ds_domcfg, periodic=True, halo=True)
        elif var_name == 'temp500m':
            data_3d = ds['thetao'].where(ds['thetao']!=0)
            data_plot = data_3d.interp(deptht=500)
        else:
            data_plot = ds[var_name]
        ax = plt.subplot(gs[n//2, n%2])
        ax.axis('equal')
        img = circumpolar_plot(data_plot, ds, ax=ax, make_cbar=False, title=scenarios[n], ctype=ctype, titlesize=14, vmin=vmin, vmax=vmax, contour=contour, lat_max=lat_max)
        if n == num_scenarios-1:
            cax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
            plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
        ds.close()
    plt.suptitle(title, fontsize=18)
    finished_plot(fig, fig_name=fig_name)


# Plot T and S profiles averaged over the given region, for each stabilisation scenario
def plot_stabilisation_profiles (region='amundsen_sea', fig_name=None):

    scenarios = ['piControl', '1.5K', '2K', '2.5K', '3K', '4K', '5K', '6K']
    colours = ['Black', 'DarkMagenta', 'Blue', 'DarkCyan', 'DarkGreen', 'GoldenRod', 'Chocolate', 'Crimson']
    in_dir = 'time_averaged/'
    file_tail = '_grid-T.nc'
    num_scenarios = len(scenarios)
    var_names = ['thetao', 'so']
    num_var = len(var_names)
    var_titles = ['Temperature', 'Salinity']
    var_units = [deg_string+'C', 'psu']

    fig = plt.figure(figsize=(8,5))
    gs = plt.GridSpec(1,num_var)
    gs.update(left=0.1, right=0.98, bottom=0.2, top=0.85, wspace=0.2)
    ax_all = [plt.subplot(gs[0,v]) for v in range(num_var)]
    for n in range(num_scenarios):
        ds = xr.open_dataset(in_dir+scenarios[n]+file_tail, decode_times=time_coder).squeeze()
        if n==0:
            mask, ds, region_name = region_mask(region, ds, option='shelf', return_name=True)
            mask_3d = xr.where(ds[var_names[0]]==0, 0, mask)
            dA = ds['area']*mask_3d
        for v in range(num_var):
            ax = ax_all[v]
            # Area-average the given variable to get a depth profile
            data = (ds[var_names[v]]*dA).sum(dim=['x','y'])/dA.sum(dim=['x','y'])            
            ax.plot(data, ds['deptht'], '-', color=colours[n], label=scenarios[n])
            # Find deepest unmasked depth
            zdeep = ds['deptht'].where(data.notnull()).max()
            ax.set_ylim([zdeep, 0])
            ax.set_title(var_titles[v], fontsize=14)
            ax.set_xlabel(var_units[v], fontsize=12)
            if v==0:
                ax.set_ylabel('Depth (m)', fontsize=12)
            ax.grid(linestyle='dotted')
    ax.legend(loc='lower center', bbox_to_anchor=(-0.1,-0.3), fontsize=10, ncol=num_scenarios)
    plt.suptitle(region_name, fontsize=16)
    finished_plot(fig, fig_name=fig_name)


# Assemble a list of lists of all the possible suite trajectories. eg, one entry is ['cx209', 'cy837', 'da697']: each simulation branches from the previous one.
def all_suite_trajectories (static_ice=False):

    suite_sequences = []
    
    # Recursive inner function to build the list
    def complete_sequence (sequence):
        # Add the (possibly partial) sequence to the list, for example to consider perpetual ramp-up case
        suite_sequences.append(sequence)
        # Find the last suite in the sequence
        suite = sequence[-1]
        # Find all new suites which branched from this original suite
        new_suites = []
        for scenario in suites_by_scenario:
            for s in suites_by_scenario[scenario]:
                if s not in suites_branched:
                    # Some suites (eg preindustrial) not considered in branching
                    continue
                if suites_branched[s] == suite:
                    new_suites.append(s)
        if len(new_suites)==0:
            # Exit condition: nothing branches from this suite; sequence is complete
            pass
        else:
            # Loop over each suite that branches, and go another level down in the recursion
            for s in new_suites:
                complete_sequence(sequence+[s])

    # Start from each ramp-up ensemble member and use it as a seed
    if static_ice:
        name0 = 'ramp_up_static_ice'
    else:
        name0 = 'ramp_up'
    for suite in suites_by_scenario[name0]:
        complete_sequence([suite])

    return suite_sequences


# Helper function to assemble a timeseries for the given trajectory and variable (precomputed).
def build_timeseries_trajectory (suite_list, var_name, base_dir='./', timeseries_file='timeseries.nc', offset=0):

    xr.set_options(keep_attrs=True)

    for suite in suite_list:
        # Figure out whether it's a ramp-up (1), stabilisation (0), or ramp-down (-1) simulation
        stype = None
        for scenario in suites_by_scenario:
            if suite in suites_by_scenario[scenario]:
                if 'ramp_up' in scenario:
                    stype = 1
                elif 'restabilise' in scenario:
                    stype = -2
                elif 'stabilise' in scenario:
                    stype = 0
                elif 'ramp_down' in scenario:
                    stype = -1
                elif 'piControl' in scenario:
                    stype = 2
                else:
                    raise Exception('invalid scenario type')
                break
        if stype is None:
            raise Exception('Simulation type not found for '+suite)
        ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file, decode_times=time_coder)
        data = ds[var_name] + offset
        data = data.assign_coords(scenario_type=('time_centered', np.ones(data.size)*stype))
        if 'time_counter' in data.coords:
            data = data.drop_vars(['time_counter'])
        if suite != suite_list[0]:
            # Find the starting date of the current suite
            start_date = data.time_centered[0]
            if data_prev.time_centered[-1].data > start_date.data:
                # Trim the previous timeseries to just before that date
                time_branch = np.argwhere(data_prev.time_centered.data == start_date.data)[0][0]
                data_prev = data_prev.isel(time_centered=slice(0,time_branch))
            # Concatenate with the previous timeseries
            data = xr.concat([data_prev, data], dim='time_centered')
        # Prepare for next iteration of loop
        data_prev = data
        ds.close()
    return data    


# Assemble a list of all possible trajectories of the given timeseries variable (precomputed).
# Can add an offset if needed (eg the negative of the preindustrial baseline temperature).
def all_timeseries_trajectories (var_name, base_dir='./', timeseries_file='timeseries.nc', static_ice=False, offset=0):

    suite_sequences = all_suite_trajectories(static_ice=static_ice)
    timeseries = []
    suite_strings = []
    # Loop over each suite trajectory and build the timeseries
    for suite_list in suite_sequences:
        data = build_timeseries_trajectory(suite_list, var_name, base_dir=base_dir, timeseries_file=timeseries_file, offset=offset)
        suite_strings.append('-'.join(suite_list))
        timeseries.append(data)
    return timeseries, suite_strings


# Helper function to check if the given suite tips.
# Input:
# suite: suite name or trajectory string, eg 'cx209' or 'cx209-cy838'
# region: 'ross' or 'filchner_ronne'
# XOR
# cavity_temp: DataArray of temperature averaged over the cavity
# Optional:
# smoothed: whether cavity_temp is already smoothed
# return_date: whether to return a cftime.datetime object showing when it tips
# return_t: whether to return the time index (integer) showing when cavity_temp tips (only if cavity_temp is set)
def check_tip (suite=None, region=None, cavity_temp=None, smoothed=False, return_date=False, return_t=False, base_dir='./'):

    smooth = 5*months_per_year

    # Error checking input
    if suite is not None and region is None:
        raise Exception('Must set region if suite is defined')
    if suite is not None and cavity_temp is not None:
        raise Exception('Cannot set both suite and cavity_temp')
    if return_t and cavity_temp is None:
        raise Exception('Can only set return_t if cavity_temp is set')

    if suite is not None:
        # Get cavity temp array
        cavity_temp = build_timeseries_trajectory(suite.split('-'), region+'_cavity_temp', base_dir=base_dir)
        smoothed = False
    if not smoothed:
        cavity_temp = moving_average(cavity_temp, smooth)

    tipped = cavity_temp.max() > tipping_threshold
    if tipped:
        t_tip = np.argwhere(cavity_temp.data > tipping_threshold)[0][0]
        date_tip = cavity_temp.time_centered[t_tip]
    else:
        t_tip = None
        date_tip = None
    if return_date and return_t:
        return tipped, date_tip, t_tip
    elif return_date:
        return tipped, date_tip
    elif return_t:
        return tipped, t_tip
    else:
        return tipped


# Helper function to check if the given suite recovers.
# Input as in tips().
def check_recover (suite=None, region=None, cavity_temp=None, smoothed=False, return_date=False, return_t=False, base_dir='./'):

    smooth = 5*months_per_year
    
    if suite is not None and region is None:
        raise Exception('Must set region if suite is defined')
    if suite is not None and cavity_temp is not None:
        raise Exception('Cannot set both suite and cavity_temp')
    if return_t and cavity_temp is None:
        raise Exception('Can only set return_t if cavity_temp is set')

    if suite is not None:
        cavity_temp = build_timeseries_trajectory(suite.split('-'), region+'_cavity_temp', base_dir=base_dir)
        smoothed = False
    if not smoothed:
        cavity_temp = moving_average(cavity_temp, smooth)

    # Check if it even tips
    tipped, t_tip = check_tip(cavity_temp=cavity_temp, smoothed=True, return_t=True)
    recovered = False
    if tipped:        
        # Check every time index after tipping
        for t in range(t_tip, cavity_temp.sizes['time_centered']):
            if cavity_temp.isel(time_centered=slice(t,None)).max() < tipping_threshold:
                recovered = True
                t_recover = t
                date_recover = cavity_temp.time_centered[t_recover]
                break
    if not tipped or not recovered:
        t_recover = None
        date_recover = None
    if return_date and return_t:
        return recovered, date_recover, t_recover
    elif return_date:
        return recovered, date_recover
    elif return_t:
        return recovered, t_recover
    else:
        return recovered    


# Analyse the cavity temperature beneath Ross and FRIS to see which scenarios tip and/or recover, under which global warming levels. Also plot this.
def tipping_stats (base_dir='./'):

    regions = ['ross', 'filchner_ronne']
    title_prefix = ['a) ', 'b) ']
    bias_print_x = [4.5, 2.5]
    bias_print_y = 1.5
    pi_suite = 'cs495'
    smooth = 5*months_per_year
    timeseries_file = 'timeseries.nc'
    timeseries_file_um = 'timeseries_um.nc'
    p0 = 0.05

    # Assemble all possible trajectories of global mean temperature anomalies relative to preindustrial
    baseline_temp = pi_baseline_temp(pi_suite=pi_suite, base_dir=base_dir)
    warming_ts, suite_strings = all_timeseries_trajectories('global_mean_sat', base_dir=base_dir, timeseries_file=timeseries_file_um, static_ice=False, offset=-1*baseline_temp)
    num_trajectories = len(warming_ts)
    # Smooth each
    for n in range(num_trajectories):
        warming_ts[n] = moving_average(warming_ts[n], smooth)

    # Loop over regions
    all_temp_tip = []
    all_temp_recover = []
    all_tips = []
    threshold_bounds = []
    all_recovery_floor = []
    for r in range(len(regions)):
        # Assemble all possible trajectories of cavity mean temperature
        cavity_temp_ts = all_timeseries_trajectories(regions[r]+'_cavity_temp', base_dir=base_dir, static_ice=False)[0]
        # Now loop over them and find the ones that have tipped and/or recovered
        suites_tipped = []
        warming_at_tip = []
        suites_recovered = []        
        warming_at_recovery = []
        if regions[r] == 'ross':
            warming_at_recovery_fris_tip = []
            warming_at_recovery_fris_notip = []
        tips = []
        recovery_floor = None
        for n in range(num_trajectories):
            # Smooth and trim/align with warming timeseries
            cavity_temp = moving_average(cavity_temp_ts[n], smooth)
            cavity_temp, warming = align_timeseries(cavity_temp, warming_ts[n])
            traj_tips, t_tip = check_tip(cavity_temp=cavity_temp, smoothed=True, return_t=True, base_dir=base_dir)
            if traj_tips:
                tip_warming = warming.isel(time_centered=t_tip)
                if np.isnan(tip_warming):
                    continue
                suites_tipped.append(suite_strings[n])
                warming_at_tip.append(tip_warming)
                recovers, t_recovers = check_recover(cavity_temp=cavity_temp, smoothed=True, return_t=True, base_dir=base_dir)
                if recovers:
                    recover_warming = warming.isel(time_centered=t_recovers)
                    if np.isnan(recover_warming):
                        continue
                    #print(suite_strings[n]+' recovers at '+str(recover_warming.item()))
                    suites_recovered.append(suite_strings[n])
                    warming_at_recovery.append(recover_warming)
                    if regions[r] == 'ross':
                        # Check if FRIS tipped
                        fris_tip = check_tip(suite=suite_strings[n], region='filchner_ronne')
                        if fris_tip:
                            warming_at_recovery_fris_tip.append(recover_warming)
                        else:
                            warming_at_recovery_fris_notip.append(recover_warming)
                else:
                    # Find the final temperature, and keep track of the coolest temperature at which a tipped trajectory still hasn't recovered.
                    final_temp = warming.isel(time_centered=-1).item()
                    if recovery_floor is None:
                        recovery_floor = final_temp
                    else:
                        recovery_floor = min(recovery_floor, final_temp)
            tips.append(traj_tips)
        tips = np.array(tips)
        # Now throw out duplicates, eg if tipping happened before a suite branched into multiple trajectories, should only count it once for the statistics.
        # Do this by only considering unique values of warming_at_tip and warming_at_recovery.
        # This assumes it's impossible for two distinct suites to tip at exactly the same global warming level, to machine precision. I think I'm happy with this!
        warming_at_tip = np.unique(warming_at_tip)
        warming_at_recovery = np.unique(warming_at_recovery)
        if regions[r] == 'ross':
            warming_at_recovery_fris_tip = np.unique(warming_at_recovery_fris_tip)
            warming_at_recovery_fris_notip = np.unique(warming_at_recovery_fris_notip)
        # Find maximum warming in each trajectory
        max_warming = np.array([warming_ts[n].max() for n in range(num_trajectories)])
        # Check if recovery_floor is actually outside the range of warming_at_recovery
        if recovery_floor >= np.min(warming_at_recovery):
            recovery_floor = None
        
        # Print some statistics about which ones tipped and recovered
        print('\n'+regions[r]+':')
        print(str(len(suites_tipped))+' trajectories tip, '+str(len(warming_at_tip))+' unique')
        print('Global warming at time of tipping has mean '+str(np.mean(warming_at_tip)+temp_correction)+'K, standard deviation '+str(np.std(warming_at_tip))+'K')
        if len(suites_recovered) == 0:
            print('No tipped trajectories recover')
        else:
            print(str(len(suites_recovered))+' tipped trajectories recover ('+str(len(suites_recovered)/len(suites_tipped)*100)+'%), '+str(len(warming_at_recovery))+' unique')
            print('Global warming at time of recovery has mean '+str(np.mean(warming_at_recovery)+temp_correction)+'K, standard deviation '+str(np.std(warming_at_recovery))+'K')
            if regions[r] == 'ross':
                print('If FRIS also tips, Ross recovery happens at mean '+str(np.mean(warming_at_recovery_fris_tip)+temp_correction)+'K, standard deviation '+str(np.std(warming_at_recovery_fris_tip))+'K, n='+str(len(warming_at_recovery_fris_tip)))
                print('If FRIS does not tip, Ross recovery happens at mean '+str(np.mean(warming_at_recovery_fris_notip)+temp_correction)+'K, standard deviation '+str(np.std(warming_at_recovery_fris_notip))+'K, n='+str(len(warming_at_recovery_fris_notip)))
                if len(warming_at_recovery_fris_tip) == 1:
                    p_val = ttest_1samp(warming_at_recovery_fris_notip, warming_at_recovery_fris_tip)[1]
                else:
                    p_val = ttest_ind(warming_at_recovery_fris_tip, warming_at_recovery_fris_notip, equal_var=False)[1]
                distinct = p_val < p0
                if distinct:
                    print('Significant difference (p='+str(p_val)+')')
                else:
                    print('No significant difference (p='+str(p_val)+')')
        if recovery_floor is not None:
            print('Trajectories as cool as '+str(recovery_floor+temp_correction)+'K still have not recovered')
        else:
            print('All trajectories which cool below '+str(np.amin(warming_at_recovery)+temp_correction)+'K recover')
        # Save results for plotting
        all_temp_tip.append(warming_at_tip)
        all_temp_recover.append(warming_at_recovery)
        all_tips.append(tips)
        all_recovery_floor.append(recovery_floor)

        # Find bounds on threshold
        # Find coolest instance of tipping
        first_tip = np.amin(max_warming[tips])
        # Step down to one simulation cooler than that: lower bound on threshold
        threshold_min = np.amax(max_warming[max_warming < first_tip]) + temp_correction
        # Find warmest instance of non-tipping
        last_notip = np.amax(max_warming[~tips])
        # Step up to one simulation warmer than that: upper bound on threshold
        threshold_max = np.amin(max_warming[max_warming > last_notip]) + temp_correction
        threshold_bounds.append([threshold_min, threshold_max])
        print('Threshold lies between '+str(threshold_min)+' and '+str(threshold_max)+', or '+str(0.5*(threshold_min+threshold_max))+' +/- '+str(0.5*(threshold_max-threshold_min)))
        threshold_min_uncertainty = threshold_min - temp_correction + temp_correction_lower
        threshold_max_uncertainty = threshold_max - temp_correction + temp_correction_upper
        print('Accounting for uncertainty in temperature correction, threshold lies between '+str(threshold_min_uncertainty)+' and '+str(threshold_max_uncertainty)+', or '+str(0.5*(threshold_min_uncertainty+threshold_max_uncertainty))+' +/- '+str(0.5*(threshold_max_uncertainty-threshold_min_uncertainty)))

    # Plot
    fig = plt.figure(figsize=(6,5))
    gs = plt.GridSpec(2,1)
    gs.update(left=0.25, right=0.95, bottom=0.1, top=0.92, hspace=0.4)
    for r in range(len(regions)):
        ax = plt.subplot(gs[r,0])
        # Violin plots: warming level at time of tipping (red), recovery (blue)
        violin_data = [np.array(all_temp_tip[r])+temp_correction, np.array(all_temp_recover[r])+temp_correction]
        y_pos = [3, 2]
        colours = ['Crimson', 'DodgerBlue']
        violins = ax.violinplot(violin_data, y_pos, vert=False, showextrema=False, showmeans=True)
        # Set colours of violin bodies and lines
        for pc, colour in zip(violins['bodies'], colours):
            pc.set_facecolor(colour)
        for bar in ['cmeans']:
            violins[bar].set_colors('black')
        # Plot individual data points
        ax.plot(np.array(all_temp_tip[r])+temp_correction, 3*np.ones(len(all_temp_tip[r])), 'o', markersize=3, color='Crimson')
        ax.plot(np.array(all_temp_recover[r])+temp_correction, 2*np.ones(len(all_temp_recover[r])), 'o', markersize=3, color='DodgerBlue')
        #if regions[r] == 'ross':
            #ax.plot(np.array(warming_at_recovery_fris_tip)+temp_correction, 2*np.ones(len(warming_at_recovery_fris_tip)), 'o', markersize=3, color='DarkOrchid')
        '''if all_recovery_floor[r] is not None:
            # Plot dotted blue line and open marker showing that recovery violin plot will extend at least this far
            ax.plot([all_recovery_floor[r]+temp_correction, np.amin(all_temp_recover[r])+temp_correction], [2, 2], color='DodgerBlue', linestyle='dotted', linewidth=1)
            ax.plot(all_recovery_floor[r]+temp_correction, 2, 'o', markersize=4, markeredgecolor='DodgerBlue', color='white')'''
        # Bottom row: peak warming in each trajectory, plotted in red (tips) or grey (doesn't tip)
        # Start with the grey, to make sure the red is visible where they overlap
        ax.plot(max_warming[~all_tips[r]]+temp_correction, np.ones(np.count_nonzero(~all_tips[r])), 'o', markersize=4, color='DarkGrey')
        ax.plot(max_warming[all_tips[r]]+temp_correction, np.ones(np.count_nonzero(all_tips[r])), 'o', markersize=4, color='Crimson')
        # Plot bounds on threshold: vertical dashed lines with labels
        ax.plot(threshold_bounds[r][0]*np.ones(2), [0, 0.9], color='black', linestyle='dashed', linewidth=1)
        plt.text(threshold_bounds[r][0]-0.05, 0.5, 'never tips', ha='right', va='center', fontsize=9)
        plt.arrow(threshold_bounds[r][0]-0.05, 0.2, -0.3, 0, head_width=0.1, head_length=0.08)
        ax.plot(threshold_bounds[r][1]*np.ones(2), [0, 0.9], color='black', linestyle='dashed', linewidth=1)
        plt.text(threshold_bounds[r][1]+0.05, 0.5, 'always tips', ha='left', va='center', fontsize=9)
        plt.arrow(threshold_bounds[r][1]+0.05, 0.2, 0.3, 0, head_width=0.1, head_length=0.08)
        ax.set_title(title_prefix[r]+region_names[regions[r]], fontsize=14)
        ax.set_xlim([1.5, 7])
        ax.set_ylim([0, 4])
        ax.set_yticks(np.arange(1,4))
        ax.set_yticklabels(['peak warming', 'at time of recovery', 'at time of tipping'])
        ax.grid(linestyle='dotted')       
    ax.set_xlabel('Effective global warming ('+deg_string+'C)', fontsize=10)
    # Manual legend
    colours = ['Crimson', 'DarkGrey', 'DodgerBlue'] #, 'DarkOrchid']
    labels = ['tips', 'does not tip', 'recovers'] #, 'Ross recovers\n(tipped FRIS)']
    handles = []
    for m in range(len(colours)):
        handles.append(Line2D([0], [0], marker='o', markersize=4, color=colours[m], label=labels[m], linestyle=''))
    '''if any([x is not None for x in all_recovery_floor]):
        handles.append(Line2D([0], [0], marker='o', markersize=4, color='white', markeredgecolor='DodgerBlue', label='not yet recovered', linestyle=''))'''
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(-0.35, 1.2), fontsize=9)
    finished_plot(fig, fig_name='figures/tipping_stats.png', dpi=300)
    

# Plot: (1) bottom temperature on continental shelf and in cavities, and (2) ice shelf basal mass loss as a function of global warming level, for 2 different regions, showing ramp-up, stabilise, and ramp-down in different colours
# Set static_ice=True to make supplementary version comparing static ice cases
def plot_bwtemp_massloss_by_gw_panels (base_dir='./', static_ice=False):

    pi_suite = 'cs495'
    regions = ['ross', 'filchner_ronne']
    num_regions = len(regions)
    highlights = ['cx209-cz376-da892', 'cx209-cz378-de943']
    arrow_loc = [[[2.04, 4.04, 5.14], [5.54], [5.29, 4.04]], [[2.24, 5.24, 7.24], [], [7.14, 5.2, 3.84]], [[4.04, 4.94], [5.74], [5.64, 4.54]], [[4.74, 6.64, 7.14], [7.72], [7.54, 5.44, 4.04]]]
    var_names = ['cavity_temp', 'massloss']
    var_titles = ['a) Ocean temperature in ice shelf cavities', 'b) Melting beneath ice shelves']
    var_units = [deg_string+'C', 'Gt/y']
    num_var = len(var_names)
    timeseries_file = 'timeseries.nc'
    smooth = [10*months_per_year, 20*months_per_year]
    if static_ice:
        sim_names = ['Ramp up (evolving ice)', 'Ramp up (static ice)']
        colours = ['Crimson', 'DarkMagenta']
        sim_dirs = [[suites_by_scenario['ramp_up'][0]], [suites_by_scenario['ramp_up_static_ice'][0]]]
    else:
        sim_names, colours, sim_dirs = minimal_expt_list(one_ens=True)
    #sample_file = base_dir+'/time_averaged/piControl_grid-T.nc'  # Just to build region masks
    #ds = xr.open_dataset(sample_file).squeeze()

    fig = plt.figure(figsize=(9,7))
    gs = plt.GridSpec(2,2)
    gs.update(left=0.08, right=0.98, bottom=0.1, top=0.92, hspace=0.47, wspace=0.15)
    for v in range(num_var):
        for n in range(num_regions):
            ax = plt.subplot(gs[v,n])
            plot_by_gw_level(sim_dirs, regions[n]+'_'+var_names[v], pi_suite=pi_suite, base_dir=base_dir, timeseries_file=timeseries_file, smooth=smooth[v], labels=sim_names, colours=colours, linewidth=(1 if static_ice else 0.5), ax=ax, temp_correct=temp_correction, highlight=(highlights[n] if not static_ice else None), highlight_arrows=(not static_ice), arrow_loc=arrow_loc[v*num_regions+n])
            ax.set_title(region_names[regions[n]], fontsize=14)
            if n == 0:
                ax.set_ylabel(var_units[v], fontsize=12)
            else:
                ax.set_ylabel('')
            if n==0 and v==0:
                ax.set_xlabel('Effective global warming ('+deg_string+'C)', fontsize=12)
            else:
                ax.set_xlabel('')
            ax.set_xlim([temp_correction,temp_correction+8])
            if v == 0:
                ax.axhline(-1.9, color='black', linestyle='dashed', linewidth=0.75)
            '''if v==0:
                # Inset panel in top left showing region
                mask = region_mask(regions[n], ds, option='all')[0]
                ax2 = inset_axes(ax, "25%", "40%", loc='upper left')
                ax2.axis('equal')                
                circumpolar_plot(mask, ds, ax=ax2, make_cbar=False, ctype='IndianRed', lat_max=-66, shade_land=True)
                ax2.axis('on')
                ax2.set_xticks([])
                ax2.set_yticks([])'''
        plt.text(0.5, 0.99-0.485*v, var_titles[v], fontsize=16, ha='center', va='top', transform=fig.transFigure)
    # Manual legend
    handles = []
    for m in range(len(colours)):
        handles.append(Line2D([0], [0], color=colours[m], label=sim_names[m], linestyle='-', linewidth=1.5))
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(-0.6,-0.2), fontsize=11, ncol=len(sim_names))
    fig_name = 'figures/temp_massloss_by_gw_panels'
    if static_ice:
        fig_name += '_static_ice'
    fig_name += '.png'
    finished_plot(fig, fig_name=fig_name, dpi=300)


# Calculate UKESM's bias in bottom salinity on the continental shelf of Ross and FRIS (both regions together). To do this, find the range of observed global warming between 2000-present from HadCRUT, and identify the years of each ramp-up ensemble member corresponding to this range. Then, average bottom salinity over those years and ensemble members, compare to observational climatologies interpolated to NEMO grid, and calculate the area-averaged bias.
# Before running this on Jasmin, do "source ~/pyenv/bin/activate" so we can use gsw
def calc_salinity_bias (base_dir='./', eos='eos80', plot=False, out_file='bwsalt_bias.nc', region='both'):

    regions = ['ross', 'filchner_ronne']  # Both together
    labels_lon = [-166, -45, -158, 162, -30, -70]
    labels_lat = [-72, -71, -79, -75, -78, -76.5]
    pi_suite = 'cs495'  # Preindustrial, static cavities
    hadcrut_file = base_dir+'/'+'HadCRUT.5.0.2.0.analysis.summary_series.global.annual.nc'
    obs_start = 2000  # Most of Shenjie's observations are from post-2005, so start date of 2000 allows for a small weighting of earlier data too. At time of access (28/08/2025) this presumably includes half of 2025.
    obs_pi = [1850, 1900]  # Suggested "pre-industrial" baseline for HadCRUT
    timeseries_file_um = 'timeseries_um.nc'
    obs_file = '/gws/nopw/j04/terrafirma/kaight/input_data/shenjie_climatology_bottom_TS.nc'  # Zhou 2025

    # Read observed global warming
    ds = xr.open_dataset(hadcrut_file)
    # Reference to PI
    obs_baseline = ds['tas_mean'].where((ds['time'].dt.year >= obs_pi[0])*(ds['time'].dt.year <= obs_pi[1])).mean()
    obs_gw = ds['tas_mean'] - obs_baseline
    # Select section of interest
    obs_gw = obs_gw.where(ds['time'].dt.year >= obs_start)
    # Save range of data
    obs_gw_min = obs_gw.min().item()
    obs_gw_max = obs_gw.max().item()
    print('Range of observed warming from '+str(obs_start)+'-present: '+str(obs_gw_min)+' to '+str(obs_gw_max)+' degC')

    # Loop over ramp-up suites (no static ice)
    ramp_up_bwsalt = None
    num_years = 0
    for suite in suites_by_scenario['ramp_up']:
        # Get timeseries of global warming relative to PI
        warming = global_warming(suite, pi_suite=pi_suite, base_dir=base_dir)
        # Take annual means
        warming = warming.groupby('time_centered.year').mean('time_centered')
        # Find last year when warming is below obs_gw_min; start counting the next year
        t_start = np.argwhere(warming.data < obs_gw_min)[-1][0]
        year_start = warming.year[t_start].item() + 1
        # Find first year when warming is above obs_gw_max; finish counting the previous year
        t_end = np.argwhere(warming.data > obs_gw_max)[0][0]
        year_end = warming.year[t_end].item() - 1
        print(suite+' matches historical warming range from '+str(year_start)+'-'+str(year_end))
        # Read all the grid-T ocean files from these years
        for year0 in range(year_start, year_end+1):
            num_years += 1
            for month0 in range(1, months_per_year+1):
                year1, month1 = add_months(year0, month0, 1)
                file_path = base_dir+'/'+suite+'/nemo_'+suite+'o_1m_'+str(year0)+str(month0).zfill(2)+'01-'+str(year1)+str(month1).zfill(2)+'01_grid-T.nc'
                ds = xr.open_dataset(file_path, decode_times=time_coder)
                bwsalt = ds['sob']
                if eos == 'teos10':
                    # Convert to absolute salinity
                    bwsalt = bwsalt_abs(ds)
                elif eos == 'eos80':
                    bwsalt = ds['sob']
                else:
                    raise Exception('Invalid EOS '+eos)
                if ramp_up_bwsalt is None:
                    # Initialise
                    ramp_up_bwsalt = bwsalt
                else:
                    # Accumulate
                    ramp_up_bwsalt += bwsalt
    # Convert from integral to average (over time and ensemble members)
    ramp_up_bwsalt /= (num_years*months_per_year)

    # Now read observations of bottom salinity and regrid to NEMO grid
    print('Reading Zhou 2025 data')
    obs = read_zhou_bottom_climatology(in_file=obs_file, eos=eos)
    obs_interp = interp_latlon_cf(obs, ds, method='bilinear')
    obs_bwsalt = obs_interp['salt']

    # Prepare masks of each region
    if region == 'both':
        masks = [region_mask(r, ds, option='shelf')[0] for r in regions]
        # Make combined mask
        mask = masks[0] + masks[1]
    else:
        print('Using region '+region)
        mask = region_mask(region, ds, option='shelf')[0]
    # Prepare coordinates for plotting masks
    x, y = polar_stereo(ds['nav_lon'], ds['nav_lat'])

    if plot:
        # Make a quick plot
        fig = plt.figure(figsize=(8,3))
        gs = plt.GridSpec(1,3)
        gs.update(left=0.1, right=0.9, bottom=0.05, top=0.8, wspace=0.1)
        ukesm_plot = ramp_up_bwsalt.where(ramp_up_bwsalt!=0).squeeze()
        # Mask cavities in UKESM to match observations
        ice_mask = build_ice_mask(ds)[0]
        ukesm_plot = ukesm_plot.where(~ice_mask)
        obs_plot = obs_bwsalt.where(ukesm_plot.notnull()*obs_bwsalt.notnull())
        obs_plot = obs_plot.where(ramp_up_bwsalt!=0).squeeze()
        data_plot = [ukesm_plot, obs_plot, ukesm_plot-obs_plot]
        titles = ['a) UKESM', 'b) Observations', 'c) Model bias']
        vmin = [34, 34, -0.5]
        vmax = [34.85, 34.85, 0.5]
        ctype = ['RdBu_r', 'RdBu_r', 'plusminus']
        region_colours = ['magenta']*2 + ['black']*4
        for n in range(3):
            ax = plt.subplot(gs[0,n])
            ax.axis('equal')
            img = circumpolar_plot(data_plot[n], ds, ax=ax, masked=True, make_cbar=False, title=titles[n], titlesize=13, vmin=vmin[n], vmax=vmax[n], ctype=ctype[n], lat_max=-63)
            if n == 2:
                ax.contour(x, y, mask, levels=[0.5], colors=('magenta'), linewidths=0.5)
            if n != 1:
                cax = cax = fig.add_axes([0.01+0.45*n, 0.1, 0.02, 0.6])
                plt.colorbar(img, cax=cax, extend='both')
        plt.suptitle('Bottom salinity (psu)', fontsize=18)
        finished_plot(fig, fig_name='figures/bwsalt_bias.png', dpi=300)

    # Save bias to NetCDF file
    data_diff = (ramp_up_bwsalt - obs_bwsalt).squeeze()
    data_diff = data_diff.where((ramp_up_bwsalt!=0)*ramp_up_bwsalt.notnull()*obs_bwsalt.notnull())
    ds_save = xr.Dataset({'bwsalt_bias':data_diff})
    print('Writing '+out_file)        
    ds_save.to_netcdf(out_file)

    # Calculate area-averaged bias
    dA = ds['area']*mask
    ukesm_mean = (ramp_up_bwsalt*dA).sum(dim=['x','y'])/dA.sum(dim=['x','y'])
    if eos == 'eos80':
        units = 'psu'
    else:
        units = ''
    print('UKESM mean '+str(ukesm_mean.data)+' '+units)
    # Might have to area-average over a smaller region with missing observational points
    mask_obs = mask.where(obs_bwsalt.notnull())
    dA_obs = ds['area']*mask_obs
    obs_mean = (obs_bwsalt*dA_obs).sum(dim=['x','y'])/dA_obs.sum(dim=['x','y'])
    print('Observational mean '+str(obs_mean.data)+' psu')
    bias = (ukesm_mean-obs_mean).item()
    print('UKESM bias '+str(bias))

    return bias


# Calculate the global warming implied by the salinity bias (from above), using a linear regression for the untipped sections of ramp-up simulations.
# Last calculation: salt_bias=-0.1517930177550042
def warming_implied_by_salinity_bias (salt_bias=None, base_dir='./'):

    pi_suite = 'cs495'
    smooth = 5*months_per_year
    timeseries_file = 'timeseries.nc'
    timeseries_file_um = 'timeseries_um.nc'
    p0 = 0.05
    regions = ['ross', 'filchner_ronne']
    sample_file = base_dir+'/time_averaged/piControl_grid-T.nc'

    if salt_bias is None:
        salt_bias = calc_salinity_bias(base_dir=base_dir)

    # Calculate area-weighting of each region
    area = []
    ds = xr.open_dataset(sample_file, decode_times=time_coder)
    for region in regions:
        mask = region_mask(region, ds, option='shelf')[0]
        dA = ds['area']*mask
        area.append(dA.sum(dim=['x','y']).item())
    total_area = area[0] + area[1]
    weights = [a/total_area for a in area]

    # Calculate regression for each region and ramp-up member
    num_ens = len(suites_by_scenario['ramp_up'])
    slopes = []
    intercepts = []
    r2 = []
    all_warming = []
    all_bwsalt = []
    for n in range(num_ens):
        suite = suites_by_scenario['ramp_up'][n]
        # Get timeseries of global warming
        warming_orig = global_warming(suite, pi_suite=pi_suite, base_dir=base_dir)
        ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file, decode_times=time_coder)
        # Get timeseries of Ross cavity temp to determine tipping: only want untipped sections, and Ross alway tips first
        ross_temp_orig = ds['ross_cavity_temp']
        # Trim and align; smooth
        warming, ross_temp = align_timeseries(warming_orig, ross_temp_orig)
        warming = moving_average(warming, smooth)
        ross_temp = moving_average(ross_temp, smooth)
        if ross_temp.max() > tipping_threshold:
            # Trim to just before the Ross tips
            t_end = np.argwhere(ross_temp.data > tipping_threshold)[0][0]
            warming = warming.isel(time_centered=slice(0,t_end))
        all_warming.append(warming)
        # Get timeseries of bottom salinity on the shelf
        # Area-average over Ross and FRIS regions together (i.e. weighted mean)
        bwsalt = None
        for region, weight in zip(regions, weights):
            bwsalt_tmp = ds[region+'_shelf_bwsalt']
            if bwsalt is None:
                bwsalt = weight*bwsalt_tmp
            else:
                bwsalt += weight*bwsalt_tmp
        # Trim and align; smooth; trim to just before Ross tips
        bwsalt = align_timeseries(warming_orig, bwsalt)[1]
        bwsalt = moving_average(bwsalt, smooth)
        if ross_temp.max() > tipping_threshold:
            bwsalt = bwsalt.isel(time_centered=slice(0,t_end))
        # Now find a linear regression of bwsalt in response to warming
        slope, intercept, r_value, p_value, std_err = linregress(warming, bwsalt)
        if p_value > p0:
            print('Warning: no significant trend for '+suite+', '+region)
        slopes.append(slope)
        intercepts.append(intercept)
        r2.append(r_value**2)
        all_bwsalt.append(bwsalt)
        ds.close()
    mean_slope = np.mean(slopes)
    implied_warming = salt_bias/mean_slope
    
    print('Temperature correction is '+str(implied_warming)+' degC')

    # Scatterplot
    fig, ax = plt.subplots()
    for n in range(num_ens):
        warming = all_warming[n]
        ax.plot(warming, all_bwsalt[n], '-', linewidth=1.5)
        # Add regression line
        x_vals = np.array([warming.min(), warming.max()])
        y_vals = slopes[n]*x_vals + intercepts[n]
        ax.plot(x_vals, y_vals, '-', color='black', linewidth=1)
    ax.grid(linestyle='dotted')
    plt.text(0.95, 0.95, 'Salinity bias '+str(np.round(salt_bias,3))+' psu', ha='right', va='top', transform=ax.transAxes)
    plt.text(0.95, 0.88, 'Ensemble mean slope '+str(np.round(mean_slope,3))+' psu/'+deg_string+'C', ha='right', va='top', transform=ax.transAxes)
    plt.text(0.95, 0.81, r'r$^2$ = '+str(np.round(np.amin(r2),3))+' - '+str(np.round(np.amax(r2),3)), ha='right', va='top', transform=ax.transAxes)
    plt.text(0.95, 0.74, 'Temperature adjustment '+str(np.round(implied_warming,3))+deg_string+'C', ha='right', va='top', transform=ax.transAxes)
    ax.set_xlabel('Simulated global warming ('+deg_string+'C)')
    ax.set_ylabel('Bottom salinity on Ross and Filchner-Ronne shelves (psu)')
    ax.set_title('Calculation of temperature adjustment', fontsize=14)
    finished_plot(fig, fig_name='figures/bwsalt_warming_regression.png', dpi=300)
    


# Plot cavity-mean temperature beneath Ross and FRIS as a function of shelf-mean bottom water salinity, in all scenarios. Colour the lines based on the global warming level relative to preindustrial, and indicate the magnitude of the salinity bias.
def plot_ross_fris_by_bwsalt (base_dir='./', compare_jacobs=False):

    regions = ['ross', 'filchner_ronne']
    title_prefix = ['a) ', 'b) ']
    obs_freshening = -0.17  # Ross, TEOS-10
    timeseries_file = 'timeseries.nc'
    timeseries_file_um = 'timeseries_um.nc'
    smooth = 5*months_per_year
    pi_suite = 'cs495'
    cmap = ['YlOrRd', 'YlGnBu'] #['Reds', 'Blues']
    p0 = 0.05
    tipping_temp = -1.9

    # Read timeseries from every experiment
    all_bwsalt = []
    all_cavity_temp = []
    all_warming = []
    max_warming = 0
    for n in range(len(regions)):
        data_bwsalt = []
        data_cavity_temp = []
        data_warming = []
        direction = []
        bwsalt_tip = []
        bwsalt_recover = []
        for scenario in suites_by_scenario:
            if 'piControl' in scenario or 'static_ice' in scenario:
                continue
            for suite in suites_by_scenario[scenario]:
                # Flag whether temperature is going up (ramp-up or stabilisation) or down (ramp-down)
                if 'ramp_down' in scenario or 'restabilise' in scenario:
                    direction.append(-1)
                elif 'ramp_up' in scenario or 'stabilise' in scenario:
                    direction.append(0)
                ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file, decode_times=time_coder)
                bwsalt = ds[regions[n]+'_shelf_bwsalt']
                cavity_temp = ds[regions[n]+'_cavity_temp']
                ds.close()
                warming = global_warming(suite, pi_suite=pi_suite, base_dir=base_dir)
                # Smooth and align
                if bwsalt.sizes['time_centered'] < smooth:
                    # Simulation hasn't run long enough to include
                    continue
                bwsalt = moving_average(bwsalt, smooth)
                cavity_temp = moving_average(cavity_temp, smooth)
                warming = moving_average(warming, smooth)+temp_correction
                bwsalt = align_timeseries(bwsalt, warming)[0]
                cavity_temp, warming = align_timeseries(cavity_temp, warming)
                max_warming = max(max_warming, warming.max())
                # Throw away any ramp-down data where global temp has overshot preindustrial and gone into negative, and apply correction after that
                data_bwsalt.append(bwsalt.where(warming>0))
                data_cavity_temp.append(cavity_temp.where(warming>0))
                data_warming.append(warming.where(warming>+temp_correction))
        all_bwsalt.append(data_bwsalt)
        all_cavity_temp.append(data_cavity_temp)
        all_warming.append(data_warming)

    # Now find the salinity at tipping and recovery points; for this we want trajectories, not suites. It means reading everything again unfortunately - can't find a cleaner way to do this (given some double-tipped trajectories)
    threshold_tip = []
    threshold_recover = []
    suite_lists = all_suite_trajectories()
    for region in regions:
        bwsalt_tip = []
        bwsalt_recover = []
        freshening_to_tip = []
        for suite_list in suite_lists:
            cavity_temp = moving_average(build_timeseries_trajectory(suite_list, region+'_cavity_temp', base_dir=base_dir), smooth)
            bwsalt = moving_average(build_timeseries_trajectory(suite_list, region+'_shelf_bwsalt', base_dir=base_dir), smooth)
            bwSA = moving_average(build_timeseries_trajectory(suite_list, region+'_shelf_bwSA', base_dir=base_dir), smooth)
            tips, t_tip = check_tip(cavity_temp=cavity_temp, smoothed=True, return_t=True, base_dir=base_dir)
            if tips:
                bwsalt_tip.append(bwsalt.isel(time_centered=t_tip))
                freshening_to_tip.append(bwSA.isel(time_centered=t_tip) - bwSA.isel(time_centered=0))
                recovers, t_recover = check_recover(cavity_temp=cavity_temp, smoothed=True, return_t=True, base_dir=base_dir)
                if recovers:
                    bwsalt_recover.append(bwsalt.isel(time_centered=t_recover))
        bwsalt_tip = np.unique(bwsalt_tip)
        bwsalt_recover = np.unique(bwsalt_recover)
        freshening_to_tip = np.unique(freshening_to_tip)
        # Print some statistics
        print(region)
        print('Shelf salinity at tipping has mean '+str(np.mean(bwsalt_tip))+' psu, std '+str(np.std(bwsalt_tip))+' psu')
        threshold_tip.append(np.mean(bwsalt_tip))
        if np.size(bwsalt_recover) > 0:
            print('Shelf salinity at recovery has mean '+str(np.mean(bwsalt_recover))+' psu, std '+str(np.std(bwsalt_recover))+' psu')
            threshold_recover.append(np.mean(bwsalt_recover))
            # 2-sample t-test to check if they're significantly different
            p_val = ttest_ind(bwsalt_tip, bwsalt_recover, equal_var=False)[1]
            distinct = p_val < p0
            if distinct:
                print('Significant difference (p='+str(p_val)+') of '+str(np.mean(bwsalt_tip)-np.mean(bwsalt_recover)))
            else:
                print('No significant difference (p='+str(p_val)+') of '+str(np.mean(bwsalt_tip)-np.mean(bwsalt_recover)))
        else:
            threshold_recover.append(None)
        print('Freshening of absolute salinity between beginning of ramp-up and tipping point has mean '+str(np.mean(freshening_to_tip))+', std '+str(np.std(freshening_to_tip)))
        if region == 'ross' and compare_jacobs:
            # Calculate bwsalt bias in TEOS-10
            ross_bias = calc_salinity_bias(base_dir=base_dir, eos='teos10', region='ross')
            # Compare observed freshening to freshening required to tip (plus correction)
            fresh_fraction = obs_freshening/(np.mean(freshening_to_tip)+ross_bias)*100
            print('Observed freshening is '+str(fresh_fraction)+'% of what is needed to tip')

    # Set up colour map to vary with global warming level
    norm = plt.Normalize(temp_correction, max_warming)
    num_suites = len(all_bwsalt[0])

    # Plot
    fig = plt.figure(figsize=(8,6))
    gs = plt.GridSpec(1,2)
    gs.update(left=0.1, right=0.98, bottom=0.25, top=0.93, wspace=0.2)
    cax1 = fig.add_axes([0.1, 0.12, 0.4, 0.03])
    cax2 = fig.add_axes([0.1, 0.075, 0.4, 0.03]) 
    for n in range(len(regions)):
        ax = plt.subplot(gs[0,n])
        for m in range(num_suites):
            # Plot each line with colour varying by global warming level
            points = np.array([all_bwsalt[n][m].data, all_cavity_temp[n][m].data]).T.reshape(-1,1,2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=truncate_colourmap(cmap[direction[m]], minval=0.2), norm=norm) #.3), norm=norm)
            lc.set_array(all_warming[n][m].data)
            lc.set_linewidth(1)
            img = ax.add_collection(lc)
            if direction[m] == 0:
                img_up = img
            else:
                img_down = img
        ax.grid(linestyle='dotted')
        ax.axhline(tipping_temp, color='black', linestyle='dashed')
        # Plot threshold salinity markers
        ax.plot(threshold_tip[n], tipping_temp, marker='o', markersize=5, markerfacecolor='Crimson', markeredgecolor='black')
        if threshold_recover[n] is not None:
            ax.plot(threshold_recover[n], tipping_temp, marker='o', markersize=5, markerfacecolor='DodgerBlue', markeredgecolor='black')
        ax.set_title(title_prefix[n]+region_names[regions[n]], fontsize=16)
        if n==0:
            ax.set_xlabel('Bottom salinity on continental shelf (psu)', fontsize=12)
            ax.set_ylabel('Temperature in ice shelf cavity ('+deg_string+'C)', fontsize=12)
    # Two colour bars: yellow/orange/red on the way up, yellow/green/blue on the way down
    cbar = plt.colorbar(img_up, cax=cax1, orientation='horizontal')
    cbar.set_ticklabels([])
    plt.colorbar(img_down, cax=cax2, orientation='horizontal')
    plt.text(0.3, 0.02, 'Effective global warming ('+deg_string+'C)', ha='center', va='center', fontsize=12, transform=fig.transFigure)
    plt.text(0.51, 0.135, 'ramp-up + zero-emission', ha='left', va='center', fontsize=10, transform=fig.transFigure)
    plt.text(0.51, 0.09, 'ramp-down', ha='left', va='center', fontsize=10, transform=fig.transFigure)
    # Manual legend
    colours = ['Crimson', 'DodgerBlue']
    labels = ['mean tipping', 'mean recovery']
    handles = []
    for m in range(len(colours)):
        handles.append(Line2D([0], [0], marker='o', markersize=5, markerfacecolor=colours[m], markeredgecolor='black', label=labels[m], linestyle=''))
    plt.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.95, -0.27))
    finished_plot(fig, fig_name='figures/ross_fris_by_bwsalt.png', dpi=300)


# Plot Amundsen Sea 500m temperature, barotropic velocity, and zero contour of barotropic streamfunction, averaged over 3 scenarios: (1) piControl, (2) 1.5K stabilisation, (3) 6K stabilisation.
def plot_amundsen_temp_velocity (base_dir='./'):

    scenarios = ['piControl', '1.5K', '6K']
    num_scenarios = len(scenarios)
    title_prefix = [r'$\bf{a}$. ', r'$\bf{b}$. ', r'$\bf{c}$. ']
    scenario_titles = ['Preindustrial', '1.5'+deg_string+'C stabilisation', '6'+deg_string+'C stabilisation']
    mean_dir = base_dir + '/time_averaged/'
    depth_temp = 500
    [xmin, xmax] = [-170, -62]
    [ymin, ymax] = [-76, -60]
    vel_scale = 0.4
    domain_cfg = '/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'
    [vmin, vmax] = [0, 4.5]

    # Set up grid for plotting
    ds_grid = xr.open_dataset(mean_dir+scenarios[0]+'_grid-T.nc', decode_times=time_coder).squeeze()
    lon_edges = cfxr.bounds_to_vertices(ds_grid['bounds_lon'], 'nvertex')
    lat_edges = cfxr.bounds_to_vertices(ds_grid['bounds_lat'], 'nvertex')
    bathy, draft, ocean_mask, ice_mask = calc_geometry(ds_grid)
    ds_domcfg = xr.open_dataset(domain_cfg, decode_times=time_coder).squeeze()
    ds_domcfg = ds_domcfg.isel(y=slice(0, ds_grid.sizes['y']))

    # Inner functions
    # Read a variable
    def read_var (var_name, scenario, gtype):
        ds = xr.open_dataset(mean_dir + scenario + '_grid-' + gtype + '.nc', decode_times=time_coder)
        data = ds[var_name].squeeze()
        ds.close()
        return data
    # Interpolate a variable to a given depth
    def interp_depth (data_3d, gtype, depth0):
        data_3d = data_3d.where(data_3d!=0)
        dim = 'depth'+gtype.lower()
        data_interp = data_3d.interp({dim:depth0})
        return data_interp
    # Vertically average a velocity component
    def barotropic (vel, gtype):
        dz = read_var('thkcello', scenario, gtype)
        mask_3d = xr.where(vel==0, 0, 1)
        dim = 'depth'+gtype.lower()
        vel_avg = (vel*mask_3d*dz).sum(dim=dim)/(mask_3d*dz).sum(dim=dim)
        return vel_avg
    # Read and vertically average a velocity component and interpolate it to the tracer grid
    def process_vel (scenario, direction):
        data_3d = read_var(direction+'o', scenario, direction.upper())
        data = barotropic(data_3d, direction.upper())
        data_t = interp_grid(data, direction, 't', periodic=True, halo=True)
        return data_t
    # Mask out anything beyond region of interest, plus ice shelf cavities
    def apply_mask (data, mask_shallow=False):
        data = data.where((ds_grid['nav_lon']>=xmin-2)*(ds_grid['nav_lon']<=xmax+2)*(ds_grid['nav_lat']>=ymin-2)*(ds_grid['nav_lat']<=ymax+2))
        data = data.where(np.invert(ice_mask))
        if mask_shallow:
            # Also mask anything shallower than 500m
            data = data.where(bathy >= depth_temp)
        return data

    # Read all the data
    all_temp = []
    all_u = []
    all_v = []
    all_strf = []
    for scenario in scenarios:
        # Temperature, interpolated to 500m
        temp_3d = read_var('thetao', scenario, 'T')
        temp = interp_depth(temp_3d, 'T', depth_temp)
        all_temp.append(apply_mask(temp))
        # Barotropic velocity, interpolated to tracer grid
        u = process_vel(scenario, 'u')
        v = process_vel(scenario, 'v')
        # Rotate to geographic components
        ug, vg = rotate_vector(u, v, domain_cfg, periodic=True, halo=True)
        all_u.append(apply_mask(ug))
        all_v.append(apply_mask(vg))
        # Barotropic streamfunction, interpolated to tracer grid
        ds_u = xr.open_dataset(mean_dir+scenario+'_grid-U.nc', decode_times=time_coder).squeeze()
        ds_v = xr.open_dataset(mean_dir+scenario+'_grid-V.nc', decode_times=time_coder).squeeze()
        strf = barotropic_streamfunction(ds_u, ds_v, ds_domcfg, periodic=True, halo=True)
        all_strf.append(apply_mask(strf, mask_shallow=True))

    # Plot
    fig = plt.figure(figsize=(6,8))
    gs = plt.GridSpec(num_scenarios, 1)
    gs.update(left=0.1, right=0.95, bottom=0.1, top=0.87, hspace=0.3)
    cax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
    cmap, vmin, vmax = set_colours(all_temp[0], ctype='RdBu_r', vmin=vmin, vmax=vmax)
    mask_plot = np.invert(ocean_mask*np.invert(ice_mask))
    mask_plot = apply_mask(mask_plot)
    mask_plot = mask_plot.where(mask_plot)
    for n in range(num_scenarios):
        ax = plt.subplot(gs[n,0])
        # Shade temperature 
        img = ax.pcolormesh(lon_edges, lat_edges, all_temp[n], cmap=cmap, vmin=vmin, vmax=vmax)
        # Overlay land+ice mask in grey
        ax.pcolormesh(lon_edges, lat_edges, mask_plot, cmap=cl.ListedColormap(['DarkGrey']), linewidth=0)
        # Overlay zero contour of streamfunction
        cs = ax.contour(ds_grid['nav_lon'], ds_grid['nav_lat'], all_strf[n], levels=[0], colors=('magenta'), linewidths=2, linestyles='solid')
        # Overlay every third velocity vector in black
        q = ax.quiver(ds_grid['nav_lon'].data[::4,::4], ds_grid['nav_lat'].data[::4,::4], all_u[n].data[::4,::4], all_v[n].data[::4,::4], scale=vel_scale, color='black')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.tick_params(direction='in')
        ax.set_title(title_prefix[n]+scenario_titles[n], fontsize=14)
        if n == 0:
            latlon_axes(ax)
            ax.clabel(cs, cs.levels, inline=True, fmt={0:'0 Sv'}, fontsize=10)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        if n == num_scenarios-1:
            plt.colorbar(img, cax=cax, orientation='horizontal', extend='max')
            plt.text(0.18, 0.04, deg_string+'C', fontsize=12, ha='right', va='bottom', transform=fig.transFigure)
            ax.quiverkey(q, 0.9, 0.05, 0.01, '1 cm/s', labelpos='E', coordinates='figure')            
    plt.suptitle('West Antarctic ocean temperature (500m)\nand barotropic velocity', fontsize=15)
    finished_plot(fig, fig_name='figures/amundsen_temp_velocity.png', dpi=300)


# Helper function to construct a nice suite title describing the given trajectory
def trajectory_title (suites):

    if not isinstance(suites, list):
        suites = suites.split('-')
    title = ''
    for suite in suites:
        # Figure out what sort of scenario it is
        for scenario in suites_by_scenario:
            if suite in suites_by_scenario[scenario]:
                if 'ramp_up' in scenario:
                    title += 'Ramp up'
                elif 'stabilise' in scenario:
                    title += ' to '+scenario[:scenario.index('K_')]+deg_string+'C, zero emissions'
                elif 'ramp_down' in scenario:
                    # Figure out length of overshoot
                    for length in suites_overshoot_lengths:
                        if suite in suites_overshoot_lengths[length]:
                            title += ' for '+length
                    # Figure out ramp-down rate
                    for rate in suites_ramp_down_rates:
                        if suite in suites_ramp_down_rates[rate]:
                            title += ', ramp down -'+rate
    return title


# Make an animation showing all sorts of things for the given trajectory.
def dashboard_animation (suite_string, region, base_dir='./', out_dir='animations/', only_precompute=False):

    pi_suite = 'cs495'
    timeseries_file = 'timeseries.nc'
    timeseries_file_um = 'timeseries_um.nc'
    smooth = 5*months_per_year
    suite_list = suite_string.split('-')
    if region not in ['ross', 'filchner_ronne']:
        raise Exception('Invalid region '+region)

    # Assemble main title of plot, to describe the trajectory
    sim_title = trajectory_title(suite_list)

    print('Reading timeseries')
    # Assemble timeseries, 5-year smoothed
    # First get preindustrial baseline temp
    baseline_temp = pi_baseline_temp(pi_suite=pi_suite, base_dir=base_dir)
    # Global mean warming
    warming = build_timeseries_trajectory(suite_list, 'global_mean_sat', base_dir=base_dir, timeseries_file=timeseries_file_um, offset=-1*baseline_temp)
    warming = moving_average(warming, smooth)
    # Inner function for oceanographic timeseries
    def process_timeseries (var_name):
        data = build_timeseries_trajectory(suite_list, var_name, base_dir=base_dir, timeseries_file = timeseries_file)
        data = moving_average(data, smooth)
        # Align with warming timeseries
        data, warming_tmp = align_timeseries(data, warming)
        return data, warming_tmp
    shelf_bwsalt = process_timeseries(region+'_shelf_bwsalt')[0]
    bwtemp = process_timeseries(region+'_bwtemp')[0]
    cavity_temp = process_timeseries(region+'_cavity_temp')[0]
    # For the last one, overwrite warming timeseries (now aligned)
    massloss, warming = process_timeseries(region+'_massloss')

    # Identify if/when cavity tips or recovers
    tips, date_tip, t_tip = check_tip(cavity_temp=cavity_temp, smoothed=True, return_date=True, return_t=True, base_dir=base_dir)
    if tips:
        tip_string = 'Tips in '+str(date_tip.dt.year.item())+' ('+str(round(warming[t_tip].item(),2))+deg_string+'C)'
        recovers, date_recover, t_recover = check_recover(cavity_temp=cavity_temp, smoothed=True, return_date=True, return_t=True, base_dir=base_dir)
        if recovers:
            recover_string = 'Recovers in '+str(date_recover.dt.year.item())+' ('+str(round(warming[t_recover].item(),2))+deg_string+'C)'
        else:
            recover_string = 'Does not recover'
            recover_time = None                
    else:
        tip_string = 'Does not tip'
        recover_string = ''
        tip_time = None
        recover_time = None

    # Set up maps
    # Open one file just to get coordinates
    # Use the file with the most retreated GL - end of ramp up
    file_list = []
    for f in os.listdir(base_dir+'/cx209'):
        if f.startswith('nemo_cx209o_1m_') and f.endswith('_grid-T.nc'):
            file_list.append(f)
    file_list.sort()
    f = file_list[-1]
    grid = xr.open_dataset(base_dir+'/cx209/'+f, decode_times=time_coder)
    # Choose bounds of map to show
    mask = region_mask(region, grid)[0]
    x, y = polar_stereo(grid['nav_lon'], grid['nav_lat'])
    xmin = x.where(mask).min()
    xmax = x.where(mask).max()
    ymin = y.where(mask).min()
    ymax = y.where(mask).max()    
    plot_mask = (x >= xmin)*(x <= xmax)*(y >= ymin)*(y <= ymax)
    # Trim in latitude direction to save memory: find bounds
    # (trimming in longitude is harder with periodic boundary)
    lat_min = grid['nav_lat'].where(plot_mask).min()
    lat_max = grid['nav_lat'].where(plot_mask).max()
    jmin = np.argwhere(grid['nav_lat'].min(dim='x').data > lat_min.item())[0][0] - 1
    jmax = np.argwhere(grid['nav_lat'].max(dim='x').data > lat_max.item())[0][0]
    grid = grid.isel(y=slice(jmin,jmax))
    plot_mask = plot_mask.isel(y=slice(jmin,jmax))
    # Set up plotting coordinates
    lon_edges = cfxr.bounds_to_vertices(grid['bounds_lon'], 'nvertex')
    lat_edges = cfxr.bounds_to_vertices(grid['bounds_lat'], 'nvertex')
    x_edges, y_edges = polar_stereo(lon_edges, lat_edges)
    x_bg, y_bg = np.meshgrid(np.linspace(x_edges.min(), x_edges.max()), np.linspace(y_edges.min(), y_edges.max()))
    mask_bg = np.ones(x_bg.shape)

    # Calculate annual means of bwtemp, bwsalt, ismr within the plotting region.
    print('Reading 2D data')
    precomputed_file = out_dir+'/precomputed/'+suite_string+'_'+region+'.nc'
    if os.path.isfile(precomputed_file):
        ds_2D = xr.open_dataset(precomputed_file, decode_times=time_coder)
        num_years = ds_2D.sizes['time_centered']
    else:
        ds_2D = None
        num_years = 0
    # Select the middle month of each 12 year chunk of timeseries data
    for t in range(12*num_years+6, massloss.sizes['time_centered'], 12):
        # What year is it?
        year = massloss.coords['time_centered'][t].dt.year.item()
        print('...'+str(year))
        # What stage in the simulation is it?
        stype = int(massloss.coords['scenario_type'][t].item())
        stype_mapping = [1, 0, 3, 2]
        suite = suite_list[stype_mapping[stype]]
        sim_dir = base_dir+'/'+suite
        # Build list of the file patterns we're going to read for this year
        nemo_files = []
        for f in os.listdir(sim_dir):
            if os.path.isdir(f'{sim_dir}/{f}'): continue
            if f.startswith('nemo_'+suite+'o_1m_'+str(year)) and f.endswith('-T.nc'):
                # Extract date codes
                date_codes = re.findall(r'\d{4}\d{2}\d{2}', f)
                file_pattern = sim_dir+'/nemo_'+suite+'o_1m_'+date_codes[0]+'-'+date_codes[1]+'*-T.nc'
                if file_pattern not in nemo_files:
                    nemo_files.append(file_pattern)
        nemo_files.sort()
        # Expect 12 file patterns (one for each month), each containing 2 files (grid-T and isf-T)
        if len(nemo_files) > months_per_year:
            raise Exception('Too many NEMO files for '+suite+', '+str(year))
        if len(nemo_files) < months_per_year:
            print('Warning: incomplete year for '+suite+', '+str(year))
        # Now read each month
        bwsalt_accum = None
        bwtemp_accum = None
        ismr_accum = None
        for file_pattern in nemo_files:
            ds = xr.open_mfdataset(file_pattern, decode_times=time_coder).isel(y=slice(jmin,jmax))
            ds.load()
            ds = ds.swap_dims({'time_counter':'time_centered'}).drop_vars(['time_counter'])
            if os.path.isfile(file_pattern.replace('*','_grid')):
                bwsalt_tmp = ds['sob'].where(ds['sob']>0).where(plot_mask)
                bwtemp_tmp = ds['tob'].where(ds['sob']>0).where(plot_mask)
                if bwsalt_accum is None:
                    bwsalt_accum = bwsalt_tmp
                    bwtemp_accum = bwtemp_tmp
                else:
                    bwsalt_accum = xr.concat([bwsalt_accum, bwsalt_tmp], dim='time_centered')
                    bwtemp_accum = xr.concat([bwtemp_accum, bwtemp_tmp], dim='time_centered')
            else:
                print('Warning: missing '+file_pattern.replace('*','_grid'))
            if os.path.isfile(file_pattern.replace('*','_isf')):
                ismr_tmp = convert_ismr(ds['sowflisf'].where(ds['sowflisf']!=0)).where(plot_mask).drop_vars({'time_counter'})
                if ismr_accum is None:
                    ismr_accum = ismr_tmp
                else:
                    ismr_accum = xr.concat([ismr_accum, ismr_tmp], dim='time_centered')
            else:
                print('Warning: missing '+file_pattern.replace('*','_isf'))
            ds.close()
        # Calculate (annual) means and save
        ds_2D_tmp = xr.Dataset({'bwsalt':bwsalt_accum, 'bwtemp':bwtemp_accum, 'ismr':ismr_accum}).mean(dim='time_centered')
        ds_2D_tmp = ds_2D_tmp.assign_coords(time_centered=bwsalt_accum.coords['time_centered'][6]).expand_dims('time_centered')
        if ds_2D is None:
            ds_2D = ds_2D_tmp
        else:
            ds_2D = xr.concat([ds_2D, ds_2D_tmp], dim='time_centered')
    # Save data for precomputing next time - 2-stage overwrite in case of errors
    print('Writing '+precomputed_file)
    ds_2D.to_netcdf(precomputed_file+'_tmp')
    os.rename(precomputed_file+'_tmp', precomputed_file)
    if only_precompute:
        return
        
    var_plot_2D = ['bwsalt', 'bwtemp', 'ismr']
    titles_2D = ['Bottom salinity (psu)', 'Bottom temperature ('+deg_string+'C)', 'Ice shelf melt rate (m/y)']
    # Calculate variable min/max over all years
    vmin = [ds_2D[var].min() for var in var_plot_2D]
    vmax = [ds_2D[var].max() for var in var_plot_2D]
    ctype = ['viridis', 'viridis', 'ismr']
    cmap = [set_colours(ds_2D[var_plot_2D[n]].isel(time_centered=0), ctype=ctype[n], vmin=vmin[n], vmax=vmax[n])[0] for n in range(3)]
    num_years = ds_2D.sizes['time_centered']

    print('Setting up plot')

    # Initialise the plot
    fig = plt.figure(figsize=(9,7))
    gs = plt.GridSpec(2,3)
    gs.update(left=0.07, right=0.98, bottom=0.08, top=0.86, wspace=0.22, hspace=0.4)
    cax = [fig.add_axes([0.095+0.33*n, 0.04, 0.2, 0.02]) for n in range(3)]
    ax = []
    for n in range(2):
        for m in range(3):
            ax.append(plt.subplot(gs[n,m]))

    # Inner function to plot a frame
    def plot_one_frame (t):
        if t == 0:
            # Trajectory information at the top
            plt.suptitle(suite_string+'\n'+sim_title, fontsize=14)
        # Top row: timeseries: plot 12 months at a time, colour-coded by simulation type
        t_start = t*months_per_year
        t_end = t_start + months_per_year
        if t != 0:
            # Make sure to connect to lines from before
            t_start -= 1
        stype = massloss.coords['scenario_type'][t_start+6].item()
        if stype == 1:
            colour = 'Crimson'
        elif stype in [0, -2]:
            colour = 'Grey'
        elif stype == -1:
            colour = 'DodgerBlue'
        # Top left panel: temperature in cavity vs bottom salinity on continental shelf
        ax[0].plot(shelf_bwsalt[t_start:t_end], cavity_temp[t_start:t_end], color=colour, linewidth=1)
        ax[0].set_xlim([shelf_bwsalt.min()-0.02, shelf_bwsalt.max()+0.02])
        ax[0].set_ylim([cavity_temp.min()-0.02, max(tipping_threshold, cavity_temp.max())+0.02])
        if t == 0:
            ax[0].grid(linestyle='dotted')
            ax[0].axhline(tipping_threshold, color='black', linestyle='dashed')
            ax[0].set_xlabel('Bottom salinity on shelf (psu)', fontsize=10)
            ax[0].set_title('Temperature in cavity ('+deg_string+'C)', fontsize=12)
        # Other two top panels: bottom temperature (shelf+cavity) and massloss vs global warming level
        for n, ts_data, title, offset in zip(range(2), [bwtemp, massloss], ['Bottom temp, shelf+cavity ('+deg_string+'C)', 'Basal mass loss (Gt/y)'], [0.02, 5]):
            ax[n+1].plot(warming[t_start:t_end], ts_data[t_start:t_end], color=colour, linewidth=1)
            ax[n+1].set_xlim([warming.min()-0.02, warming.max()+0.02])
            ax[n+1].set_ylim([ts_data.min()-offset, ts_data.max()+offset])
            if t == 0:
                ax[n+1].grid(linestyle='dotted')
                ax[n+1].set_xlabel('Global warming ('+deg_string+'C)', fontsize=10)
                ax[n+1].set_title(title, fontsize=12)
        # Bottom three panels: maps of 2D data
        for n in range(3):
            # This time clear all previous plotting data - we're not adding to it
            ax[n+3].cla()
            # Shade land in grey
            ocean_mask = xr.where(ds_2D['bwsalt'].isel(time_centered=t)>0, 1, 0)
            ocean_mask = ocean_mask.where(ocean_mask)
            ax[n+3].pcolormesh(x_bg, y_bg, mask_bg, cmap=cl.ListedColormap(['DarkGrey']))
            ax[n+3].pcolormesh(x_edges, y_edges, ocean_mask, cmap=cl.ListedColormap(['white']))
            # Plot the data
            img = ax[n+3].pcolormesh(x_edges, y_edges, ds_2D[var_plot_2D[n]].isel(time_centered=t), cmap=cmap[n], vmin=vmin[n], vmax=vmax[n])
            ax[n+3].set_xlim([xmin, xmax])
            ax[n+3].set_ylim([ymin, ymax])
            ax[n+3].set_xticks([])
            ax[n+3].set_yticks([])
            ax[n+3].set_title(titles_2D[n], fontsize=12)
            if t == 0:
                cbar = plt.colorbar(img, cax=cax[n], orientation='horizontal')
        # Print tipping information
        plt.text(0.99, 0.99, tip_string, fontsize=14, ha='right', va='top', transform=fig.transFigure)
        plt.text(0.99, 0.95, recover_string, fontsize=14, ha='right', va='top', transform=fig.transFigure)
        # Print the year and global warming level
        year_string = str(massloss.coords['time_centered'][t_start+6].dt.year.item())
        temp_string = str(round(warming[t_start+6].item(),2))+deg_string+'C'        
        year_text = plt.text(0.01, 0.99, year_string, fontsize=14, ha='left', va='top', transform=fig.transFigure)
        temp_text = plt.text(0.01, 0.95, temp_string, fontsize=14, ha='left', va='top', transform=fig.transFigure)
        
    # First frame
    plot_one_frame(0)

    # Function to update figure with the given frame
    def animate(t):
        plot_one_frame(t)

    # Call this for each frame
    anim = animation.FuncAnimation(fig, func=animate, frames=list(range(num_years)))
    writer = animation.FFMpegWriter(bitrate=5000, fps=5)
    print('Saving animation')
    anim.save(out_dir+'/'+suite_string+'_'+region+'.mp4', writer=writer)


# Call the batch script precompute_animations.sh for every trajectory. Assumes base_dir and out_dir are default values.
def precompute_all_animations ():

    for suite_list in all_suite_trajectories():
        suite_string = '-'.join(suite_list)
        for region in ['ross', 'filchner_ronne']:
            command = 'sbatch --export=SUITES='+suite_string+',CAVITY='+region+' precompute_animation.sh'
            print(command)
            sbatch_id = subprocess.check_output(command, shell=True, text=True)
            print(sbatch_id)


# Call dashboard_animation for every precomputed file.
# To do: only the ones which have actually changed since last time - look at timestamps on precomputed files vs MP4s?
def animate_all (out_dir='animations/'):

    for f in os.listdir(out_dir+'/precomputed/'):
        if f.endswith('.nc'):
            # Extract suite string and region name
            suite_string = f[:f.index('_')]
            region = f[f.index('_')+1:f.index('.nc')]
            print('Processing '+suite_string+' '+region)
            dashboard_animation(suite_string, region, out_dir=out_dir)
            plt.close('all')


# Plot a histogram of how long after stabilisation tipping happens, for each region. Also print some statistics.
def tipping_time_histogram (base_dir='./', fig_name=None):

    regions = ['ross', 'filchner_ronne']
    smooth = 5*months_per_year
    timeseries_file = 'timeseries.nc'
    num_bins = 10

    all_times = []
    for region in regions:
        times = []
        # Find all trajectories of cavity temperature
        cavity_temp_ts = all_timeseries_trajectories(region+'_cavity_temp', base_dir=base_dir)[0]
        # Loop over them
        for n in range(len(cavity_temp_ts)):
            cavity_temp = moving_average(cavity_temp_ts[n], smooth)
            # Find time index of when stabilisation starts
            if cavity_temp.scenario_type.min() == 1:
                # Perpetual ramp-up; skip it
                continue
            stab_time = np.argwhere(cavity_temp.scenario_type.data==0)[0][0]
            tips, tip_time = check_tip(cavity_temp=cavity_temp, smoothed=True, return_t=True, base_dir=base_dir)
            if tips and  tip_time >= stab_time:
                # Tips after stabilisation
                # Calculate years since emissions stabilised
                times.append((tip_time-stab_time)/months_per_year)
        # Throw away duplicates
        times = np.unique(times)
        all_times.append(times)
        # Print some statistics
        print(region+':')
        print(str(np.size(times))+' tips happen after stabilisation')
        print('Range: '+str(np.amin(times))+' to '+str(np.amax(times))+' years')
        print('Mean: '+str(np.mean(times))+' years')
        print('Standard deviation: '+str(np.std(times))+' years')
    tmax = np.amax([np.amax(times) for times in all_times])
    bins = np.linspace(0, tmax, num=num_bins)

    # Plot
    fig = plt.figure(figsize=(5,5))
    gs = plt.GridSpec(2,1)
    gs.update(left=0.05, right=0.99, bottom=0.1, top=0.85, hspace=0.4)
    for r in range(len(regions)):
        ax = plt.subplot(gs[r,0])
        ax.hist(all_times[r], bins=bins)
        ax.set_title(region_names[regions[r]], fontsize=12)
        if r==0:
            ax.set_ylabel('# simulations', fontsize=10)
        if r==1:
            ax.set_xlabel('years since emissions stopped', fontsize=10)
    plt.suptitle('Tipping points reached after climate stabilisation', fontsize=14)
    finished_plot(fig, fig_name=fig_name)


# After downloading some more variables from MASS for one suite, merge these files (in a subdirectory) into the main files.
def merge_sfc_files (suite='cx209', subdir='sfc'):

    for f in os.listdir(suite+'/'+subdir):
        if not f.endswith('.nc'):
            continue
        print('Processing '+f)
        ds1 = xr.open_dataset(suite+'/'+f, decode_times=time_coder)
        ds2 = xr.open_dataset(suite+'/'+subdir+'/'+f, decode_times=time_coder)
        ds = xr.merge([ds1, ds2])
        ds.to_netcdf(suite+'/'+f+'_tmp')
        ds.close()
        ds1.close()
        ds2.close()
        os.rename(suite+'/'+f+'_tmp', suite+'/'+f)    


# Calculate some extra timeseries for one suite
def sfc_FW_timeseries (suite='cx209', base_dir='./'):

    update_simulation_timeseries(suite, ['all_iceberg_melt', 'all_pminuse', 'all_runoff', 'all_seaice_meltfreeze'], timeseries_file='timeseries_sfc.nc', sim_dir=base_dir+'/'+suite+'/', freq='m', halo=True, gtype='T')


def bwsalt_timeseries (suite, base_dir='./'):

    regions = ['filchner_trough', 'ronne_depression', 'LAB_trough', 'drygalski_trough']
    update_simulation_timeseries(suite, [region+'_shelf_bwsalt' for region in regions], timeseries_file='timeseries_bwsalt.nc', sim_dir=base_dir+'/'+suite+'/', freq='m', halo=True, gtype='T')
    

# Helper function to get the start and end dates of each suite-segment in a trajectory
def find_stages_start_end (suite_list, base_dir='./', timeseries_file='timeseries.nc'):
    stage_start = []
    stage_end = []
    for suite in suite_list:
        file_path = base_dir+'/'+suite+'/'+timeseries_file
        ds = xr.open_dataset(file_path, decode_times=time_coder)
        stage_start.append(ds.time_centered[0].item())
        stage_end.append(ds.time_centered[-1].item())
        ds.close()
    # Now deal with overlaps
    stage_end[:-1] = stage_start[1:]
    return stage_start, stage_end


# Helper function to find if/when a given suite cools below PI.
def truncate_rampdown_PI (suite):

    # Check if 5-year running mean GMST goes below PI and stays there
    smooth = 5*months_per_year
    warming = moving_average(global_warming(suite), smooth)
    if warming.min() < 0 and warming.isel(time_centered=-1) < 0:
        t_start = np.where(warming.data < 0)[0][0]
        for t in range(t_start, warming.sizes['time_centered']):
            if warming.isel(time_centered=slice(t,None)).max() < 0:
                date_end = warming.coords['time_centered'].data[t]
                break
        return date_end
    else:
        return None


# Helper function to get time axis in years since beginning; pass DataArray with coordinate 'time_centered'
def time_in_years (data, year0=None, return_year0=False):
    year0 = data.time_centered[0].dt.year.item()
    years = np.array([(date.dt.year.item() - year0) + (date.dt.month.item() - 1)/months_per_year + 0.5 for date in data.time_centered])
    if return_year0:
        return years, year0
    else:
        return years    


# Timeseries of various freshwater fluxes, relative to preindustrial baseline, for one trajectory.
def plot_FW_timeseries (base_dir='./'):

    suite_string = 'cx209-cz378-de943'
    suite_list = suite_string.split('-')
    pi_suite = 'cs568'  # Want evolving ice PI control so the ice mass is partitioned the same way into calving, basal melting, runoff.
    pi_years = 50
    var_names = ['all_massloss', 'all_seaice_meltfreeze', 'all_pminuse', 'all_runoff', 'all_iceberg_melt']
    var_titles = ['Ice shelves', 'Sea ice', 'Precip - evap', 'Surface melt',  'Icebergs']
    timeseries_files = ['timeseries.nc'] + 4*['timeseries_sfc.nc']
    factors = [rho_fw/rho_ice*1e6/sec_per_year] + 4*[1e-3/sec_per_year]
    units = 'mSv'
    colours = [(0.6,0.6,0.6), (0.8,0.47,0.65), (0,0.62,0.45), (0,0.45,0.7), (0.9,0.62,0)]   
    #fw_colours = [(0.6,0.6,0.6), '#e41a1c', '#4daf4a', '#984ea3', '#dede00']
    num_vars = len(var_names)
    #ismr_regions = ['ross', 'filchner_ronne', 'west_antarctica', 'east_antarctica', 'all']
    #ismr_titles = [region_names[region] for region in ismr_regions[:-1]] + ['Total']
    #ismr_colours = [(0,0.45,0.7), (0.8,0.47,0.65), (0,0.62,0.45), (0.9,0.62,0), (0.6,0.6,0.6)]
    tip_regions = ['ross', 'filchner_ronne'] 
    tip_titles = ['Ross', 'FRIS']
    smooth = 10*months_per_year
    stage_colours = ['Crimson', 'white', 'DodgerBlue']
    label_y = 120
    #sample_file = base_dir+'/time_averaged/piControl_grid-T.nc'
    #ds_grid = xr.open_dataset(sample_file).squeeze()
    
    # Inner function to read the variable from the main trajectory and subtract the PI baseline
    def read_var_anomaly (var, fname):
        data = build_timeseries_trajectory(suite_list, var, base_dir=base_dir, timeseries_file=fname)
        ds = xr.open_dataset(base_dir+'/'+pi_suite+'/'+fname, decode_times=time_coder)
        baseline = ds[var].isel(time_centered=slice(0,pi_years*months_per_year)).mean(dim='time_centered')
        ds.close()
        return data-baseline

    # Loop over variables and read all the data
    data_plot = []
    for v in range(num_vars):
        data = read_var_anomaly(var_names[v], timeseries_files[v])
        if v == 0:
            # Save first year of simulation, before smoothing
            year0 = data.time_centered[0].dt.year.item()
        # Convert units and smooth in time
        data = moving_average(data, smooth)*factors[v]
        data_plot.append(data)
    #ismr_plot = []
    #for region in ismr_regions:
    #    data = read_var_anomaly(region+'_massloss', timeseries_files[0])
    #    data = moving_average(data, smooth)*factors[0]
    #    ismr_plot.append(data)

    # Find the first and last year of each stage in the simulation
    stage_start, stage_end = find_stages_start_end(suite_list, base_dir=base_dir, timeseries_file=timeseries_files[0])
    stage_start = [date.year-year0 for date in stage_start]
    stage_end = [date.year-year0 for date in stage_end]
    # Check if ramp-down cooled below PI
    date_end = truncate_rampdown_PI(suite_list[-1])
    if date_end is not None:
        stage_end[-1] = date_end.year-year0

    # Find the time of tipping and recovery for each cavity
    tip_times = [check_tip(suite=suite_string, region=region, return_date=True, base_dir=base_dir)[1] for region in tip_regions]
    recover_times = [check_recover(suite=suite_string, region=region, return_date=True, base_dir=base_dir)[1] for region in tip_regions]

    # Plot
    fig = plt.figure(figsize=(8,3.5))
    gs = plt.GridSpec(1,1)
    gs.update(left=0.08, right=0.99, bottom=0.15, top=0.9, hspace=0.05)
    ax = plt.subplot(gs[0,0])
    for v in range(len(data_plot)):
        # Get time axis in years since beginning
        years = time_in_years(data_plot[v], year0=year0)
        ax.plot(years, data_plot[v], '-', color=colours[v], label=var_titles[v], linewidth=1)
    # Shade each stage
    for t in range(len(stage_start)):
        ax.axvspan(stage_start[t], stage_end[t], alpha=0.1, color=stage_colours[t])
    # Dashed lines to show tipping and recovery
    for r in range(len(tip_regions)):
        if tip_times[r] is not None:
            tip_year = tip_times[r].dt.year.item() - year0
            ax.axvline(tip_year, color='black', linestyle='dashed', linewidth=1)
            plt.text(tip_year, label_y, tip_titles[r]+' tips', ha='left', va='top', rotation=-90)
            if recover_times[r] is not None:
                recover_year = recover_times[r].dt.year.item() - year0
                ax.axvline(recover_year, color='black', linestyle='dashed', linewidth=1)
                plt.text(recover_year, label_y, tip_titles[r]+' recovers', ha='left', va='top', rotation=-90)
    ax.grid(linestyle='dotted')
    ax.axhline(0, color='black')
    ax.set_ylabel(units)
    ax.set_xlim([stage_start[0], stage_end[-1]])
    ax.set_title('Antarctic freshwater fluxes (anomalies from preindustrial)', fontsize=14)
    plt.text(615, ax.get_ylim()[0]-7, 'years', ha='left', va='top')
    plt.text(0.5, 0.01, trajectory_title(suite_string), ha='center', va='bottom', transform=fig.transFigure, fontsize=12)
    ax.legend(loc='upper left')
    finished_plot(fig, fig_name='figures/FW_timeseries.png', dpi=300)


# Plot shelf bwsalt and its time-derivative for the Ross and FRIS regions in untipped trajectories, with the given level of smoothing (in years).
def plot_untipped_salinity (smooth=30, base_dir='./'):

    regions = ['ross', 'filchner_ronne']
    stype = [1, 0, -1]
    colours = ['Crimson', 'Grey', 'DodgerBlue']
    labels = ['ramp-up', 'stabilise', 'ramp-down']
    suite_sequences = all_suite_trajectories()

    # Make one figure per untipped trajectory per region
    for region in regions:
        # Find untipped trajectories
        for suite_list in suite_sequences:
            suite_string = '-'.join(suite_list)
            if not check_tip(suite=suite_string, region=region):
                bwsalt = moving_average(build_timeseries_trajectory(suite_list, region+'_shelf_bwsalt', base_dir=base_dir), smooth*months_per_year)
                # Time-derivative of smoothed array; convert from days to centuries (30-day months in UKESM)
                ds_dt = bwsalt.differentiate('time_centered', datetime_unit='D')*360*1e2
                # Smooth it again
                ds_dt = moving_average(ds_dt, smooth*months_per_year)
                # Plot different phases in different colours
                fig = plt.figure(figsize=(5,7))
                gs = plt.GridSpec(2,1)
                gs.update(left=0.15, right=0.95, bottom=0.1, top=0.9, hspace=0.2)
                ax1 = plt.subplot(gs[0,0])
                ax2 = plt.subplot(gs[1,0])
                for m in range(len(stype)):
                    index = bwsalt.scenario_type == stype[m]
                    ax1.plot_date(bwsalt.time_centered.where(bwsalt.scenario_type==stype[m], drop=True), bwsalt.where(bwsalt.scenario_type==stype[m], drop=True), '-', color=colours[m], linewidth=1)
                    ax2.plot_date(ds_dt.time_centered.where(ds_dt.scenario_type==stype[m], drop=True), ds_dt.where(ds_dt.scenario_type==stype[m], drop=True), '-', color=colours[m], linewidth=1)
                ax1.set_title('Shelf bottom salinity', fontsize=12)
                ax1.grid(linestyle='dotted')
                ax2.set_title('Time derivative', fontsize=12)
                ax2.axhline(0, color='black', linewidth=0.5)
                ax2.grid(linestyle='dotted')
                plt.suptitle(suite_string+', '+region_names[region], fontsize=14)
                ax1.set_ylabel('psu')
                ax2.set_ylabel('psu/century')
                # Manual legend
                handles = []
                for m in range(len(stype)):
                    handles.append(Line2D([0], [0], color=colours[m], label=labels[m], linestyle='-'))
                ax2.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5,-0.25), ncol=3)
                finished_plot(fig)

# Calculate the timescales for different stages in the tipping process; plot 1x3 timeseries (shelf salinity, cavity temperature, mass loss) for each trajectory and region (optional), and histograms of 4 different timescales. Also print information about how many trajectories tip and recover in what combinations.
def stage_timescales (base_dir='./', fig_dir=None, plot_traj=False):

    regions = ['ross', 'filchner_ronne']
    smooth = 10*months_per_year  # Not used for tipping/recovery date
    titles = ['Bottom salinity on continental shelf', 'Temperature in cavity', 'Basal mass loss']
    units = ['psu', deg_string+'C', 'Gt/y']
    stypes = [1, 0, -1]
    colours = ['Crimson', 'Grey', 'DodgerBlue']
    labels = ['ramp-up', 'stabilise', 'ramp-down']
    num_bins = 10

    # Inner function to calculate years between two dates; timedelta is returning negative weirdly (overflow with dates?)
    def years_between (date1, date2):
        years = date2.dt.year.item() - date1.dt.year.item()
        months = date2.dt.month.item() - date1.dt.month.item()
        return years + months/months_per_year

    # Inner function to find the date at which the given scenario type starts
    def stype_date (data, stype, return_index=False):
        if np.count_nonzero(data.scenario_type==stype)==0:
            t0 = None
            date = None
        else:
            t0 = np.argwhere(data.scenario_type.data==stype)[0][0]
            date = data.time_centered[t0]
        if return_index:
            return date, t0
        else:
            return date

    none_tip = 0
    ross_only_tip = 0
    ross_only_tip_no_recover = 0
    ross_only_tip_recovers = 0
    both_tip = 0
    both_tip_no_recover = 0
    both_tip_only_fris_recover = 0
    both_tip_both_recover = 0
    suite_lists = all_suite_trajectories()
    for suite_list in suite_lists:
        suite_string = '-'.join(suite_list)
        ross_tips = check_tip(suite=suite_string, region='ross')
        fris_tips = check_tip(suite=suite_string, region='filchner_ronne')      
        if not ross_tips and not fris_tips:
            none_tip += 1
        elif ross_tips and not fris_tips:
            ross_only_tip += 1
            ross_recovers = check_recover(suite=suite_string, region='ross')
            if not ross_recovers:
                ross_only_tip_no_recover += 1
            else:
                ross_only_tip_recovers += 1
        elif ross_tips and fris_tips:
            both_tip += 1
            ross_recovers = check_recover(suite=suite_string, region='ross')
            fris_recovers = check_recover(suite=suite_string, region='filchner_ronne')
            if not ross_recovers and not fris_recovers:
                both_tip_no_recover += 1
            elif fris_recovers and not ross_recovers:
                both_tip_only_fris_recover += 1
            elif ross_recovers and fris_recovers:
                both_tip_both_recover += 1
            else:
                raise Exception('Ross recovers but not FRIS for '+suite_string)
        else:
            raise Exception('FRIS tips but not Ross for '+suite_string)
    print('# trajectories: '+str(len(suite_lists)))
    print('Neither tips: '+str(none_tip))
    print('Only Ross tips: '+str(ross_only_tip))
    print('  does not recover: '+str(ross_only_tip_no_recover))
    print('  recovers: '+str(ross_only_tip_recovers))
    print('Both tip: '+str(both_tip))
    print('  neither recovers: '+str(both_tip_no_recover))
    print('  only FRIS recovers: '+str(both_tip_only_fris_recover))
    print('  both recover: '+str(both_tip_both_recover))

    stab_to_tip_all = []
    tip_to_melt_max_all = []
    ramp_down_to_recovery_all = []
    tip_to_recovery_all = []
    ramp_down_to_min_s_all = []
    # Loop over regions
    for region in regions:
        all_shelf_bwsalt, suite_strings = all_timeseries_trajectories(region+'_shelf_bwsalt', base_dir=base_dir)
        all_cavity_temp = all_timeseries_trajectories(region+'_cavity_temp', base_dir=base_dir)[0]
        all_massloss = all_timeseries_trajectories(region+'_massloss', base_dir=base_dir)[0]
        stab_to_tip = []
        tip_to_melt_max = []
        ramp_down_to_recovery = []
        tip_to_recovery = []
        ramp_down_to_min_s = []
        num_tip = 0
        tip_ramp_up = 0
        tip_stab = 0
        tip_ramp_down = 0
        num_recover = 0
        recover_stab = 0
        recover_ramp_down = 0
        no_recover_PI = 0
        # Loop over trajectories
        for suite_string, shelf_bwsalt, cavity_temp, massloss in zip(suite_strings, all_shelf_bwsalt, all_cavity_temp, all_massloss):
            print(suite_string+', '+region)
            tip_time = None
            stab_time = None
            melt_max_time = None
            ramp_down_time = None
            recover_time = None
            min_s_time = None
            shelf_bwsalt_smooth = moving_average(shelf_bwsalt, smooth)
            cavity_temp_smooth = moving_average(cavity_temp, smooth)
            massloss_smooth = moving_average(massloss, smooth)
            # Check for trajectories which tip
            tips, tip_time = check_tip(cavity_temp=cavity_temp, smoothed=False, return_date=True, base_dir=base_dir)
            if tips:
                num_tip += 1
                # Check for trajectories which tip during each phase
                stab_time = stype_date(cavity_temp, 0)
                ramp_down_time = stype_date(cavity_temp, -1)
                if ramp_down_time is not None and tip_time > ramp_down_time:
                    # Also includes restabilisation case
                    tip_ramp_down += 1
                elif stab_time is not None and tip_time > stab_time:
                    tip_stab += 1
                    # Save years between stabilisation and tipping
                    stab_to_tip.append(years_between(stab_time, tip_time))
                else:
                    tip_ramp_up += 1
                # Find time of max smoothed mass loss
                melt_max_time = massloss_smooth.time_centered[massloss_smooth.argmax()]
                # Save years between tipping and max mass loss
                tip_to_melt_max.append(years_between(tip_time, melt_max_time))
                # Check for trajectories which recover
                recovers, recover_time = check_recover(cavity_temp=cavity_temp, smoothed=False, return_date=True, base_dir=base_dir)
                if recovers:
                    num_recover += 1
                    # Save years between tipping and recovery
                    tip_to_recovery.append(years_between(tip_time, recover_time))
                    if ramp_down_time is not None and recover_time > ramp_down_time:
                        recover_ramp_down += 1
                        # Save years between ramp-down and recovery
                        ramp_down_to_recovery.append(years_between(ramp_down_time, recover_time))
                    elif stab_time is not None and recover_time > stab_time:
                        recover_stab += 1
                    else:
                        raise Exception(suite_string+' recovers during ramp-up')
                else:
                    # Check if final temp below PI
                    ramp_down_suite = suite_string.split('-')[-1]
                    date_PI = truncate_rampdown_PI(ramp_down_suite)
                    if date_PI is not None:
                        no_recover_PI += 1
            else:
                # Untipped
                recovers = False
                # Check for trajectories which have a ramp-down
                ramp_down_time, t0 = stype_date(cavity_temp, -1, return_index=True)
                if ramp_down_time is not None:
                    # Find time of min smoothed salinity after ramp down starts, unless it's in the last 10 years (not stabilised yet)
                    ts = shelf_bwsalt_smooth.isel(time_centered=slice(t0,None)).argmin() + t0
                    if ts < shelf_bwsalt_smooth.sizes['time_centered']-10*months_per_year:
                        min_s_time = shelf_bwsalt_smooth.time_centered[ts]
                        # Save years between ramp-down and min salinity
                        ramp_down_to_min_s.append(years_between(ramp_down_time, min_s_time))
            if plot_traj:
                # Plot
                fig = plt.figure(figsize=(5,8))
                gs = plt.GridSpec(3,1)
                gs.update(left=0.15, right=0.95, bottom=0.05, top=0.92, hspace=0.3)
                data_plot = [shelf_bwsalt_smooth, cavity_temp_smooth, massloss_smooth]
                time = data_plot[0].time_centered
                for n in range(3):
                    ax = plt.subplot(gs[n,0])
                    # Plot different phases in different colours
                    for m in range(len(stypes)):
                        index = shelf_bwsalt_smooth.scenario_type == stypes[m]
                        ax.plot_date(time.where(index, drop=True), data_plot[n].where(index, drop=True), '-', color=colours[m], linewidth=1)
                    ax.set_title(titles[n], fontsize=12)
                    ax.set_ylabel(units[n], fontsize=10)
                    ax.grid(linestyle='dotted')
                    ax.set_xlim([time[0], time[-1]])
                    ymax = ax.get_ylim()[-1]
                    if tips:
                        ax.axvline(tip_time.item(), color='black', linestyle='dashed', linewidth=1)
                        if n==0:
                            plt.text(tip_time.item(), ymax, ' tips', ha='left', va='top')
                    if recovers:
                        ax.axvline(recover_time.item(), color='black', linestyle='dashed', linewidth=1)
                        if n==0:
                            plt.text(recover_time.item(), ymax, ' recovers', ha='left', va='top')
                plt.suptitle(suite_string+', '+region_names[region], fontsize=14)
                if fig_dir is not None:
                    fig_name = fig_dir+'/'+suite_string+'_'+region+'.png'
                    finished_plot(fig, fig_name=fig_name)
                else:
                    # Purposefully wait until user closes window before proceeding to next plot, or you get 106 
                    plt.show()
                
        stab_to_tip_all.append(np.unique(stab_to_tip))
        tip_to_melt_max_all.append(np.unique(tip_to_melt_max))
        tip_to_recovery_all.append(np.unique(tip_to_recovery))
        ramp_down_to_recovery_all.append(np.unique(ramp_down_to_recovery))
        ramp_down_to_min_s_all.append(np.unique(ramp_down_to_min_s))
        print(region+' tips: '+str(num_tip))
        print(region+' tips during ramp-up: '+str(tip_ramp_up))
        print(region+' tips during stabilisation: '+str(tip_stab))
        print(region+' tips during ramp-down: '+str(tip_ramp_down))
        print(region+' recovers: '+str(num_recover))
        print(region+' recovers during stabilisation: '+str(recover_stab))
        print(region+' recovers during ramp-down: '+str(recover_ramp_down))
        print(region+' does not recover when temperatures cool below PI: '+str(no_recover_PI))

    # Plot histograms
    def plot_histogram (all_times, title, abbrev):
        tmax = 0
        for times in all_times:
            if len(times) > 0:
                tmax = max(tmax, np.amax(times))
        bins = np.linspace(0, tmax, num=num_bins)
        fig = plt.figure(figsize=(5,5))
        gs = plt.GridSpec(2,1)
        gs.update(left=0.1, right=0.99, bottom=0.1, top=0.85, hspace=0.4)
        for r in range(len(regions)):
            ax = plt.subplot(gs[r,0])
            ax.hist(all_times[r], bins=bins, color='DodgerBlue')
            ax.set_ylim([0,3.5])
            ax.set_title(region_names[regions[r]], fontsize=12)
            if r==0:
                ax.set_ylabel('# simulations', fontsize=10)
            if r==1:
                ax.set_xlabel('years', fontsize=10)
            if len(all_times[r]) > 0:
                plt.text(0.02, 0.98, 'Mean '+str(int(np.round(np.mean(all_times[r]))))+' years', ha='left', va='top', transform=ax.transAxes)
                plt.text(0.02, 0.88, 'Range '+str(int(round(np.amin(all_times[r]))))+'-'+str(int(round(np.amax(all_times[r]))))+' years', ha='left', va='top', transform=ax.transAxes)
                plt.text(0.02, 0.78, 'n='+str(np.size(all_times[r])), ha='left', va='top', transform=ax.transAxes)
        plt.suptitle(title, fontsize=14)
        if fig_dir is not None:
            fig_name = fig_dir+'/histogram_'+abbrev+'.png'
        else:
            fig_name = None
        finished_plot(fig, fig_name=fig_name)

    for all_times, title, abbrev in zip([stab_to_tip_all, tip_to_melt_max_all, tip_to_recovery_all, ramp_down_to_recovery_all, ramp_down_to_min_s_all], ['climate stabilisation and tipping', 'tipping and maximum basal mass loss', 'tipping and recovery', 'ramp-down and recovery', 'ramp-down and minimum salinity'], ['stab_to_tip', 'tip_to_melt_max', 'tip_to_recovery', 'ramp_down_to_recovery', 'ramp_down_to_min_s']):
        plot_histogram(all_times, 'Time between '+title, abbrev)

        
def fix_all_missing_months (base_dir='./'):

    file_names = ['timeseries.nc', 'timeseries_um.nc']
    for scenario in suites_by_scenario:
        for suite in suites_by_scenario[scenario]:
            print('Checking '+suite)
            for timeseries_file in file_names:                
                fix_missing_months(base_dir+'/'+suite+'/'+timeseries_file)


def check_all_nans (base_dir='./'):

    file_names = ['timeseries.nc', 'timeseries_um.nc']
    var_lists = [['all_massloss', 'all_bwtemp'], ['global_mean_sat']]

    for scenario in suites_by_scenario:
        for suite in suites_by_scenario[scenario]:
            print('Checking '+suite)
            for timeseries_file, var_names in zip(file_names, var_lists):
                check_nans(base_dir+'/'+suite+'/'+timeseries_file, var_names=var_names)


# Plot a series of maps showing snapshots of the given variable in each cavity for selected scenarios: initial state, tipping point, 100 years later, recovery point. Works for bwtemp, bwsalt, ismr. Also plot BISICLES ice speed in the grounded ice.
def map_snapshots (var_name='bwtemp', base_dir='./'):

    regions = ['ross', 'filchner_ronne']
    subfig = ['a) ', 'b) ']
    num_regions = len(regions)
    num_snapshots = 4
    suite_strings = ['cx209-cz376-da892', 'cx209-cz378-de943']
    year_titles = [['Initial', 'Tipping', '100 years later', 'Recovery'] for n in range(num_regions)]
    mask_pad = 5e4
    ice_dir = '/gws/nopw/j04/terrafirma/tm17544/TerraFIRMA_overshoots/raw_data/'
    ice_file_head = '/icesheet/bisicles_'
    ice_file_mid = 'c_'
    ice_file_tail = '0101_plot-AIS.hdf5'
    vmax_speed = 1e3
    sample_file = base_dir+'/time_averaged/piControl_grid-T.nc'  # Just to build region masks and shelf break contour
    labels = ['LABT', 'FT']
    lon0 = [-163, -37]
    lat0 = [-78, -77]
    lon1 = [-158, -26]
    lat1 = [-75, -77]
    lont = [-157, -22]
    latt = [-74, -77]
    depth0 = 1500

    ds_grid = xr.open_dataset(sample_file, decode_times=time_coder).squeeze()
    bathy0, draft0, ocean_mask0, ice_mask0 = calc_geometry(ds_grid)
    # Mask cavities out of bathymetry for shelf break contour
    bathy0 = bathy0.where(np.invert(ice_mask0))

    # Set variable title, NEMO name, units
    if var_name == 'bwtemp':
        var_title = 'Temperature at seafloor'
        units = deg_string+'C'
        nemo_var = 'tob'
        vmin = -2
        vmax = 3.5
        ctype = 'RdBu_r'
        colour_GL = 'yellow'
    elif var_name == 'bwsalt':
        var_title = 'Salinity at seafloor'
        units = 'psu'
        nemo_var = 'sob'
        vmin = 33
        vmax = 34.9
        ctype = 'viridis'
        colour_GL = 'white'
    elif var_name == 'ismr':
        var_title = 'Ice shelf melt rate'
        units = 'm/y'
        nemo_var = 'sowflisf'
        vmin = -2
        vmax = 20
        ctype = 'ismr'
        colour_GL = 'blue'
    else:
        raise Exception('Invalid variable '+var_name)

    # Construct suite titles describing each trajectory
    suite_titles = [trajectory_title(suites) for suites in suite_strings]

    # Inner function to figure out which suite in the trajectory the given year is, given a list of suites and corresponding starting years
    def find_suite (year, suite_list, start_years):
        # Loop backwards through suites
        for suite, year0 in zip(suite_list[::-1], start_years[::-1]):
            if year >= year0:
                return suite
        raise Exception('Year '+str(year)+' is too early')

    # Inner function to find the most retreated mask in the given direction from the list of masks, and set the plotting bound to be a bit larger than that
    def set_bound (masks, coord, option):
        if option == 'min':
            bound = coord.max()
        elif option == 'max':
            bound = coord.min()
        else:
            raise Exception('Invalid option '+option)
        for mask in masks:
            if option == 'min':
                bound = min(bound, coord.where(mask).min())
            elif option == 'max':
                bound = max(bound, coord.where(mask).max())
        if option == 'min':
            bound -= mask_pad
        elif option == 'max':
            bound += mask_pad
        return bound
    
    # Loop over regions and read all the things we need
    data_plot = []
    omask_GL = []
    imask_front = []
    x_bounds = []
    y_bounds = []
    ice_speed_plot = []
    for n in range(num_regions):
        # Find initial, tipping, and recovery years and suites
        tips, date_tip = check_tip(suite=suite_strings[n], region=regions[n], return_date=True, base_dir=base_dir)
        if not tips:
            raise Exception(suite_string+' does not tip')
        recovers, date_recover = check_recover(suite=suite_strings[n], region=regions[n], return_date=True, base_dir=base_dir)
        if not recovers:
            raise Exception(suite_strings[n]+' does not recover')
        # Find starting year of each suite in trajectory
        start_years = []
        suite_list = suite_strings[n].split('-')
        for suite in suite_list:
            file_names = []
            for f in os.listdir(base_dir+'/'+suite):
                if f.startswith('nemo_'+suite+'o_1m_') and f.endswith('-T.nc'):
                    file_names.append(f)
            file_names.sort()
            date_code = re.findall(r'\d{4}\d{2}\d{2}', file_names[0])
            start_years.append(int(date_code[0][:4]))
        # Now locate the years and suites we want to plot
        plot_years = [start_years[0], date_tip.dt.year.item(), date_tip.dt.year.item()+100, date_recover.dt.year.item()]
        plot_suites = [find_suite(year, suite_list, start_years) for year in plot_years]
        for year, suite, title in zip(plot_years, plot_suites, year_titles[n]):
            print(regions[n]+' '+title+': '+str(year-start_years[suite_list.index(suite)])+' years into '+suite)
        # Add the tipping and recovery years (since initial) to the titles
        for m in [1, 3]:
            year_titles[n][m] += ' (year '+str(plot_years[m]-plot_years[0])+')'
        # Now read the data for the four years, annually averaging
        data_region = []
        omask_region = []
        imask_region = []
        ice_speed_region = []
        for year, suite in zip(plot_years, plot_suites):
            # Build a list of the NEMO file patterns to read for this year
            files_to_read = []
            for f in os.listdir(base_dir+'/'+suite):
                if f.startswith('nemo_'+suite+'o_1m_'+str(year)) and f.endswith('-T.nc'):
                    date_codes = re.findall(r'\d{4}\d{2}\d{2}', f)
                    file_pattern = base_dir+'/'+suite+'/nemo_'+suite+'o_1m_'+date_codes[0]+'-'+date_codes[1]+'*-T.nc'
                    if file_pattern not in files_to_read:
                        files_to_read.append(file_pattern)
            files_to_read.sort()
            # Check there are 12 files
            if len(files_to_read) != months_per_year:
                raise Exception(str(len(files_to_read))+' files found for '+suite+', year '+str(year))
            # Annually average, carefully with masks
            data_accum = None
            ocean_mask = None
            ice_mask = None
            for file_pattern in files_to_read:
                ds = xr.open_mfdataset(file_pattern, decode_times=time_coder)
                ds.load()
                ds = ds.swap_dims({'time_counter':'time_centered'}).drop_vars(['time_counter'])
                data_tmp = ds[nemo_var].where(ds[nemo_var]!=0)
                ocean_mask_tmp = region_mask(regions[n], ds)[0]
                ice_mask_tmp = region_mask(regions[n], ds, option='cavity')[0]
                if data_accum is None:
                    data_accum = data_tmp
                    ocean_mask = ocean_mask_tmp
                    ice_mask = ice_mask_tmp
                    if n==0:
                        # Set up plotting coordinates
                        x, y = polar_stereo(ds['nav_lon'], ds['nav_lat'])
                        lon_edges = cfxr.bounds_to_vertices(ds['bounds_lon'], 'nvertex')
                        lat_edges = cfxr.bounds_to_vertices(ds['bounds_lat'], 'nvertex')
                        x_edges, y_edges = polar_stereo(lon_edges, lat_edges)
                        x_bg, y_bg = np.meshgrid(np.linspace(x_edges.min().item(), x_edges.max().item()), np.linspace(y_edges.min().item(), y_edges.max().item()))
                        mask_bg = np.ones(x_bg.shape)
                else:
                    data_accum = xr.concat([data_accum, data_tmp], dim='time_centered')
                    # Save most retreated GL
                    ocean_mask = xr.where(ocean_mask_tmp+ocean_mask, True, False)
                    ice_mask = xr.where(ice_mask_tmp+ice_mask, True, False)
                ds.close()
            data_mean = data_accum.mean(dim='time_centered')
            if var_name == 'ismr':
                data_mean = convert_ismr(data_mean)
            data_region.append(data_mean)
            omask_region.append(ocean_mask)
            imask_region.append(ice_mask)
            # Now read the BISICLES data - just one file (stamped with following year)
            ice_file = ice_dir + suite + ice_file_head + suite + ice_file_mid + str(year+1) + ice_file_tail
            ds_ice = read_bisicles(ice_file, ['xVel', 'yVel', 'activeBasalThicknessSource'], level=0, order=0)
            # Calculate speed
            speed = np.sqrt(ds_ice['xVel']**2 + ds_ice['yVel']**2)
            # Only consider regions with nonzero speed and zero basal melt (grounded ice)
            speed = speed.where(speed > 0)
            speed = speed.where(ds_ice['activeBasalThicknessSource']==0)
            ice_speed_region.append(speed)
        data_plot.append(data_region)
        ice_speed_plot.append(ice_speed_region)
        # Set plotting bounds on x (based on cavity) and y (based on cavity and shelf)
        xmin = set_bound(imask_region, x, 'min')
        xmax = set_bound(imask_region, x, 'max')
        ymin = set_bound(omask_region, y, 'min')
        ymax = set_bound(omask_region, y, 'max')
        # Add a bit extra to some sides of the mask to show more grounded ice
        if regions[n] == 'ross':
            xmin -= mask_pad*4
            ymax += mask_pad*3
            xmax += mask_pad*3
            ymin += mask_pad*5
        elif regions[n] == 'filchner_ronne':
            xmin += mask_pad*3
            ymin -= mask_pad*2
            xmax += mask_pad*5
        x_bounds.append([xmin, xmax])
        y_bounds.append([ymin, ymax])            
        # Prepare initial GL and ice front for contouring
        omask_ini = omask_region[0].astype('float')
        imask_ini = imask_region[0].astype('float')
        mask_sum = omask_ini + imask_ini  # 0 in grounded ice, 1 in open ocean, 2 in cavity
        # Mask open ocean for GL
        omask_GL.append(omask_ini.where(mask_sum!=1))
        # Mask grounded ice for ice front
        imask_front.append(imask_ini.where(mask_sum!=0))
        # Print actual min and max in this region to help choosing manual bounds
        plot_region = (x>=xmin)*(x<=xmax)*(y>=ymin)*(y<=ymax)
        vmin_real = np.amin([data.where(plot_region).min() for data in data_region])
        vmax_real = np.amax([data.where(plot_region).max() for data in data_region])
        print(regions[n]+' bounds from '+str(vmin_real)+' to '+str(vmax_real))
    # Get BISICLES coordinates relative to the centre of the domain
    x_ice = ice_speed_plot[0][0].coords['x']
    y_ice = ice_speed_plot[0][0].coords['y']
    x_c = 0.5*(x_ice[0] + x_ice[-1])
    y_c = 0.5*(y_ice[0] + y_ice[-1])
    x_ice = x_ice - x_c
    y_ice = y_ice - y_c

    # Plot
    cmap = set_colours(data_plot[0][0], ctype=ctype, vmin=vmin, vmax=vmax)[0]
    fig = plt.figure(figsize=(7,6))
    gs = plt.GridSpec(num_regions, num_snapshots)
    gs.update(left=0.02, right=0.98, bottom=0.23, top=0.89, wspace=0.1, hspace=0.5)
    for n in range(num_regions):
        for m in range(num_snapshots):
            ax = plt.subplot(gs[n,m])
            # Plot the data
            img = ax.pcolormesh(x_edges, y_edges, data_plot[n][m], cmap=cmap, vmin=vmin, vmax=vmax)
            # Plot the ice speed in white to black
            img_ice = ax.pcolormesh(x_ice, y_ice, ice_speed_plot[n][m].squeeze(), cmap='Greys', norm=cl.PowerNorm(0.5, vmax=vmax_speed))
            # Contour initial GL
            ax.contour(x, y, omask_GL[n], levels=[0.5], colors=(colour_GL), linewidths=0.5)
            # Contour ice front
            ax.contour(x, y, imask_front[n], levels=[0.5], colors=('white'), linewidths=0.5)
            # Contour shelf break
            ax.contour(x, y, bathy0, levels=[depth0], colors=('DarkMagenta'), linewidths=0.5)
            ax.set_xlim(x_bounds[n])
            ax.set_ylim(y_bounds[n])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(year_titles[n][m], fontsize=12)
            if m == 1:
                # Label troughs during tipping point
                x0, y0 = polar_stereo(lon0[n], lat0[n])
                x1, y1 = polar_stereo(lon1[n], lat1[n])
                xt, yt = polar_stereo(lont[n], latt[n])
                ax.plot([x0, x1], [y0, y1], color='black', linewidth=0.5)
                plt.text(xt, yt, labels[n], ha='center', va='center', fontsize=9)                    
        plt.text(0.5, 0.99-0.395*n, subfig[n]+region_names[regions[n]]+' Ice Shelf', ha='center', va='top', fontsize=14, transform=fig.transFigure)
        plt.text(0.5, 0.96-0.395*n, suite_titles[n], ha='center', va='top', fontsize=10, transform=fig.transFigure)
    cax1 = fig.add_axes([0.41, 0.17, 0.45, 0.02])
    cbar1 = plt.colorbar(img, cax=cax1, orientation='horizontal', extend='both')
    cbar1.ax.tick_params(labelsize=8)
    plt.text(0.635, 0.12, var_title+' ('+units+')', ha='center', va='center', fontsize=10, transform=fig.transFigure)
    cax2 = fig.add_axes([0.41, 0.07, 0.45, 0.02])
    cbar2 = plt.colorbar(img_ice, cax=cax2, orientation='horizontal', extend='max')
    cbar2.ax.tick_params(labelsize=8)
    plt.text(0.635, 0.02, 'Ice sheet speed (m/y)', ha='center', va='center', fontsize=10, transform=fig.transFigure)
    # Inset map showing regions
    ax2 = fig.add_axes([0.16, 0.01, 0.2, 0.2])
    ax2.axis('equal')
    # Shade open ocean in light blue, cavities in grey
    circumpolar_plot(ocean_mask0.where(ocean_mask0), ds_grid, ax=ax2, make_cbar=False, ctype='LightSkyBlue', lat_max=-66, shade_land=False)
    circumpolar_plot(ice_mask0.where(ice_mask0), ds_grid, ax=ax2, make_cbar=False, ctype='DarkGrey', lat_max=-66, shade_land=False)
    ax2.set_title('')
    ax2.axis('on')
    ax2.set_xticks([])
    ax2.set_yticks([])
    for n in range(num_regions):
        [xmin, xmax] = x_bounds[n]
        [ymin, ymax] = y_bounds[n]
        corners_x = [xmin, xmax, xmax, xmin, xmin]
        corners_y = [ymax, ymax, ymin, ymin, ymax]
        ax2.plot(corners_x, corners_y, color='black', linewidth=0.5)
        xpos = (3*xmin+xmax)/4
        if n == 0:
            ypos = (3*ymin+ymax)/4
            label = 'a'
        elif n == 1:
            ypos = (ymin+3*ymax)/4
            label = 'b'
        plt.text(xpos, ypos, label, ha='center', va='center', fontsize=10)
    finished_plot(fig, fig_name='figures/map_snapshots_'+var_name+'.png', dpi=300)


def plot_SLR_timeseries (base_dir='./', draft=False):

    vaf_dir = '/gws/nopw/j04/terrafirma/tm17544/TerraFIRMA_overshoots/processed_data/netcdf_files/'
    file_head = 'vaf_'
    file_tail = '_timeseries.nc'
    timeseries_file = 'timeseries.nc'
    pi_suite = 'cs568'  # Evolving ice
    baseline_suite = 'cx209'  # First member ramp-up
    regions = ['ross', 'filchner_ronne']
    prefixes = ['a) ', 'b) ']
    num_regions = len(regions)
    colours = ['DarkGrey', 'Crimson', 'DodgerBlue']
    labels = ['untipped', 'tipped', 'recovered']
    trend_years = 200  # Calculate drift in control over the last 200 years
    total_years = 1000  # Extend control to be 1000 years long (just needs to be longer than any other trajectory)
    obs_slr_ais = 7.4*10/28 # Otosaka et al. 2023: AIS contributed 7.4 mm over 1992-2020; convert to cm/century

    # Inner function to add the given DataArray to the axes with the given colour
    def add_line (data, ax, colour, year0):
        ax.plot(data.coords['time']-year0, data, '-', color=colour, linewidth=0.8)

    # Set up plot
    fig = plt.figure(figsize=(4.5,6))
    gs = plt.GridSpec(num_regions, 1)
    gs.update(left=0.13, right=0.87, bottom=0.15, top=0.9, hspace=0.3)
    for n in range(num_regions):
        ax = plt.subplot(gs[n,0])
        tipped_trends = []
        trend_weights = []
        if not draft:
            # Get baseline initial VAF from first member ramp-up (should be consistent between members as evolving ice has just been switched on)
            ds = xr.open_dataset(vaf_dir+'/'+file_head+baseline_suite+file_tail)
            vaf0 = ds[regions[n]+'_vaf'].isel(time=0)
            year0 = ds['time'].isel(time=0)
            ds.close()
        num_tip = 0
        num_recover = 0
        pi_slr = None
        # Loop over scenarios and check if file exists
        for scenario in suites_by_scenario:
            if 'static_ice' in scenario:
                continue
            for suite in suites_by_scenario[scenario]:
                if draft:
                    if suite == pi_suite:
                        continue
                    file_path = base_dir + '/' + suite + '/' + timeseries_file
                    ds = xr.open_dataset(file_path)
                    # Get annual values for year and draft
                    time_data = [t.dt.year.item() for t in ds['time_centered'][::12]]
                    time = xr.DataArray(time_data, coords={'time':time_data})
                    if suite == baseline_suite:
                        year0 = time[0]
                    data = xr.DataArray(ds[regions[n]+'_draft'][::12].data, coords={'time':time_data})
                    ds.close()
                else:
                    file_path = vaf_dir + '/' + file_head + suite + file_tail
                    if not os.path.isfile(file_path):
                        print('Warning: '+suite+' missing')
                        continue
                    # Read data
                    ds = xr.open_dataset(file_path)
                    # Offset of 1 year as per Tom's email (BISICLES output is snapshot at beginning of next year)
                    time = ds['time'] - 1
                    vaf = ds[regions[n]+'_vaf']
                    # Convert from VAF to sea level rise in cm
                    if suite == pi_suite:
                        # Different initial state to the rest
                        baseline = vaf.isel(time=0)
                    else:
                        baseline = vaf0
                    slr = (vaf-baseline)*vaf_to_gmslr*1e2
                    if suite == pi_suite:
                        # Calculate linear trend over last 200 years of control
                        pi_trend = linregress(time.data[-trend_years:], slr.data[-trend_years:])[0]
                        print(regions[n]+' drift at end of control simulation is '+str(pi_trend)+' cm/y')
                        # Extend the control suite with this linear trend                    
                        time_extend = np.arange(time[-1]+1, time[0]+total_years)
                        slr_extend = slr.data[-1] + pi_trend*(time_extend - time[-1].data)
                        slr_extend = xr.DataArray(data=slr_extend, dims=['time'], coords=dict(time=time_extend+1))
                        pi_slr = xr.concat([slr, slr_extend], dim='time')
                        continue
                    # Subtract drift
                    slr_trim, pi_slr_trim = align_timeseries(slr, pi_slr, time_coord='time')
                    if slr_trim.size != slr.size:
                        # Shouldn't have needed to trim the base timeseries
                        raise Exception('Problem with aligning timeseries')
                    data = slr_trim - pi_slr_trim
                tips, date_tip = check_tip(suite=suite, region=regions[n], return_date=True, base_dir=base_dir)
                if tips:
                    # Check if parent suite also tipped before branching point; if so, tipping pre-dates this suite
                    parent_suite = suites_branched[suite]
                    year_tip = date_tip.dt.year
                    if parent_suite is not None:
                        parent_tips, parent_date_tip = check_tip(suite=parent_suite, region=regions[n], return_date=True, base_dir=base_dir)
                        if parent_tips and parent_date_tip.dt.year < time[0]:
                            year_tip = time[0]
                    if year_tip <= time[-1]:
                        num_tip += 1
                    else:
                        print('Warning: '+suite+' does not extend to tipping date')
                        continue
                    # Select untipped section
                    untipped = data.where(time < year_tip, drop=True)
                    recovers, date_recover = check_recover(suite=suite, region=regions[n], return_date=True, base_dir=base_dir)
                    if recovers:
                        year_recover = date_recover.dt.year
                        if year_recover <= time[-1]:
                            num_recover += 1
                        else:
                            print('Warning: '+suite+' does not extend to recovery date')
                        tipped = data.where((time >= year_tip)*(time < year_recover), drop=True)
                        recovered = data.where(time >= year_recover, drop=True)
                        add_line(recovered, ax, colours[labels.index('recovered')], year0)
                    else:
                        tipped = data.where(time >= year_tip, drop=True)
                    add_line(tipped, ax, colours[labels.index('tipped')], year0)
                    # Calculate trend in tipped section
                    tipped_trends.append(linregress(tipped.coords['time'], tipped)[0])
                    # Weight by number of years
                    trend_weights.append(tipped.sizes['time'])
                else:
                    untipped = data
                    recovers = False
                add_line(untipped, ax, colours[labels.index('untipped')], year0)
        print(regions[n]+': '+str(num_tip)+' tipped, '+str(num_recover)+' recovered')
        # Calculate weighted average trend in cm/century
        avg_trend = np.average(tipped_trends, weights=trend_weights)*1e2
        print('Average tipped trend is '+str(avg_trend)+' cm/century: '+str(avg_trend/obs_slr_ais*100)+'% of observed AIS contribution rate')
        ax.grid(linestyle='dotted')
        ax.axhline(0, color='black', linewidth=0.5)
        if n == 1:
            ax.set_xlabel('Years')
        ax.set_xlim([0, None])
        if draft:
            ax.set_title(prefixes[n]+region_names[regions[n]], fontsize=12)
            plt.suptitle('Ice shelf draft', fontsize=14)
            ax.set_ylabel('m')
            fig_name = 'figures/draft_timeseries.png'
        else:
            ax.set_title(prefixes[n]+region_names[regions[n]]+' catchment', fontsize=12)
            plt.suptitle('Sea level contribution', fontsize=14)
            ax.set_ylabel('cm')
            fig_name = 'figures/SLR_timeseries.png'
    # Manual legend
    handles = []
    for m in range(len(colours)):
        handles.append(Line2D([0], [0], color=colours[m], label=labels[m], linestyle='-'))
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=3)
    finished_plot(fig, fig_name=fig_name, dpi=300)


# Count the total number of years in all overshoot simulations (excluding piControl and static ice).
def count_simulation_years (base_dir='./'):

    timeseries_file = 'timeseries.nc'
    smooth = 5*months_per_year+1

    years = 0
    for scenario in suites_by_scenario:
        if 'piControl' in scenario or 'static_ice' in scenario:
            continue
        for suite in suites_by_scenario[scenario]:
            file_path = base_dir+'/'+suite+'/'+timeseries_file
            ds = xr.open_dataset(file_path, decode_times=time_coder)
            if 'ramp_down' in scenario:
                # Check if simulation has cooled below PI
                date_end = truncate_rampdown_PI(suite)
                if date_end is not None:
                    t_truncate = np.argwhere(ds['time_centered'].data > date_end)[0][0]
                    print('Removing last '+str((ds.sizes['time_centered']-t_truncate+1)/months_per_year)+' years of '+suite+' which cools below PI')
                    ds = ds.isel(time_centered=slice(0,t_truncate))
            num_months = ds.sizes['time_centered']
            ds.close()
            years += num_months/months_per_year
    print('Total '+str(years)+' years')


# Identify NEMO files where the precomputed timeseries of ice draft jumps by an unreasonable amount. The pull from MASS did something weird and the output is sprinkled with weird files, most easily identifiable by huge changes in the ice draft which then change right back.
# At the same time I'm comparing the timeStamp global attribute, but this requires a slow pull of all the file headers from MASS. I'll make sure all the files identified here are in that list too.
def find_corrupted_files (base_dir='./', log=False):

    # Logfile to save a list of all the affected files
    log_file = base_dir+'/corrupted_files'
    log_file2 = base_dir+'/problem_events'
    # MASS command file
    #mass_file = base_dir+'/moo_replace_corrupted.sh'
    timeseries_file = 'timeseries.nc'
    threshold_ini = 50  # Allow larger jump when ice sheets are first switched on
    threshold_big = 10
    threshold_small = 1e-3  # approx. machine precision
    coupling_month = 1
    file_types = ['grid', 'isf']
    file_tails = ['grid-T.nc', 'isf-T.nc']
    ref_file = '/gws/nopw/j04/terrafirma/kaight/overshoots/cw988/nemo_cw988o_1m_21491201-21500101_grid-T.nc'  # compare draft to file that is definitely an example of the problem

    ds_ref = xr.open_dataset(ref_file, decode_times=time_coder)
    draft_ref = calc_geometry(ds_ref)[1]
    ds_ref.close()

    if log:
        # Open files
        f_log = open(log_file, 'w')
        f_log2 = open(log_file2, 'w')
        #f_mass = open(mass_file, 'w')
    num_months = 0
    num_problems = 0
    num_blocks_ref = 0
    num_blocks_other = 0

    # Construct the filenames corresponding to the given suite and date, and add them to the logfile.
    def add_files (suite, date, add=True):
        year0 = date.dt.year.item()
        month0 = date.dt.month.item()
        year1, month1 = add_months(year0, month0, 1)
        for file_type in file_types:
            file_path = 'nemo_'+suite+'o_1m_'+str(year0)+str(month0).zfill(2)+'01-'+str(year1)+str(month1).zfill(2)+'01_'+file_type+'-T.nc'
            if file_type == 'grid':
                file_path0 = file_path
                if not add:
                    break
            if not os.path.isfile(suite+'/'+file_path):
                raise Exception('Missing file '+file_path)
            if log and add:
                print('Problem with '+file_path)
                f_log.write(file_path+'\n')
                #f_mass.write('rm '+suite+'/'+file_path+'\n')
                #f_mass.write('moo filter '+file_type+'T.moo_ncks_opts :crum/u-'+suite+'/onm.nc.file/'+file_path+' '+suite+'/\n')
        return file_path0

    timestamps = []
    # Loop over all suites
    for scenario in suites_by_scenario:
        if 'static_ice' in scenario:
            continue
        for suite in suites_by_scenario[scenario]:
            num_blocks = 0
            print('Processing '+suite+' ('+scenario+')')
            # Read precomputed timeseries of mean ice draft
            ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file, decode_times=time_coder)
            draft = ds['all_draft']
            time = ds['time_centered']
            ds.close()
            num_months += np.size(time)
            problem = False
            last_good = draft[0]
            # Loop over time
            for t in range(1, np.size(time)):
                # Choose threshold to use
                if ('piControl' in scenario or 'ramp_up' in scenario) and time[t].dt.year == time[0].dt.year+1 and time[t].dt.month == coupling_month:
                    # Ice draft could have a moderate jump in first coupling timestep after
                    threshold = threshold_ini
                elif time[t].dt.month == coupling_month or problem:
                    # Ice draft could have changed a small amount from annual coupling
                    threshold = threshold_big
                else:
                    # Ice draft should not have changed beyond machine precision
                    threshold = threshold_small
                if not problem:
                    # Not currently in a block of problem files
                    # Compare to previous month
                    if np.abs(draft[t]-draft[t-1]) > threshold:
                        problem = True
                        num_problems += 1
                        num_blocks += 1
                        # Add to log file
                        nemo_file = add_files(suite, time[t])
                        print('\nBlock starts at '+str(time[t].dt.year.item())+'-'+str(time[t].dt.month.item()))
                        # Check if draft matches the reference file
                        try:
                            ds = xr.open_dataset(base_dir+'/'+suite+'/'+nemo_file, decode_times=time_coder)
                            draft_2D = calc_geometry(ds)[1]
                            is_ref = (draft_2D==draft_ref).all()
                            if is_ref:
                                print('Draft matches reference geometry')
                                num_blocks_ref += 1
                                if log:
                                    f_log2.write(nemo_file+'\n')
                            else:
                                print('Draft is something different')
                                num_blocks_other +=1
                            # Check timestamp
                            date = datetime.datetime.strptime(ds.attrs['timeStamp'], "%Y-%b-%d %H:%M:%S UTC")
                            ds.close()
                            print('In real time, '+str(date.year)+'-'+str(date.month)+'-'+str(date.day))
                            if is_ref:
                                timestamps.append(date)
                            # Now check timestamp in file before
                            last_file = add_files(suite, time[t-1], add=False)
                            ds_before = xr.open_dataset(base_dir+'/'+suite+'/'+last_file, decode_times=time_coder)
                            date_before = datetime.datetime.strptime(ds_before.attrs['timeStamp'], "%Y-%b-%d %H:%M:%S UTC")
                            ds_before.close()
                            delta = date - date_before
                            print('Timestamp jump of '+str(delta.days)+' days')
                        except(FileNotFoundError):
                            print('Warning: '+nemo_file+' missing')
                        # Check date relative to tipping
                        # Check tipping and recovery in this suite
                        for region in ['ross', 'filchner_ronne']:
                            tips, date_tip = check_tip(suite=suite, region=region, return_date=True)
                            recovers, date_recover = check_recover(suite=suite, region=region, return_date=True)
                            if tips:
                                if time[t] < date_tip:
                                    print('Block happens before '+region+' tips')
                                else:
                                    if recovers:
                                        if time[t] < date_recover:
                                            print('Block happens after '+region+' tips, before recovers')
                                        else:
                                            print('Block happens after '+region+' recovers')
                                    else:
                                        print('Block happens after '+region+' tips (does not recover)')
                            else:
                                print(region+' does not tip')                                
                    else:
                        # Update the last good ice draft
                        last_good = draft[t]
                else:
                    # In a block of problem files
                    # Compare to last good ice draft
                    if np.abs(draft[t]-last_good) > threshold:
                        num_problems += 1
                        # Add to log file
                        nemo_file = add_files(suite, time[t])
                    else:
                        # Out of the problem block
                        problem = False
                        last_good = draft[t]
                        print('Block ends at '+str(time[t].dt.year.item())+'-'+str(time[t].dt.month.item()))
            if num_blocks > 0:
                print('\n'+str(num_blocks)+' blocks in '+suite+'\n')
    if log:
        f_log.close()
        f_log2.close()
        #f_mass.close()
    print(str(num_problems)+' of '+str(num_months)+' months affected ('+str(num_problems/num_months*100)+'%)')
    print(str(num_blocks_ref)+' blocks of reference geometry, '+str(num_blocks_other)+' blocks of other geometry')

    # Plot distribution of timestamps
    fig, ax = plt.subplots(figsize=(8,3))
    for ts in timestamps:
        ax.plot_date(ts, np.random.rand(), '*')
    ax.set_yticks([])
    ax.set_title('Problems in real time')
    ax.grid(linestyle='dotted')
    plt.tight_layout()
    fig.savefig('figures/problem_timestamps.png')


# For each corrupted file name listed in the given file (created above), re-calculate that time index of data for the given timeseries file.
def overwrite_corrupted_timeseries (in_file='corrupted_files', timeseries_file='timeseries.nc', base_dir='./'):

    # Read file into list
    f = open(in_file, 'r')
    file_paths = f.read().splitlines()
    f.close()
    
    # Build list of affected suites
    suites = []
    for fname in file_paths:
        i0 = fname.index('_')+1
        i1 = fname.index('o_1m_')
        suite = fname[i0:i1]
        if suite not in suites:
            suites.append(suite)

    # Loop over suites
    for suite in suites2:
        # Build list of affected file patterns
        file_patterns = []
        for fname in file_paths:
            if suite not in fname:
                continue
            file_head = fname[:fname.rfind('_')]
            file_pattern = file_head + '*.nc'
            if file_pattern not in file_patterns:
                file_patterns.append(file_pattern)
        # Read timeseries file and get list of variables to precompute
        file_path_ts = base_dir+'/'+suite+'/'+timeseries_file
        ds_ts = xr.open_dataset(file_path_ts, decode_times=time_coder)
        var_names = [var for var in ds_ts]
        for file_pattern in file_patterns:
            print('Processing '+file_pattern)
            ds_nemo = xr.open_mfdataset(suite+'/'+file_pattern, decode_times=time_coder)
            ds_nemo.load()
            # Remove halo
            ds_nemo = ds_nemo.isel(x=slice(1,-1))
            # Find the matching time index in the timeseries file
            t0 = np.argwhere(ds_ts['time_centered'].data==ds_nemo['time_centered'][0].data)[0][0]
            for var in var_names:
                # Recalculate variable
                try:
                    data, ds_nemo = calc_timeseries(var, ds_nemo)
                except(KeyError):
                    print('Warning: missing data')
                    # Flag with NaN
                    data = ds_nemo['time_counter'].where(False)
                data = data.swap_dims({'time_counter':'time_centered'})
                # Overwrite in timeseries dataset
                ds_ts[var][t0] = data.squeeze()
        # Overwrite timeseries file
        overwrite_file(ds_ts, file_path_ts)


# Check the NetCDF global attribute timeStamp for every NEMO file in every suite, compared to the nearly-empty header files pulled more recently from MASS (only containing time variable). Make a list of the ones where the header timeStamp is newer: these have been overwritten on MASS and should be re-pulled again. Also write a moose command script as in find_corrupted_files.
def find_updated_files (suite, base_dir='./'):

    header_dir = base_dir+'/headers/'
    log_file = base_dir+'/'+suite+'/updated_files'
    mass_file = base_dir+'/'+suite+'/moo_replace_updated.sh'

    f_log = open(log_file, 'w')
    f_mass = open(mass_file, 'w')
    num_files = 0
    num_updated = 0

    # Inner function to open the given NetCDF file and return the timeStamp attribute as a datetime object
    def parse_timestamp (file_path):
        id = nc.Dataset(file_path, 'r')
        date = datetime.datetime.strptime(id.timeStamp, "%Y-%b-%d %H:%M:%S UTC")
        id.close()
        return date

    # Add the given filename to the logfile and MASS file
    def add_files (fname):
        print('Problem with '+fname)
        file_type = fname[fname.rfind('_')+1:fname.index('-T.nc')]
        f_log.write(fname+'\n')
        f_mass.write('rm '+suite+'/'+fname+'\n')
        f_mass.write('moo filter '+file_type+'T.moo_ncks_opts :crum/u-'+suite+'/onm.nc.file/'+fname+' '+suite+'/\n')

    for f in os.listdir(base_dir+'/'+suite):
        if not f.startswith('nemo_'+suite):
            continue
        num_files += 1
        if num_files % 100 == 0:
            print('...done '+str(num_files)+' files')
        if not os.path.isfile(header_dir+'/'+suite+'/'+f):
            print('Warning: missing header for '+f)
            continue
        date_orig = parse_timestamp(base_dir+'/'+suite+'/'+f)
        date_header = parse_timestamp(header_dir+'/'+suite+'/'+f)
        if date_header < date_orig:
            print('Warning: MASS timestamp is older than local timestamp for '+f)
        elif date_header > date_orig:
            # MASS timestamp is newer - will have to re-pull this one
            add_files(f)
            num_updated += 1
    f_log.close()
    f_mass.close()
    print(str(num_updated)+' of '+str(num_files)+' files affected ('+str(num_updated/num_files*100)+'%)')


# Helper function to read the list of problem events, and make a dictionary of suites and dates affected.
def find_problem_suites (base_dir='./', in_file='problem_events'):

    f = open(in_file, 'r')
    file_paths = f.read().splitlines()
    f.close()   
    problems_by_suite = {}
    for file_path in file_paths:
        # Extract suite name
        suite = file_path[len('nemo_'):file_path.index('o_1m_')]
        # Get date object from relevant file
        ds = xr.open_dataset(suite+'/'+file_path, decode_times=time_coder)
        date = ds['time_centered'].squeeze().item()
        ds.close()
        if suite in problems_by_suite:
            problems_by_suite[suite] = problems_by_suite[suite] + [date]
        else:
            problems_by_suite[suite] = [date]
    return problems_by_suite


# Helper function to find all the trajectories affected by the geometry bug.
# Returns a dictionary of affected trajectories (suite-strings), each corresponding to a list of dates when the problems occur.
def find_problem_trajectories (base_dir='./', in_file='problem_events'):

    problems_by_suite = find_problem_suites(base_dir=base_dir, in_file=in_file)

    # Loop through all trajectories
    all_traj = all_suite_trajectories()
    problems_by_traj = {}
    for traj in all_traj:
        suite_string = '-'.join(traj)
        # Check if any suites are affected
        if any([suite in problems_by_suite for suite in traj]):
            # Check if any problems actually happen during the trajectory (instead of, eg, in a perpetual ramp-up after the stabilisation here has already branched off)
            # Find start and end date of each suite segment in trajectory
            start_dates, end_dates = find_stages_start_end(traj, base_dir=base_dir)
            problems_in_traj = []
            for suite, suite_start, suite_end in zip(traj, start_dates, end_dates):
                if suite in problems_by_suite:
                    for date in problems_by_suite[suite]:
                        if date >= suite_start and date <= suite_end:
                            # Add to dictionary
                            if suite_string in problems_by_traj:
                                problems_by_traj[suite_string] = problems_by_traj[suite_string] + [date]
                            else:
                                problems_by_traj[suite_string] = [date]
    return problems_by_traj                


# Find all the trajectories affected by the geometry bug, and plot the affected ones showing the dates of each problem relative to tipping/recovery and suite transitions.
def plot_problem_trajectories (base_dir='./', in_file='problem_events'):

    regions = ['ross', 'filchner_ronne']
    colours = ['DarkRed', 'DarkSlateBlue', 'Crimson', 'DodgerBlue']  # Ross tips, Ross recovers, FRIS tips, FRIS recovers
    stage_colours = ['Crimson', 'white', 'DodgerBlue']
    
    problems_by_traj = find_problem_trajectories(base_dir=base_dir, in_file=in_file)
    for suite_string in problems_by_traj:
        suite_list = suite_string.split('-')
        # Check Ross and FRIS tipping/recovery dates
        plot_dates = []
        for region in regions:
            plot_dates.append(check_tip(suite=suite_string, region=region, return_date=True)[1])
            plot_dates.append(check_recover(suite=suite_string, region=region, return_date=True)[1])
        start_dates, end_dates = find_stages_start_end(suite_string.split('-'), base_dir=base_dir)
        # Plot
        fig, ax = plt.subplots(figsize=(8,2))
        # Star for each problem
        for date in problems_by_traj[suite_string]:
            ax.plot_date(date, 1, '*', color='black')
        # Dashed lines for tipping and recovery
        for date, colour in zip(plot_dates, colours):
            if date is not None:
                ax.axvline(date.item(), color=colour, linestyle='dashed')
        # Shade colours and labels for each suite
        for t in range(len(start_dates)):
            ax.axvspan(start_dates[t], end_dates[t], alpha=0.1, color=stage_colours[t])
            plt.text(start_dates[t], 1.5, suite_list[t], ha='left', va='top')
        ax.set_xlim([start_dates[0], end_dates[-1]])
        ax.set_ylim([0.5, 1.5])
        ax.set_yticks([])
        ax.set_title(trajectory_title(suite_list))
        plt.tight_layout()
        fig.show()


# Compare global warming at the time of tipping/recovery between trajectories affected by the geometry bug, and trajectories unaffected.
def bug_impact_tipping_recovery (base_dir='./', in_file='problem_events'):

    regions = ['ross', 'filchner_ronne']
    var_names = ['global_warming'] #, 'shelf_bwsalt']
    timeseries_file = 'timeseries.nc'
    timeseries_file_um = 'timeseries_um.nc'
    smooth = 5*months_per_year
    pi_suite = 'cs495'
    p0 = 0.05

    all_traj = all_suite_trajectories()
    problems_by_traj = find_problem_trajectories(base_dir=base_dir, in_file=in_file)
    baseline_temp = pi_baseline_temp(pi_suite=pi_suite, base_dir=base_dir)

    # Loop over regions
    for region in regions:
        # Loop over the variables we want to test
        for var in var_names:
            print('\nTesting impact on '+var+' for '+region)                
            at_tip_problem = []
            at_tip_noproblem = []
            at_recovery_problem = []
            at_recovery_noproblem = []
            # Check every trajectory
            for suite_list in all_traj:
                # Check if it tips
                suite_string = '-'.join(suite_list)
                cavity_temp = build_timeseries_trajectory(suite_list, region+'_cavity_temp', base_dir=base_dir, timeseries_file=timeseries_file)
                cavity_temp = moving_average(cavity_temp, smooth)
                tips, date_tip, t_tip = check_tip(cavity_temp=cavity_temp, smoothed=True, return_date=True, return_t=True)
                if tips:
                    # Now build the data
                    if var == 'global_warming':
                        data = build_timeseries_trajectory(suite_list, 'global_mean_sat', base_dir=base_dir, timeseries_file=timeseries_file_um, offset=-baseline_temp)
                    else:
                        data = build_timeseries_trajectory(suite_list, region+'_'+var, base_dir=base_dir, timeseries_file=timeseries_file)
                    data = moving_average(data, smooth)
                    cavity_temp, data = align_timeseries(cavity_temp, data)
                    tip_data = data.isel(time_centered=t_tip)
                    # Check if there is a problem before the tipping point
                    if suite_string in problems_by_traj:
                        problem = any([date <= date_tip for date in problems_by_traj[suite_string]])
                    else:
                        problem = False
                    if problem:
                        at_tip_problem.append(tip_data)
                    else:
                        at_tip_noproblem.append(tip_data)
                    # Check if it recovers
                    recovers, date_recover, t_recover = check_recover(cavity_temp=cavity_temp, smoothed=True, return_date=True, return_t=True)
                    if recovers:
                        recover_data = data.isel(time_centered=t_recover)
                        # Check if there is a problem before recovery
                        if suite_string in problems_by_traj:
                            problem = any([date <= date_recover for date in problems_by_traj[suite_string]])
                        else:
                            problem = False
                        if problem:
                            at_recovery_problem.append(recover_data)
                        else:
                            at_recovery_noproblem.append(recover_data)
            # Now check statistics
            for problem, noproblem, suptitle in zip([at_tip_problem, at_recovery_problem], [at_tip_noproblem, at_recovery_noproblem], ['tipping', 'recovery']):            
                # Make sure we're not double-counting branched trajectories
                problem = np.unique(problem)
                noproblem = np.unique(noproblem)
                if np.size(problem) == 0:
                    print('No problems before '+suptitle)
                elif np.size(noproblem) == 0:
                    print('Entirely problems before '+suptitle)
                else:
                    full = np.concatenate((problem, noproblem), axis=0)
                    for data, title in zip([full, noproblem, problem], ['all', 'no problems', 'problems']):
                        print('Mean at '+suptitle+' ('+title+'): '+str(np.mean(data))+', n='+str(np.size(data)))
                    # Check if problems within range of no-problems
                    in_range = 0
                    for x in problem:
                        if x >= np.amin(noproblem) and x <= np.amax(noproblem):
                            in_range += 1
                    print(str(in_range)+' of '+str(np.size(problem))+' problems within range of no-problems')
                    # Check if significant difference between all combinations of the 3 samples
                    samples = [problem, noproblem, full]
                    names = ['problems', 'no-problems', 'full']
                    num_samples = len(samples)
                    for i in range(num_samples):
                        for j in range(i+1, num_samples):
                            p_val = ttest_ind(samples[i], samples[j], equal_var=False)[1]
                            if p_val < p0:
                                print('Significant difference of '+str(np.mean(samples[i])-np.mean(samples[j]))+' between '+names[i]+' and '+names[j]+', p='+str(p_val))
                            else:
                                print('No significant difference between '+names[i]+' and '+names[j])


# Plot the differences between a simulation with the geometry bug, and a re-run version without the bug, to see the recovery timescale.
def bug_recovery_timescale (base_dir='./'):

    suite_old = 'dc130'
    suite_new = 'dn966'
    rampup_suite = 'cx209'
    timeseries_file = 'timeseries.nc'
    var_names = ['massloss', 'cavity_temp', 'cavity_salt', 'bwtemp', 'bwsalt', 'shelf_bwsalt']
    regions = ['all', 'ross', 'filchner_ronne']
    smooth_detrend = 30*months_per_year
    smooth = months_per_year  # Seasonality
    std_cutoff = 1

    def read_var (suite, var):
        ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file, decode_times=time_coder)
        data = ds[var]
        ds.close()
        return data

    for var in var_names:
        for region in regions:
            if var == 'shelf_bwsalt' and region == 'all':
                continue
            var_full = region+'_'+var
            # Get std of detrended ramp-up
            data_rampup = read_var(rampup_suite, var_full)
            # Take 30-year rolling mean to detrend
            rampup_smooth = moving_average(data_rampup, smooth_detrend)
            # Separately take 1-year mean to remove seasonal cycle
            data_rampup = moving_average(data_rampup, smooth)
            # Now get difference on correct time indices
            data_rampup, rampup_smooth = align_timeseries(data_rampup, rampup_smooth)
            rampup_detrend = data_rampup - rampup_smooth
            # Now get std
            std = rampup_detrend.std()
            
            # Read data from each simulation and take 1-year mean
            data_old = read_var(suite_old, var_full)
            data_new = read_var(suite_new, var_full)
            # Save first non-problem year: year after new simulation starts
            year0 = data_new['time_centered'][0].dt.year.item()+1
            data_old, data_new = align_timeseries(data_old, data_new)
            data_old = moving_average(data_old, smooth)
            data_new = moving_average(data_new, smooth)
            # Get difference
            data_diff = data_old - data_new
            # Get time axis: years since problem ended
            years = time_in_years(data_diff, year0=year0)
            # Plot
            fig, ax = plt.subplots()
            ax.plot(years, data_diff, '-', color='blue')
            ax.set_xlabel('Years since problem ended')
            ax.set_ylabel(data_old.units)
            ax.grid(linestyle='dotted')
            [y0, y1] = ax.get_ylim()
            ax2 = ax.twinx()
            ax2.plot(years, data_diff/std, '-', color='blue')
            ax2.set_ylim([y0/std, y1/std])
            ax2.set_ylabel('std')
            ax2.axhline(std_cutoff, color='black', linestyle='dashed', linewidth=1)
            ax2.axhline(-std_cutoff, color='black', linestyle='dashed', linewidth=1)
            ax.set_title(var_full)
            fig.show()


# Given the recovery timescale determined above (9 years for everything to recover within 1 std), mask out every problem event and the following 9 years in the ocean timeseries. Keep the unmasked versions, but rename them.
def mask_problems (base_dir='./', in_file='problem_events'):

    timeseries_file = 'timeseries.nc'
    timeseries_copy = 'timeseries_unmasked.nc'
    mask_years = 9

    problems_by_suite = find_problem_suites(base_dir=base_dir, in_file=in_file)
    # Loop over affected suites
    for suite in problems_by_suite:
        print('Processing '+suite)
        file_path = base_dir+'/'+suite+'/'+timeseries_file
        # Make a copy of timeseries
        shutil.copyfile(file_path, base_dir+'/'+suite+'/'+timeseries_copy)
        # Read timeseries and get time axis
        ds = xr.open_dataset(file_path, decode_times=time_coder)
        time = ds['time_centered']
        # Loop over problems in this suite
        masked_months = 0
        for date in problems_by_suite[suite]:
            # Find the time index where the problem starts
            t0 = np.argwhere(time.data==date)[0][0]
            year0 = time[t0].dt.year
            month0 = time[t0].dt.month
            # Mask out that time index, and the rest of that year
            mask = (time.dt.year == year0)*(time.dt.month >= month0)
            masked_months += np.count_nonzero(mask)
            ds = ds.where(~mask)
            # Now mask out the following 9 full years
            mask = (time.dt.year > year0)*(time.dt.year <= year0+mask_years)
            masked_months += np.count_nonzero(mask)
            ds = ds.where(~mask)
        print('Masked '+str(masked_months)+' months ('+str(masked_months/ds.sizes['time_centered']*100)+'% of timeseries)')
        # Overwrite timeseries
        overwrite_file(ds, file_path)


# Confirm that the Ross always tips first, and FRIS always recovers first
def check_tipping_recovery_order (base_dir='./'):

    smooth = 5*months_per_year

    all_traj = all_suite_trajectories()
    for suite_list in all_traj:
        suite_string = '-'.join(suite_list)
        fris_temp = moving_average(build_timeseries_trajectory(suite_list, 'filchner_ronne_cavity_temp', base_dir=base_dir), smooth)
        ross_temp = moving_average(build_timeseries_trajectory(suite_list, 'ross_cavity_temp', base_dir=base_dir), smooth)
        fris_tip, fris_t = check_tip(cavity_temp=fris_temp, smoothed=True, return_t=True)
        if fris_tip:
            ross_tip, ross_t = check_tip(cavity_temp=ross_temp, smoothed=True, return_t=True)
            if not ross_tip:
                print(suite_string+': FRIS tips but not Ross')
            elif fris_t < ross_t:
                print(suite_string+': FRIS tips before Ross')
            fris_recover, fris_t = check_recover(cavity_temp=fris_temp, smoothed=True, return_t=True)
            ross_recover, ross_t = check_recover(cavity_temp=ross_temp, smoothed=True, return_t=True)
            if ross_recover and not fris_recover:
                print(suite_string+': Ross recovers but not FRIS')
            elif not ross_recover:
                continue
            elif ross_t < fris_t:
                print(suite_string+': Ross recovers before FRIS')


# Check if any ramp-downs tip.
def check_rampdown_tip (base_dir='./'):

    regions = ['ross', 'filchner_ronne']
    trajectories = all_suite_trajectories()
    smooth = 5*months_per_year

    for region in regions:
        for suite_list in trajectories:
            suite_string = '-'.join(suite_list)
            cavity_temp = moving_average(build_timeseries_trajectory(suite_list, region+'_cavity_temp', base_dir=base_dir), smooth)
            tips, tip_t = check_tip(cavity_temp=cavity_temp, smoothed=True, return_t=True)
            if tips:
                stype = cavity_temp.scenario_type[tip_t]
                if stype == -1:
                    print(suite_string+': '+region+' tips during ramp-down')


# Calculate linear regression of bottom salinity against global temperature for every point in the Ross and FRIS shelf regions, and every ramp-up ensemble member.
def spatial_regression_bwsalt_gw (base_dir='./', out_file='bwsalt_warming_regression.nc'):

    num_ens = len(suites_by_scenario['ramp_up'])
    regions = ['ross', 'filchner_ronne']
    sample_file = base_dir+'/time_averaged/piControl_grid-T.nc'
    timeseries_file = 'timeseries.nc'
    timeseries_file_um = 'timeseries_um.nc'
    smooth_years = 5
    smooth = smooth_years*months_per_year

    # Make combined mask
    ds = xr.open_dataset(sample_file, decode_times=time_coder)
    masks = [region_mask(region, ds, option='shelf')[0] for region in regions]
    mask = (masks[0] + masks[1]).squeeze()
    ds.close()

    # Set up array to save slopes
    slopes = (0*mask).expand_dims(dim={'ens':np.arange(num_ens)})

    # Loop over ensemble members
    for n, suite in zip(range(num_ens), suites_by_scenario['ramp_up']):
        
        print('Processing '+suite)
        # Read global temperature
        warming = global_warming(suite, base_dir=base_dir)
        # Find date at which Ross tips
        tips, tip_date = check_tip(suite=suite, region='ross', return_date=True)
        tip_year = tip_date.dt.year
        print('Ross tips in '+str(tip_year.item()))
        
        # Make list of files to read; stop 5 years after tipping
        sim_dir = base_dir+'/'+suite+'/'
        file_head = 'nemo_'+suite+'o_1m_'
        file_tail = '_grid-T.nc'
        nemo_files = []
        for f in os.listdir(sim_dir):
            if f.startswith(file_head) and f.endswith(file_tail):
                year = int(f[len(file_head):len(file_head)+4])
                if year < tip_year + smooth_years:
                    nemo_files.append(sim_dir+f)
        # Make sure in chronological order
        nemo_files.sort()
        
        print('Reading data')
        # Read bottom salinity from every file
        for file_path in nemo_files:
            ds = xr.open_dataset(file_path, decode_times=time_coder)
            # Only keep the indices inside the mask to save memory
            bwsalt = ds['sob'].where(mask).swap_dims({'time_counter':'time_centered'})
            bwsalt = bwsalt.where(bwsalt.notnull(), drop=True)
            if file_path == nemo_files[0]:
                bwsalt_all = bwsalt
            else:
                # Concatenate to previous data
                bwsalt_all = xr.concat([bwsalt_all, bwsalt], dim='time_centered')
            ds.close()
            
        # Align with warming timeseries - should work even though bwsalt_all has extra dimensions
        warming, bwsalt_all = align_timeseries(warming, bwsalt_all)
        warming_smooth = moving_average(warming, smooth)

        print('Calculating regressions')
        # Loop over x and y dimensions (restricted by dropping points outside mask)
        for y0 in bwsalt_all.coords['y']:
            for x0 in bwsalt_all.coords['x']:
                # Extract timeseries at this point
                bwsalt_ts = bwsalt_all.isel(y=y0, x=x0)
                if all(bwsalt_ts.isnull()):
                    # Not inside mask
                    continue                
                # Smooth
                bwsalt_smooth = moving_average(bwsalt_ts, smooth)
                # Calculate regression of bwsalt in response to warming
                slope0 = linregress(warming_smooth, bwsalt_smooth)[0]
                # Save to master slope array
                slopes = xr.where((slopes.ens==n)*(slopes['nav_lat']==bwsalt_all['nav_lat'].isel(x=x0, y=y0))*(slopes['nav_lon']==bwsalt_all['nav_lon'].isel(x=x0, y=y0)), slope0, slopes)

    # Save to file
    ds = xr.Dataset({'slope':slopes})
    print('Writing '+out_file)
    ds.to_netcdf(out_file)
    ds.close()


# List for Jing of if/when all the stabilisation runs tip.
def stabilisation_tipping_list (base_dir='./', out_file='stabilisation_tipping_times'):

    regions = ['ross', 'filchner_ronne']
    f = open(out_file, 'w')
    timeseries_file = 'timeseries.nc'

    for scenario in suites_by_scenario:
        if 'stabilise' not in scenario:
            continue
        for suite in suites_by_scenario[scenario]:
            f.write(suite+' ('+scenario+')\n')
            # Get starting date
            ds = xr.open_dataset(suite+'/'+timeseries_file, decode_times=time_coder)
            date0 = ds['time_centered'][0]
            ds.close()
            f.write('Starts in '+str(date0.dt.year.item())+'\n')
            for region in regions:
                region_name = region_names[region]
                # Check if parent ramp-up tips before starting date
                parent_suite = suites_branched[suite]
                parent_tips, parent_tip_date = check_tip(suite=parent_suite, region=region, return_date=True)
                if parent_tips and parent_tip_date < date0:
                    f.write(region_name+' tips before stabilisation\n')
                else:
                    # Check if stabilisation run tips
                    tips, tip_date = check_tip(suite=suite, region=region, return_date=True)
                    if tips:
                        f.write(region_name+' tips in '+str(tip_date.dt.year.item())+'\n')
                    else:
                        f.write(region_name+' does not tip\n')
            f.write('\n')
    f.close()


def temp_correction_uncertainty (base_dir='./', bias_file='bwsalt_bias.nc', slope_file='bwsalt_warming_regression.nc'):

    sample_file = base_dir+'/time_averaged/piControl_grid-T.nc'
    regions = ['ross', 'filchner_ronne']
    cutoff = 3.5
    p0 = 0.05

    # Make combined mask
    ds_grid = xr.open_dataset(sample_file, decode_times=time_coder)
    masks = [region_mask(region, ds_grid, option='shelf')[0] for region in regions]
    mask = (masks[0] + masks[1]).squeeze()

    # Read bias and slopes
    ds = xr.open_dataset(bias_file, decode_times=time_coder)
    bias = ds['bwsalt_bias'].squeeze().where(mask)
    ds.close()
    ds = xr.open_dataset(slope_file, decode_times=time_coder)
    slopes = ds['slope'].where(mask)
    ds.close()
    # Get ensemble mean slope and mask out regions where not significant
    slope = slopes.mean(dim='ens')
    t_val, p_val = ttest_1samp(slopes.data, 0, axis=0)
    p_val = xr.DataArray(p_val, coords=slope.coords)
    slope = slope.where(p_val < p0)
    # Mask out regions where it doesn't freshen - CDW on continental slope, different water mass.
    bias = bias.where(slope<0)
    slope = slope.where(slope<0)
    # Calculate temperature correction at every point
    temp_correction_2D = bias/slope

    # Calculate 10-90% range
    pc10 = temp_correction_2D.quantile(0.1)
    pc90 = temp_correction_2D.quantile(0.9)
    print('10-90% range '+str(pc10.item())+' - '+str(pc90.item())+' degC')

    # 3-panel plot for supplementary
    fig = plt.figure(figsize=(8,6))
    gs = plt.GridSpec(2,2)
    gs.update(left=0.08, right=0.95, bottom=0.08, top=0.9, hspace=0.2, wspace=0.1)
    # Map of slope
    ax = plt.subplot(gs[0,0])
    ax.axis('equal')
    circumpolar_plot(slope, ds_grid, ax=ax, title='a) Ensemble mean slope (psu/'+deg_string+'C)', titlesize=14, lat_max=-66, ctype='plusminus')
    # Map of temperature correction
    ax = plt.subplot(gs[0,1])
    ax.axis('equal')
    circumpolar_plot(temp_correction_2D, ds_grid, ax=ax, title='b) Temperature adjustment ('+deg_string+'C)', titlesize=14, lat_max=-66, ctype='plusminus', vmin=-cutoff, vmax=cutoff, cbar_kwags={'extend':'both'})
    # Histogram showing distribution of points
    ax = plt.subplot(gs[1,:])    
    temp_correction_2D = temp_correction_2D.where((temp_correction_2D >= -cutoff)*(temp_correction_2D < cutoff))
    [n, bins, patches] = ax.hist(temp_correction_2D.data.ravel(), bins=50)
    ax.axvline(temp_correction, linestyle='dashed', color='black')
    ax.axvline(pc10, color='black')
    ax.axvline(pc90, color='black')
    ax.grid(linestyle='dotted')
    ax.set_title('c) Distribution of temperature adjustment values', fontsize=14)
    ax.set_xlabel(deg_string+'C', fontsize=12)
    ax.set_ylabel('# grid cells', fontsize=12)
    finished_plot(fig, fig_name='figures/temp_correction_uncertainty.png', dpi=300)

    # Print central value of top 10 bins sorted by frequency; from this can determine two peaks of distribution
    print('Top 10 bins:')
    bin_centres = 0.5*(bins[:-1] + bins[1:])
    for i in range(10):
        print(bin_centres[n.argmax()])
        n[n.argmax()] = 0


# A couple of the re-run problem simulations ran for long enough to replace the old ones. Merge into the old simulations where they branched just before the first problem.
def merge_rerun_suite (suite_old, suite_new, base_dir='./', vaf=False):

    if vaf:
        timeseries_files = ['_timeseries.nc']
        vaf_dir = '/gws/nopw/j04/terrafirma/tm17544/TerraFIRMA_overshoots/processed_data/netcdf_files/'
        time_coord = 'time'
    else:
        timeseries_files = ['timeseries.nc', 'timeseries_um.nc']
        time_coord = 'time_centered'

    for ts_file in timeseries_files:
        if vaf:
            file_old = vaf_dir+'vaf_'+suite_old+ts_file
            file_new = vaf_dir+'vaf_'+suite_new+ts_file
        else:
            file_old = base_dir+'/'+suite_old+'/'+ts_file
            file_new = base_dir+'/'+suite_new+'/'+ts_file
        ds_old = xr.open_dataset(file_old, decode_times=time_coder)
        ds_new = xr.open_dataset(file_new, decode_times=time_coder)
        date_branch = ds_new[time_coord][0]
        if vaf:
            print('Merging at '+str(date_branch.item()))
        else:
            print('Merging at '+str(date_branch.dt.year.item())+'-'+str(date_branch.dt.month.item()))
        t_merge = np.argwhere(ds_old[time_coord].data == date_branch.data)[0][0]
        ds_merge = xr.concat([ds_old.isel({time_coord:slice(0,t_merge)}), ds_new], dim=time_coord)
        ds_old.close()
        ds_new.close()
        print('Overwriting '+file_new)
        overwrite_file(ds_merge, file_new)


# Compare the global mean SAT at the time of Ross recovery between a particularly badly affected problem suite, and a re-run suite.
def problem_effect_recovery (base_dir='./'):

    suite_old = 'dc123'
    suite_new = 'do135'
    region = 'ross'
    smooth = 5*months_per_year
    timeseries_file = 'timeseries.nc'
    
    for suite in [suite_old, suite_new]:
        print(suite)
        warming = moving_average(global_warming(suite, base_dir=base_dir), smooth)
        ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file, decode_times=time_coder)
        cavity_temp = moving_average(ds[region+'_cavity_temp'], smooth)
        cavity_temp, warming = align_timeseries(cavity_temp, warming)
        recovers, date_recovers, t_recovers = check_recover(cavity_temp=cavity_temp, smoothed=True, return_date=True, return_t=True, base_dir=base_dir)
        recover_warming = warming.isel(time_centered=t_recovers)
        print('Recovers in '+str(date_recovers.dt.year.item())+'-'+str(date_recovers.dt.month.item())+'; global warming '+str(recover_warming.item())+' C')
        ds.close()


# Check for drift in piControl mass loss for the two caviites.
def check_pi_drift (base_dir='./'):

    regions = ['ross', 'filchner_ronne']
    timeseries_file = 'timeseries.nc'
    smooth = 5*months_per_year
    p0 = 0.05

    # Identify the piControl suites
    for scenario in suites_by_scenario:
        if 'piControl' in scenario:
            for suite in suites_by_scenario[scenario]:
                print(suite+' ('+scenario+')')
                ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file, decode_times=time_coder)
                for region in regions:
                    print(region+', '+suite)
                    massloss = moving_average(ds[region+'_massloss'], smooth)
                    time_years = [(t.dt.year.item() - massloss.coords['time_centered'][0].dt.year.item()) + (t.dt.month.item() - 1)/months_per_year + 0.5 for t in massloss.coords['time_centered']]
                    slope, intercept, r_value, p_value, std_err = linregress(time_years, massloss)
                    if p_value < p0:
                        # Convert to % of baseline / century
                        baseline = massloss.mean(dim='time_centered').item()
                        trend_percent = slope/baseline*1e4
                        print('Significant trend of '+str(slope)+' Gt/y/y, or '+str(trend_percent)+' %/century')
                    else:
                        print('No significant trend')
                ds.close()


def case_study_timeseries (base_dir='./'):

    regions = ['ross', 'filchner_ronne']
    colours = ['DarkGreen', 'Purple']
    suite_strings = ['cx209-cz376-da892', 'cx209-cz378-de943']
    suite_lists = [suite_string.split('-') for suite_string in suite_strings]
    region_titles = ['Ross', 'Filchner-Ronne']
    suite_titles = [trajectory_title(suites) for suites in suite_strings]
    num_regions = len(regions)
    var_names = ['global_mean_sat', 'cavity_temp', 'massloss']
    var_titles = ['a) Simulated global warming', 'b) Ocean temperature in ice shelf cavities', 'c) Melting beneath ice shelves']
    units = 2*[deg_string+'C'] + ['Gt/y']
    num_var = len(var_names)
    timeseries_file = ['timeseries_um.nc'] + 2*['timeseries.nc']
    baseline_temp = pi_baseline_temp(base_dir=base_dir)
    smooth = 5*months_per_year

    # Read data
    data_plot = []
    for v in range(num_var):
        data_var = []
        for n in range(num_regions):
            if var_names[v] == 'global_mean_sat':
                data = build_timeseries_trajectory(suite_lists[n], var_names[v], base_dir=base_dir, timeseries_file=timeseries_file[v], offset=-baseline_temp)
            else:
                data = build_timeseries_trajectory(suite_lists[n], regions[n]+'_'+var_names[v], base_dir=base_dir, timeseries_file=timeseries_file[v])
            # Smooth
            data = moving_average(data, smooth)
            data_var.append(data)
        data_plot.append(data_var)
    date_end = [truncate_rampdown_PI(suite_list[-1]) for suite_list in suite_lists]

    # Plot
    fig = plt.figure(figsize=(5,7))
    gs = plt.GridSpec(num_var,1)
    gs.update(left=0.14, right=0.93, bottom=0.2, top=0.95, hspace=0.5)
    for v in range(num_var):
        ax = plt.subplot(gs[v,0])
        last_year = None
        for n in range(num_regions):
            data = data_plot[v][n]
            if date_end[n] is not None:
                t_truncate = np.argwhere((data.coords['time_centered'] > date_end[n]).data)[0][0]
                data = data.isel(time_centered=slice(0,t_truncate))
            # Get time axis in years since beginning
            years = time_in_years(data)
            if last_year is None:
                last_year = years[-1]
            else:
                last_year = max(last_year, years[-1])
            i0 = suite_titles[n].index('for')
            ax.plot(years, data, '-', color=colours[n], label=region_titles[n]+': '+suite_titles[n][:i0]+'\n'+suite_titles[n][i0:], linewidth=1.5)
        ax.grid(linestyle='dotted')
        ax.set_xlim([0, last_year])
        if var_names[v] == 'global_mean_sat':
            plt.text(100, 2.8, 'ramp-up', rotation=35, ha='center', va='center')
            plt.text(325, 5.15, 'zero-\nemission', ha='center', va='center')
            plt.text(550, 3.4, 'ramp-down', rotation=-35, ha='center', va='center')
        elif var_names[v] == 'cavity_temp':
            ax.axhline(tipping_threshold, color='black', linestyle='dashed', linewidth=1)
            ax.set_ylim([-3.2, None])
            plt.text(10, -2.5, 'untipped', ha='left', va='top')
            plt.text(170, -1.7, 'tipped', ha='right', va='bottom')
            plt.text(650, -2.2, 'recovered', ha='right', va='top')
        ax.set_ylabel(units[v])
        ax.set_title(var_titles[v], fontsize=13)
        plt.text(0.94, -0.15, 'years', fontsize=10, transform=ax.transAxes)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5,-1))
    finished_plot(fig, fig_name='figures/case_study_timeseries.png', dpi=300)


def plot_ross_special_cases (base_dir='./'):

    region = 'ross'
    var_name = 'cavity_temp'
    var_title = 'Temperature in Ross Ice Shelf cavity ('+deg_string+'C)'
    suite_strings = ['cx209-cy838', 'cw988-cz859', 'cx209-cz375']
    suite_title_prefix = ['a) ', 'b) ', 'c) ']
    suite_titles = [trajectory_title(suites) for suites in suite_strings]
    num_traj = len(suite_strings)
    stypes = [1, 0] #, -1]
    colours = ['Crimson', 'Grey'] #, 'DodgerBlue']
    labels = ['ramp-up', 'zero-emission'] #, 'ramp-down']
    smooth = 5*months_per_year
    min_temp = -2.1
    max_temp = 2

    # Read cavity temperature from each trajectory
    data = [moving_average(build_timeseries_trajectory(suite_string.split('-'), region+'_'+var_name, base_dir=base_dir), smooth) for suite_string in suite_strings]
    # Get tipping and recovery dates, if relevant
    tip_dates = [check_tip(suite=suite_string, region=region, return_date=True)[1] for suite_string in suite_strings]
    recovery_dates = [check_recover(suite=suite_string, region=region, return_date=True)[1] for suite_string in suite_strings]
    
    fig = plt.figure(figsize=(10, 3.25))
    gs = plt.GridSpec(1,3)
    gs.update(left=0.04, right=0.96, bottom=0.21, top=0.8, wspace=0.15)
    for n in range(num_traj):
        ax = plt.subplot(gs[0,n])
        # Get time axis in years since beginning
        years, year0 = time_in_years(data[n], return_year0=True)
        for m in range(len(stypes)):
            index = data[n].scenario_type == stypes[m]
            ax.plot(years, data[n].where(index), '-', color=colours[m], linewidth=1.5, label=labels[m])
        if n == 2:
            plt.text(0.98, -0.115, 'years', fontsize=12, transform=ax.transAxes)
        ax.grid(linestyle='dotted')
        ax.tick_params(axis='both', labelsize=12)
        ax.set_ylim([min_temp, max_temp])
        ax.axhline(tipping_threshold, color='black', linestyle='dashed', linewidth=1)
        if tip_dates[n] is not None:
            tip_year = tip_dates[n].dt.year.item() - year0
            ax.axvline(tip_year, color='black', linestyle='dashed', linewidth=1)
            plt.text(tip_year, max_temp, ' tips', ha='left', va='top', rotation=-90, fontsize=12)
        if recovery_dates[n] is not None:
            recovery_year = recovery_dates[n].dt.year.item() - year0
            ax.axvline(recovery_year, color='black', linestyle='dashed', linewidth=1)
            plt.text(recovery_year, max_temp, ' recovers', ha='left', va='top', rotation=-90, fontsize=12)
        i0 = suite_titles[n].index(' to ')
        i1 = suite_titles[n].index(', zero emissions')
        title = suite_title_prefix[n] + suite_titles[n][i0+4:i1] + ' GWL'
        ax.set_title(title, fontsize=14)
        if n == 1:
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.37), ncol=2, fontsize=13)
    plt.suptitle(var_title, fontsize=16)
    finished_plot(fig, fig_name='figures/ross_special_cases.png', dpi=300)
        
    

    
            
        

    
                    
                
    
    

    

    
            
    

    
    
        
                

    
        
    
    
    
        
    

    
    


    
    
    

    

    
            
            

    



        
                
                  
            
    
       

    
        



                    

    
