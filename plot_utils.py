import matplotlib.colors as cl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ..constants import deg_string

# Helper functions for colourmaps

def truncate_colourmap (cmap, minval=0.0, maxval=1.0, n=-1):
    
    # From https://stackoverflow.com/questions/40929467/how-to-use-and-plot-only-a-part-of-a-colorbar-in-matplotlib    
    if n== -1:
        n = cmap.N
    new_cmap = cl.LinearSegmentedColormap.from_list('trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plusminus_cmap (vmin, vmax, val0, reverse=False):

    if val0 is None:
        val0 = 0

    # Truncate the RdBu_r colourmap as needed, so that val0 is white and no unnecessary colours are shown.    
    if abs(vmin-val0) > vmax-val0:
        min_colour = 0
        max_colour = 0.5*(1 - (vmax-val0)/(vmin-val0))
    else:
        min_colour = 0.5*(1 + (vmin-val0)/(vmax-val0))
        max_colour = 1
    if reverse:
        cmap = plt.get_cmap('RdBu')
    else:
        cmap = plt.get_cmap('RdBu_r')
    return truncate_colourmap(cmap, min_colour, max_colour)


# Create a linear segmented colourmap from the given values and colours. Helper function for ismr_cmap.
def special_cmap (cmap_vals, cmap_colours, vmin, vmax, name):

    vmin_tmp = min(vmin, np.amin(cmap_vals))
    vmax_tmp = max(vmax, np.amax(cmap_vals))

    cmap_vals_norm = (cmap_vals-vmin_tmp)/(vmax_tmp-vmin_tmp)
    cmap_vals_norm[-1] = 1
    cmap_list = []
    for i in range(cmap_vals.size):
        cmap_list.append((cmap_vals_norm[i], cmap_colours[i]))
    cmap = cl.LinearSegmentedColormap.from_list(name, cmap_list)

    if vmin > vmin_tmp or vmax < vmax_tmp:
        min_colour = (vmin - vmin_tmp)/(vmax_tmp - vmin_tmp)
        max_colour = (vmax - vmin_tmp)/(vmax_tmp - vmin_tmp)
        cmap = truncate_colourmap(cmap, min_colour, max_colour)

    return cmap


def ismr_cmap (vmin, vmax, change_points=None):

    # First define the colours we'll use
    ismr_blue = (0.26, 0.45, 0.86)
    ismr_white = (1, 1, 1)
    ismr_yellow = (1, 0.9, 0.4)
    ismr_orange = (0.99, 0.59, 0.18)
    ismr_red = (0.5, 0.0, 0.08)
    ismr_pink = (0.96, 0.17, 0.89)

    if change_points is None:            
        # Set change points to yield a linear transition between colours
        change_points = 0.25*vmax*np.arange(1,3+1)
    if len(change_points) != 3:
        print('Error (ismr_cmap): wrong size for change_points list')
        sys.exit()

    if vmin < 0:
        # There is refreezing here; include blue for elements < 0
        cmap_vals = np.concatenate(([vmin], [0], change_points, [vmax]))
        cmap_colours = [ismr_blue, ismr_white, ismr_yellow, ismr_orange, ismr_red, ismr_pink]
        return special_cmap(cmap_vals, cmap_colours, vmin, vmax, 'ismr')
    else:
        # No refreezing; start at 0
        cmap_vals = np.concatenate(([0], change_points, [vmax]))
        cmap_colours = [ismr_white, ismr_yellow, ismr_orange, ismr_red, ismr_pink]
        return special_cmap(cmap_vals, cmap_colours, vmin, vmax, 'ismr')


# Set up colourmaps of type ctype, which can be any existing matplotlib colourmap or any of the following custom ones:
# 'plusminus': a red/blue colour map where 0 is white
# 'plusminus_r': same, but with red and blue reversed
# 'ismr': a special colour map for ice shelf melting/refreezing, with negative values in blue, 0 in white, and positive values moving from yellow to orange to red to pink.
# Keyword arguments:
# vmin, vmax: min and max values to enforce for the colourmap. They may be modified eg to make sure ismr includes 0. If you don't specify them, they will be determined based on the entire array of data.
# change_points: only matters for 'ismr'. List of size 3 containing values where the colourmap should hit the colours yellow, orange, and red. It should not include the minimum value, 0, or the maximum value. Setting these parameters allows for a nonlinear transition between colours, and enhanced visibility of the melt rate. If it is not defined, the change points will be determined linearly.
def set_colours (data, ctype='viridis', vmin=None, vmax=None, change_points=None):

    # Work out bounds
    if isinstance(data, xr.DataArray):
        data_min = data.min()
        data_max = data.max()
    elif isinstance(data, np.array):
        data_min = np.amin(data)
        data_max = np.amax(data)
    if vmin is None:
        vmin = data_min
    if vmax is None:
        vmax = data_max
    vmin = float(vmin)  # just in case user input an integer
    vmax = float(vmax)

    if ctype == 'plusminus':
        return plusminus_cmap(vmin, vmax, 0), vmin, vmax
    elif ctype == 'plusminus_r':
        return plusminus_cmap(vmin, vmax, 0, reverse=True), vmin, vmax
    elif ctype == 'ismr':
        return ismr_cmap(vmin, vmax, change_points=change_points), vmin, vmax
    else:
        # First assume it's the name of a preset colourmap
        try:
            return plt.get_cmap(ctype), vmin, vmax
        except(ValueError):
            return cl.ListedColormap([ctype]), vmin, vmax





# Round the given number to the given maximum number of decimals, with no unnecessary trailing zeros.
def round_to_decimals (x, max_decimals):
    for d in range(max_decimals+1):
        if round(x,d) == x or d == max_decimals:
            fmt = '{0:.'+str(d)+'f}'
            label = fmt.format(round(x,d))
            break
    return label


# Format the latitude or longitude x as a string, rounded to max_decimals (with no unnecessary trailing zeros), and expressed as a compass direction eg 30 <degrees> W instead of -30.
def latlon_label (x, suff_minus, suff_plus, max_decimals):

    # Figure out if it's south/west or north/east
    if x < 0:
        x = -x
        suff = suff_minus
    else:
        suff = suff_plus

    # Round to the correct number of decimals, with no unnecessary trailing 0s
    label = round_to_decimals(x, max_decimals)
    return label + suff


def lon_label (x, max_decimals=0):
    return latlon_label(x, deg_string+'W', deg_string+'E', max_decimals)


def lat_label (x, max_decimals=0):
    return latlon_label(x, deg_string+'S', deg_string+'N', max_decimals)


# Now, give the lon and lat axes nice labels.
def latlon_axes (ax, max_decimals=0):

    lon_ticks = ax.get_xticks()
    lat_ticks = ax.get_yticks()
    lon_labels = []
    for x in lon_ticks:
        lon_labels.append(lon_label(x, max_decimals=max_decimals))
    ax.set_xticklabels(lon_labels)
    lat_labels = []
    for y in lat_ticks:
        lat_labels.append(lat_label(y, max_decimals=max_decimals))
    ax.set_yticklabels(lat_labels)
            
            

    
