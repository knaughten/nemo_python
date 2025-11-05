import numpy as np
 
# Acceleration due to gravity (m/s^2)
gravity = 9.81
# Density of freshwater (kg/m^3)
rho_fw = 1e3
# Density of ice
rho_ice = 917.
# Seconds per hour
sec_per_hour = 60*60.
# Seconds per day
sec_per_day = 24*sec_per_hour
# Seconds per year
sec_per_year = 365.25*sec_per_day
# Months per year
months_per_year = 12
# Celsius to Kelvins intercept
temp_C2K = 273.15
# Constants for vapor pressure calculation over water (Buck, 1981):
vap_pres_c1 = 611.21; vap_pres_c3 = 17.502; vap_pres_c4=32.19;
# Gas constant for dry air
Rdry = 287.0597
# Gas constant for water vapor
Rvap = 461.5250
# Degrees to radians conversion factor
deg2rad = np.pi/180.0
rad2deg = 1./deg2rad
# Radius of Earth (m)
rEarth = 6.371e6
# Rotation rate of Earth
omegaEarth = 7.2921e-5 #rad/s
# Conversion from ice VAF change to global mean sea level contribution (m^-2)
vaf_to_gmslr = -0.918e-12/361.8
# Degrees formatted nicely in a string
deg_string = r'$^{\circ}$'
# 10^-3 formatted nicely in a string (for absolute salinity)
gkg_string = r'10$^{-3}$'

# Dictionary of bounds on different regions
# lon_min, lon_max, lat_min, lat_max
region_bounds = {
    'bear_ridge_S': [-110.35, -110.35, -74.35, -73.8791],
    'bear_ridge_N': [-112.05, -109.05, -73.8791, -72.7314],
    'pine_island_bay': [-104, -100.5, -75.2, -74.2],
    'dotson_bay': [-114, -110.5, -74.3, -73.5],
    'amundsen_west_shelf_break': [-115, -112, -72, -71],
    'weddell_gyre': [-60., 30., -90., -50.],
    'ross_gyre': [150, -140, -90, -60],
    'filchner_trough': [-45, -30, -79, -75],
    'ronne_depression': [-70, -55, -76, -73],
    'LAB_trough': [-166, -157, -80, -75],
    'drygalski_trough': [163, 168.5, -80, -71],
}
# Isobaths restricting some regions: shallow bound, then deep bound
region_bathy_bounds = {
    'bear_ridge_N': [None, 350],
    'filchner_trough': [600, 1200],
    'ronne_depression': [525, None],
    'LAB_trough': [525, 800],
    'drygalski_trough': [500, 1200],
}
# Latitude to bound continental shelf
shelf_lat = -58
# Depth of continental shelf
shelf_depth = 2000
# Lon and lat of a point which is definitely on the given continental shelf (so we can isolate seamounts disconnected from this)
shelf_point0 = [-51.5, -74.5]
shelf_point1 = [171, -75]
# Drake Passage transect
drake_passage_lon0 = -68  # Corresponds to i=219 in eORCA1
drake_passage_lat_bounds = [-67.3, -52.6]  # Corresponds to j=79:109 in eORCA1
# Lon and lat of a point which is definitely in the Ross Gyre
ross_gyre_point0 = [-160, -70]

# Dictionary of x,y, coordinates in AntArc configuration delineating roughly the Weddell Sea and Ross sea convection regions 
# (should generalize to lat lons at some point)
weddell_convect = {'x':slice(930,1150), 'y':slice(295,360)}
ross_convect    = {'x':slice(450,580),  'y':slice(220,300)}

# Dictionary of lons and lats describing key waypoints for transect paths.
transect_amundsen = {
    'shelf_west':[[-115.5, -115, -116, -120.6], [-74.6, -73.5, -72.5, -70.78]],
    'shelf_mid' :[[-112.6,-114.6], [-75.2,-70.48]],
    'shelf_east':[[-100.6,-104,-106,-106,-104,-102.6,-101.8], [-75.16,-74.68,-74.2,-73.3,-72.4,-71.2,-69.1]],
    'shelf_edge':[[-137,-135,-132,-131,-127.6,-127,-120.6,-118.6,-114,-102,-97,-93], [-73.7,-73.6,-73.3,-73,-72.7,-72.3,-72.5,-71.7,-71.2,-70.9,-70,-70.3]],
    'Getz_left':[[-129, -131], [-75, -72.5]],
    'Getz_right':[[-122, -123.5], [-75.2, -71.8]],
    'Dotson':[[-112, -115], [-75.4, -70.8]],
    'PI_trough':[[-106, -109], [-75.4, -70.5]],
} # transect locations chosen to cross most-sampled regions

# Dictionary of lon-lat points bounding given region. Will be used to "cut" the continental shelf mask (build_shelf_mask in utils.py) either north-south or east-west depending on the value of region_edges_flag. The first point and its connected N/S (or E/W) neighbours will be included in the mask, but not the second. The direction of travel is west to east around the coastline.
region_edges = {
    'abbot'             : [[-103.2, -71.8] , [-83, -72]],
    'amery'             : [[66.5, -67.5]   , [79.5, -68]],
    'amundsen_sea'      : [[-157.5, -76.5] , [-102.75, -72.5]],
    'bellingshausen_sea': [[-102.75, -72.5], [-57.25, -62]],
    'cosgrove'          : [[-104.24, -73.846], [-102.91, -73.2]],
    'dotson_crosson'    : [[-114.7, -73.8] , [-107.5, -75.3]],
    'dotson_front'      : [[-112.5, -74.4] , [-110.5, -73.85]], # just shelf
    'pine_island'       : [[-102.6, -75.1] , [-101.5, -74.2]],
    'pine_island_bay'   : [[-104.0, -74.8] , [-103, -74.2]], # just shelf
    'dotson_cosgrove'   : [[-114.7, -73.8] , [-102.91, -73.2]],
    'east_antarctica'   : [[-26, -75.5]    , [169.5 , -71]], # includes Amery
    'filchner_ronne'    : [[-57, -71.5]    , [-26.0 , -75.5]],
    'getz'              : [[-135, -74.5]   , [-114.7, -73.8]],
    'larsen'            : [[-57.25, -62]   , [-57, -71.5]],
    'ross'              : [[169.5, -71]    , [-157.5, -76.5]],
    'thwaites'          : [[-107.5, -75.3] , [-103.6, -74.5]],
    'west_antarctica'   : [[-157.5, -76.5] , [-57.25, -62]], # Amundsen and Bellingshausen
}
region_edges_flag = {
    'abbot'             : ['NS', 'NS'],
    'amery'             : ['NS', 'NS'],
    'amundsen_sea'      : ['NS', 'NS'],
    'bellingshausen_sea': ['NS', 'NS'],
    'cosgrove'          : ['EW', 'EW'],
    'dotson_crosson'    : ['NS', 'NS'],
    'dotson_front'      : ['EW', 'EW'],
    'dotson_cosgrove'   : ['NS', 'EW'],
    'east_antarctica'   : ['NS', 'NS'],
    'filchner_ronne'    : ['EW', 'NS'],
    'getz'              : ['NS', 'NS'],
    'larsen'            : ['NS', 'EW'],
    'pine_island'       : ['NS', 'EW'], 
    'pine_island_bay'   : ['NS', 'EW'],
    'ross'              : ['NS', 'NS'],
    'thwaites'          : ['NS', 'EW'],
    'west_antarctica'   : ['NS', 'NS'],
}
# Dictionary of lon-lat points which are definitely in the given region. The region is then defined by connectivity to that point (eg selecting specific ice shelf cavities in single_cavity_mask in utils.py).
region_points = {
    'abbot': [-95, -73],
    'brunt': [-20, -75],
    'pine_island': [-101, -75],
}
# Names of each region
region_names = {
    'all'               : 'Antarctic',
    'abbot'             : 'Abbot Ice Shelf',
    'amery'             : 'Amery',
    'amundsen_sea'      : 'Amundsen Sea',
    'amundsen_west_shelf_break': 'Western Amundsen Sea shelf break',
    'bellingshausen_sea': 'Bellingshausen Sea',
    'brunt'             : 'Brunt and Riiser-Larsen Ice Shelves',
    'cosgrove'          : 'Cosgrove Ice Shelf',
    'dotson_crosson'    : 'Dotson-Crosson Ice Shelf',
    'dotson_front'      : 'front of Dotson',
    'east_antarctica'   : 'East Antarctica',
    'filchner_ronne'    : 'Filchner-Ronne',
    'getz'              : 'Getz Ice Shelf',
    'larsen'            : 'Larsen',
    'pine_island'       : 'Pine Island Ice Shelf',
    'pine_island_bay'   : 'Pine Island Bay',
    'ross'              : 'Ross',
    'thwaites'          : 'Thwaites Ice Shelf',
    'west_antarctica'   : 'West Antarctica',
    'filchner_trough'   : 'Filchner Trough',
    'ronne_depression'  : 'Ronne Depression',
    'LAB_trough'        : 'Little America Basin Trough',
    'drygalski_trough'  : 'Drygalski Trough',
    'dotson_cosgrove'   : 'Dotson to Cosgrove',
    'weddell_gyre'      : 'Weddell Gyre',
    'ross_gyre'         : 'Ross Gyre',
}
# Default colours to use for plotting lines
line_colours = ['black', 'Crimson', 'blue', 'DarkMagenta', 'DimGrey', 'DarkGreen', 'DeepPink', 'DeepSkyBlue']
land_colour     = '#9999a3'
iceshelf_colour = '#d4d5da' 

# Keep track of ensemble members downloaded for all variables:
cesm2_ensemble_members = ['1011.001','1031.002','1051.003','1071.004','1091.005','1111.006','1131.007','1151.008','1171.009','1191.010'] + \
                         [f'1231.0{i:02}' for i in range(11,16)] + [f'1251.0{i:02}' for i in range(11,16)] + [f'1281.0{i:02}' for i in range(11,16)] + \
                         [f'1301.0{i:02}' for i in range(11,16)]

# Dictionary of Adusumilli 2020 satellite estimates of melt rates for specific ice shelves and regions, over 1994-2018.
adusumilli_melt = {
    'all' : 1264.3,
    'larsen': 112.7,
    'filchner_ronne': 81.4,
    'east_antarctica': 407.7,  # Wilkes + Amery + Queen Maud
    'amery': 45.6,
    'ross': 123.3,
    'west_antarctica': 539.3,  # Amundsen + Bellingshausen
    'dotson_cosgrove': 210.3,  # Cosgrove + Pine Island + Thwaites + Crosson + Dotson
}
adusumilli_std = {
    'all' : 147.4,
    'larsen' : 103.6,
    'filchner_ronne': 122.9,
    'east_antarctica': 156.9,
    'amery': 40.0,
    'ross': 83.5,
    'west_antarctica': 113.4,
    'dotson_cosgrove': 33.5,
}
# Dictionary of transport observations for Drake Passage (Donohue et al. 2016), Weddell Gyre (Klatt et al. 2005), Ross Gyre including ASC (Dotto et al. 2018)
transport_obs = {
    'drake_passage' : 173.3,
    'weddell_gyre'  : 56,
    'ross_gyre'     : 29,
}
transport_std = {
    'drake_passage' : 10.7,
    'weddell_gyre'  : 8,
    'ross_gyre'     : 8,
}


