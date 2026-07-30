suite_id = "cx209"

suite = {
    "suite_id": suite_id, # suite id to be processed
    "input_dir": "/gws/ssde/j25b/terrafirma/nicdet/mass/u-cx209/", # path where input data is stored
    "output_dir": "/home/users/nicdet/boundary_conditions/testdata", # output path
    "prefix_ocean": f"nemo_{suite_id}o",
    "prefix_ice": f"cice_{suite_id}i",
    "startyear": 1852,
    "endyear": 1852,
    "frequency": "1m",
    "files_per_year": 12,
    "ts_fillvalue": 0, # fillvalue in input data
    "uv_fillvalue": 0, # fillvalue in input data
    "ice_fillvalue": None, # fillvalue in input data
    "out_fillvalue": -9999, # fillvalue in output data
    "file_types": [
        "grid-T", # files with T, S, SSH
        "grid-U", # files with UVEL
        "grid-V", # files with VVEL
    ],
    "target_grid": "/gws/ssde/j25b/terrafirma/nicdet/explore/mesh_mask-20260121.nc", # mesh mask file (eORCA25 grid)
    "regrid_method": "nearest_s2d", # regridding method for horizontal regridding
}