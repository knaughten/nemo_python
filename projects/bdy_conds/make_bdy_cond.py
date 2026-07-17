from pathlib import Path
import re
from datetime import datetime
import xarray as xr
import xesmf as xe
import gsw
import warnings
import sys
import time

from nemo_python.interpolation import extend_into_mask

def find_boundary_files():
    """
    Find input files for the configured suite.

    Returns
    -------
    dict
        Dictionary containing lists of files by file type.
    """

    path = Path(suite["input_dir"])

    print(f"Looking for {suite['suite_id']}")
    print(f"Years: {suite['startyear']} - {suite['endyear']}")
    print(f"Directory: {path}")
    print(f"Prefix Ocean: {suite['prefix_ocean']}")
    print(f"Prefix Ice: {suite['prefix_ice']}")
    print(f"Frequency: {suite['frequency']}")

    files = {
        ftype: []
        for ftype in suite["file_types"]
    }

    # Add container for ice files
    files["ice_file"] = []

    date_pattern = re.compile(r"(\d{8})-(\d{8})")

    for f in path.iterdir():

        if not f.is_file():
            continue

        filename = f.name

        # Check filename prefix
        is_ocean = suite["prefix_ocean"] in filename
        is_ice = suite["prefix_ice"] in filename

        if not is_ocean and not is_ice:
            continue

        # Extract dates
        match = date_pattern.search(filename)

        if not match:
            continue

        start_date = datetime.strptime(
            match.group(1),
            suite.get("date_format", "%Y%m%d")
        )

        end_date = datetime.strptime(
            match.group(2),
            suite.get("date_format", "%Y%m%d")
        )

        # Skip files that span multiple years
        if start_date.year != suite["startyear"]:
            continue

        # Check overlap with requested period
        if (
            end_date.year < suite["startyear"]
            or start_date.year > suite["endyear"]
        ):
            continue

        # Ice files do not have a file type suffix
        if is_ice:
            files["ice_file"].append(f)
            continue

        # Identify ocean file type
        for ftype in suite["file_types"]:
            if ftype in filename:
                files[ftype].append(f)
                break

    # Sort chronologically
    for ftype in files:
        files[ftype].sort()

    return files


def check_file_completeness(files):
    """
    Check whether each file category contains the expected number of files.
    """

    expected = (
        suite["endyear"] - suite["startyear"] + 1
    ) * suite["files_per_year"]

    print("\nFile completeness check")
    print("-----------------------")

    date_pattern = re.compile(r"(\d{8})-(\d{8})")

    for ftype, filelist in files.items():

        found = len(filelist)

        print(
            f"{ftype:10s}: found {found:4d} / expected {expected:4d}",
            end=""
        )

        if found == expected:
            print("  ✓")
            continue

        print("  ⚠")

        # Determine which months are present
        found_months = set()

        for f in filelist:

            match = date_pattern.search(f.name)

            if match:
                date = datetime.strptime(
                    match.group(1),
                    suite.get("date_format", "%Y%m%d")
                )
                found_months.add((date.year, date.month))

        missing = []

        for year in range(
            suite["startyear"],
            suite["endyear"] + 1
        ):
            for month in range(1, 13):

                if (year, month) not in found_months:
                    missing.append(f"{year}-{month:02d}")

        if missing:
            print("   Missing:")
            for m in missing:
                print(f"      {m}")

def regrid_grid_T(files):
    """
    Regrid all grid-T files.

    Parameters
    ----------
    files : dict
        Dictionary of files returned by find_boundary_files().
    """

    # Ignore a regridding warning. The warning is solely about performance.
    warnings.filterwarnings(
        "ignore",
        message="Input array is not C_CONTIGUOUS. Will affect performance.",
        category=UserWarning,
        module="xesmf.smm",
    )

    grid_T_files = files["grid-T"]

    if not grid_T_files:
        print("No grid-T files found")
        return

    # Load source and target grids.
    source_grid = xr.open_dataset(grid_T_files[0])
    target_grid = xr.open_dataset(suite["target_grid"])

    # Tell the regridder where longitude and latitude are.
    reg_in = xr.Dataset(
        {
            "lat": (["y", "x"], source_grid.nav_lat.data),
            "lon": (["y", "x"], source_grid.nav_lon.data),
        }
    )

    reg_out = xr.Dataset(
        {
            "lat": (["y", "x"], target_grid.gphit.squeeze().data),
            "lon": (["y", "x"], target_grid.glamt.squeeze().data),
        }
    )

    # Compute the regridding weights once so they can be reused.
    regridder = xe.Regridder(
        reg_in,
        reg_out,
        method=suite["regrid_method"],
        periodic=True,
    )

    # Process every grid-T file.
    for infile in grid_T_files:

        infile = Path(infile)

        print(f"Regridding {infile.name}")

        ds = xr.open_dataset(infile)

        # Set fill values to NaN before horizontal regridding.
        # SSH fill values appear to differ between files, so determine the
        # missing value from a land point.
        keys = ds.keys()

        mask = (
            (ds[keys] != suite["ts_fillvalue"])
            & (ds[keys] != ds.zos[0, 0, 0])
        )

        ds_reg_h = regridder(ds[keys].where(mask))

        # Select boundary slice.
        to_fill = ds_reg_h.isel(y=slice(451, 452))

        # Fill missing values (extend_into_mask expects NumPy arrays).
        thetao_filled_np = extend_into_mask(
            to_fill.thetao.fillna(-9999).values.copy(),
            use_3d=True,
            num_iters=5,
        )

        so_filled_np = extend_into_mask(
            to_fill.so.fillna(-9999).values.copy(),
            use_3d=True,
            num_iters=5,
        )

        zos_filled_np = extend_into_mask(
            to_fill.zos.fillna(-9999).values.copy(),
            use_2d=True,
            num_iters=5,
        )

        # Put the filled arrays back into an xarray Dataset.
        ds_filled = to_fill.copy()

        ds_filled["thetao"] = xr.DataArray(
            thetao_filled_np,
            dims=to_fill.thetao.dims,
            coords=to_fill.thetao.coords,
            attrs=to_fill.thetao.attrs,
        )

        ds_filled["so"] = xr.DataArray(
            so_filled_np,
            dims=to_fill.so.dims,
            coords=to_fill.so.coords,
            attrs=to_fill.so.attrs,
        )

        ds_filled["zos"] = xr.DataArray(
            zos_filled_np,
            dims=to_fill.zos.dims,
            coords=to_fill.zos.coords,
            attrs=to_fill.zos.attrs,
        )

        # Restore masked values.
        vars_to_mask = ["so", "thetao", "zos"]
        ds_filled[vars_to_mask] = ds_filled[vars_to_mask].where(
            ds_filled[vars_to_mask] != -9999
        )

        # Perform vertical interpolation.
        ds_reg_v = ds_filled.interp(
            deptht=target_grid.nav_lev,
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )

        # Recover longitude and latitude from the target grid.
        target_slice = target_grid.isel(x=slice(0, 1440), y=slice(451, 452))

        # Compute TEOS-10 variables.
        AbsSal = gsw.SA_from_SP(
            ds_reg_v.so,
            ds_reg_v.deptht,
            target_slice.nav_lon,
            target_slice.nav_lat,
        )

        ConsTemp = gsw.CT_from_pt(AbsSal.values, ds_reg_v.thetao)

        # Construct output filenames.
        match = re.search(r"(\d{4})(\d{2})\d{2}-", infile.name)

        year = match.group(1)
        month = match.group(2)

        outfile_s = (
            Path(suite["output_dir"])
            / f"AbsSal_{suite['suite_id']}_y{year}m{month}.nc"
        )
        outfile_t = (
            Path(suite["output_dir"])
            / f"ConsTemp_{suite['suite_id']}_y{year}m{month}.nc"
        )
        outfile_zos = (
            Path(suite["output_dir"])
            / f"SSH_{suite['suite_id']}_y{year}m{month}.nc"
        )

        (
            AbsSal.where(target_slice.tmask != 0)
            .rename("AbsSal")
            .drop_vars(["time_centered", "deptht"])
            .rename({"nav_lev": "deptht"})
            .fillna(suite["out_fillvalue"])
            .to_netcdf(outfile_s, unlimited_dims=["time_counter"])
        )

        (
            ConsTemp.where(target_slice.tmask != 0)
            .rename("ConsTemp")
            .drop_vars(["time_centered", "deptht"])
            .rename({"nav_lev": "deptht"})
            .fillna(suite["out_fillvalue"])
            .to_netcdf(outfile_t, unlimited_dims=["time_counter"])
        )

        (
            ds_filled.zos.where(target_slice.tmask.isel(nav_lev=0) != 0)
            .drop_vars(["time_centered"])
            .rename("SSH")
            .fillna(suite["out_fillvalue"])
            .to_netcdf(outfile_zos, unlimited_dims=["time_counter"])
        )

        ds.close()
        ds_reg_h.close()
        ds_filled.close()
        ds_reg_v.close()


def regrid_grid_V(files):
    """
    Regrid all grid-V files.

    Parameters
    ----------
    files : dict
        Dictionary of files returned by find_boundary_files().
    """

    # Ignore a regridding warning. The warning is solely about performance.
    warnings.filterwarnings(
        "ignore",
        message="Input array is not C_CONTIGUOUS. Will affect performance.",
        category=UserWarning,
        module="xesmf.smm",
    )

    grid_V_files = files["grid-V"]

    if not grid_V_files:
        print("No grid-V files found")
        return

    # Load source and target grids.
    source_grid = xr.open_dataset(grid_V_files[0])
    target_grid = xr.open_dataset(suite["target_grid"])

    # Tell the regridder where longitude and latitude are.
    reg_in = xr.Dataset(
        {
            "lat": (["y", "x"], source_grid.nav_lat.data),
            "lon": (["y", "x"], source_grid.nav_lon.data),
        }
    )

    reg_out = xr.Dataset(
        {
            "lat": (["y", "x"], target_grid.gphiv.squeeze().data),
            "lon": (["y", "x"], target_grid.glamv.squeeze().data),
        }
    )

    # Compute the regridding weights once so they can be reused.
    regridder = xe.Regridder(
        reg_in,
        reg_out,
        method=suite["regrid_method"],
        periodic=True,
    )

    # Process every grid-V file.
    for infile in grid_V_files:

        infile = Path(infile)

        print(f"Regridding {infile.name}")

        ds = xr.open_dataset(infile)

        # Set fill values to NaN before horizontal regridding.
        keys = ds.keys()

        ds_reg_h = regridder(
            ds[keys].where(ds[keys] != suite["uv_fillvalue"])
        )

        # Select boundary slice.
        to_fill = ds_reg_h.isel(y=slice(451, 452))

        # Fill missing values (extend_into_mask expects NumPy arrays).
        vo_filled_np = extend_into_mask(
            to_fill.vo.fillna(-9999).values.copy(),
            use_3d=True,
            num_iters=5,
        )

        # Put the filled array back into an xarray Dataset.
        ds_filled = to_fill.copy()

        ds_filled["vo"] = xr.DataArray(
            vo_filled_np,
            dims=to_fill.vo.dims,
            coords=to_fill.vo.coords,
            attrs=to_fill.vo.attrs,
        )

        # Restore masked values.
        ds_filled["vo"] = ds_filled["vo"].where(ds_filled["vo"] != -9999)

        # Perform vertical interpolation.
        ds_reg_v = ds_filled.interp(
            depthv=target_grid.nav_lev,
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )

        # Recover the target grid coordinates.
        target_slice = target_grid.isel(
            x=slice(0, 1440),
            y=slice(450, 451),
        )

        # Construct the output filename.
        match = re.search(r"(\d{4})(\d{2})\d{2}-", infile.name)

        year = match.group(1)
        month = match.group(2)

        outfile_v = (
            Path(suite["output_dir"])
            / f"VVEL_{suite['suite_id']}_y{year}m{month}.nc"
        )

        (
            ds_reg_v.vo.where(target_slice.vmask != 0)
            .fillna(suite["out_fillvalue"])
            .rename("VVEL")
            .drop_vars(["time_centered", "depthv"])
            .rename({"nav_lev": "depthv"})
            .to_netcdf(outfile_v, unlimited_dims=["time_counter"])
        )

        ds.close()
        ds_reg_h.close()
        ds_filled.close()
        ds_reg_v.close()

def regrid_grid_U(files):
    """
    Regrid all grid-U files.

    Parameters
    ----------
    files : dict
        Dictionary of files returned by find_boundary_files().
    """

    # Ignore a regridding warning. The warning is solely about performance.
    warnings.filterwarnings(
        "ignore",
        message="Input array is not C_CONTIGUOUS. Will affect performance.",
        category=UserWarning,
        module="xesmf.smm",
    )

    grid_U_files = files["grid-U"]

    if not grid_U_files:
        print("No grid-U files found")
        return

    # Load source and target grids.
    source_grid = xr.open_dataset(grid_U_files[0])
    target_grid = xr.open_dataset(suite["target_grid"])

    # Tell the regridder where longitude and latitude are.
    reg_in = xr.Dataset(
        {
            "lat": (["y", "x"], source_grid.nav_lat.data),
            "lon": (["y", "x"], source_grid.nav_lon.data),
        }
    )

    reg_out = xr.Dataset(
        {
            "lat": (["y", "x"], target_grid.gphiu.squeeze().data),
            "lon": (["y", "x"], target_grid.glamu.squeeze().data),
        }
    )

    # Compute the regridding weights once so they can be reused.
    regridder = xe.Regridder(
        reg_in,
        reg_out,
        method=suite["regrid_method"],
        periodic=True,
    )

    # Process every grid-U file.
    for infile in grid_U_files:

        infile = Path(infile)

        print(f"Regridding {infile.name}")

        ds = xr.open_dataset(infile)

        # Set fill values to NaN before horizontal regridding.
        keys = ds.keys()

        ds_reg_h = regridder(
            ds[keys].where(ds[keys] != suite["uv_fillvalue"])
        )

        # Select boundary slice.
        to_fill = ds_reg_h.isel(y=slice(451, 452))

        # Fill missing values (extend_into_mask expects NumPy arrays).
        uo_filled_np = extend_into_mask(
            to_fill.uo.fillna(-9999).values.copy(),
            use_3d=True,
            num_iters=5,
        )

        # Put the filled array back into an xarray Dataset.
        ds_filled = to_fill.copy()

        ds_filled["uo"] = xr.DataArray(
            uo_filled_np,
            dims=to_fill.uo.dims,
            coords=to_fill.uo.coords,
            attrs=to_fill.uo.attrs,
        )

        # Restore masked values.
        ds_filled["uo"] = ds_filled["uo"].where(ds_filled["uo"] != -9999)

        # Perform vertical interpolation.
        ds_reg_v = ds_filled.interp(
            depthu=target_grid.nav_lev,
            method="linear",
            kwargs={"fill_value": "extrapolate"},
        )

        # Recover the target grid coordinates.
        target_slice = target_grid.isel(
            x=slice(0, 1440),
            y=slice(451, 452),
        )

        # Construct the output filename.
        match = re.search(r"(\d{4})(\d{2})\d{2}-", infile.name)

        year = match.group(1)
        month = match.group(2)

        outfile_u = (
            Path(suite["output_dir"])
            / f"UVEL_{suite['suite_id']}_y{year}m{month}.nc"
        )

        (
            ds_reg_v.uo.where(target_slice.umask != 0)
            .fillna(suite["out_fillvalue"])
            .rename("UVEL")
            .drop_vars(["time_centered", "depthu"])
            .rename({"nav_lev": "depthu"})
            .to_netcdf(outfile_u, unlimited_dims=["time_counter"])
        )

        ds.close()
        ds_reg_h.close()
        ds_filled.close()
        ds_reg_v.close()

def regrid_ice(files):
    """
    Regrid all ice files.

    Parameters
    ----------
    files : dict
        Dictionary of files returned by find_boundary_files().
    """

    # Ignore a regridding warning. The warning is solely about performance.
    warnings.filterwarnings(
        "ignore",
        message="Input array is not C_CONTIGUOUS. Will affect performance.",
        category=UserWarning,
        module="xesmf.smm",
    )

    ice_files = files["ice_file"]

    if not ice_files:
        print("No ice files found")
        return

    # Load source and target grids.
    source_grid = xr.open_dataset(ice_files[0])
    target_grid = xr.open_dataset(suite["target_grid"])

    # Tell the regridder where longitude and latitude are.
    reg_in = xr.Dataset(
        {
            "lat": (["nj", "ni"], source_grid.TLAT.data),
            "lon": (["nj", "ni"], source_grid.TLON.data),
        }
    )

    reg_out = xr.Dataset(
        {
            "lat": (["y", "x"], target_grid.gphit.squeeze().data),
            "lon": (["y", "x"], target_grid.glamt.squeeze().data),
        }
    )

    # Compute the regridding weights once so they can be reused.
    regridder = xe.Regridder(
        reg_in,
        reg_out,
        method=suite["regrid_method"],
        periodic=True,
    )

    # Process every ice file.
    for infile in ice_files:

        infile = Path(infile)

        print(f"Regridding {infile.name}")

        ds = xr.open_dataset(infile)

        # Set fill values to NaN before horizontal regridding.

        keys = ['aice', 'hs', 'hi']

        if suite["ice_fillvalue"] is None:
            ds_reg_h = regridder(ds[keys])
        else:
            ds_reg_h = regridder(
                ds[keys].where(ds[keys] != suite["ice_fillvalue"])
            )

        # Select boundary slice.
        to_fill = ds_reg_h.isel(y=slice(451, 452))

        # Fill missing values (extend_into_mask expects NumPy arrays).
        sia_filled_np = extend_into_mask(
            to_fill.aice.fillna(-9999).values.copy(),
            use_2d=True,
            num_iters=5,
        )

        hi_filled_np = extend_into_mask(
            to_fill.hi.fillna(-9999).values.copy(),
            use_2d=True,
            num_iters=5,
        )

        hs_filled_np = extend_into_mask(
            to_fill.hs.fillna(-9999).values.copy(),
            use_2d=True,
            num_iters=5,
        )

        # Put the filled arrays back into an xarray Dataset.
        ds_filled = to_fill.copy()

        ds_filled["aice"] = xr.DataArray(
            sia_filled_np,
            dims=to_fill.aice.dims,
            coords=to_fill.aice.coords,
            attrs=to_fill.aice.attrs,
        )

        ds_filled["hi"] = xr.DataArray(
            hi_filled_np,
            dims=to_fill.hi.dims,
            coords=to_fill.hi.coords,
            attrs=to_fill.hi.attrs,
        )

        ds_filled["hs"] = xr.DataArray(
            hs_filled_np,
            dims=to_fill.hs.dims,
            coords=to_fill.hs.coords,
            attrs=to_fill.hs.attrs,
        )

        # Restore masked values.
        vars_to_mask = ['aice', 'hs', 'hi']
        ds_filled[vars_to_mask] = ds_filled[vars_to_mask].where(
            ds_filled[vars_to_mask] != -9999
        )

        # Recover longitude and latitude from the target grid.
        target_slice = target_grid.isel(x=slice(0, 1440), y=slice(451, 452))

        # Construct output filenames.
        match = re.search(r"(\d{4})(\d{2})\d{2}-", infile.name)

        year = match.group(1)
        month = match.group(2)

  
        outfile_sia = (
            Path(suite["output_dir"])
            / f"SIarea_{suite['suite_id']}_y{year}m{month}.nc"
        )

        outfile_hi = (
            Path(suite["output_dir"])
            / f"SIheff_{suite['suite_id']}_y{year}m{month}.nc"
        )

        outfile_hs = (
            Path(suite["output_dir"])
            / f"SIhsnow_{suite['suite_id']}_y{year}m{month}.nc"
        )

        (
            ds_filled.aice.where(target_slice.tmask.isel(nav_lev=0) != 0)
            .drop_vars(["time"])
            .squeeze('time')
            .rename("SIarea")
            .fillna(suite["out_fillvalue"])
            .to_netcdf(outfile_sia, unlimited_dims=["time_counter"])
        )

        (
            ds_filled.hi.where(target_slice.tmask.isel(nav_lev=0) != 0)
            .drop_vars(["time"])
            .squeeze('time')
            .rename("SIheff")
            .fillna(suite["out_fillvalue"])
            .to_netcdf(outfile_hi, unlimited_dims=["time_counter"])
        )

        (
            ds_filled.hs.where(target_slice.tmask.isel(nav_lev=0) != 0)
            .drop_vars(["time"])
            .squeeze('time')
            .rename("SIhsnow")
            .fillna(suite["out_fillvalue"])
            .to_netcdf(outfile_hs, unlimited_dims=["time_counter"])
        )
        
        ds.close()
        ds_reg_h.close()
        ds_filled.close()


from config import suite

def main():
    """
    Run the full regridding workflow.
    """
    
    start_time = time.perf_counter()

    output_dir = Path(suite["output_dir"])
    
    if not output_dir.exists():
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
    print("Finding files...")
    files = find_boundary_files()

    print("Checking completeness...")
    check_file_completeness(files)

    print("Regridding temperature, salinity and sea surface height...")
    regrid_grid_T(files)

    print("Regridding zonal velocity...")
    regrid_grid_U(files)

    print("Regridding meridional velocity...")
    regrid_grid_V(files)

    print("Regridding ice...")
    regrid_ice(files)
    
    print("All done.")

    elapsed_time = time.perf_counter() - start_time

    print(f"Finished in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    main()
