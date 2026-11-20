import json
import re
import warnings
from pathlib import Path
import iris
import numpy as np
import scipy
from matplotlib import pyplot as plt
import iris.quickplot as qplot


def coarsen_ceres_data(clim_fine, target):
    try:
        target.coord("latitude").guess_bounds()
    except ValueError as v:
        print(v)
    try:
        target.coord("longitude").guess_bounds()
    except ValueError as v:
        print(v)
    try:
        clim_fine.coord("latitude").guess_bounds()
    except ValueError as v:
        print(v)
    try:
        clim_fine.coord("longitude").guess_bounds()
    except ValueError as v:
        print(v)
    clim_fine = clim_fine.collapsed(
        "Climatological Monthly Means Based on 07/2005 to 06/2015",
        iris.analysis.MEAN,
    )
    clim = clim_fine.regrid(target, iris.analysis.AreaWeighted())
    return clim


def convert_precip_units_to_mm_per_year(cube):
    s_per_year = 3600 * 24 * 365.25
    cube = s_per_year * cube
    cube.units = "mm / year"
    cube.var_name = "precipitation"
    return cube


def calculate_global_mean(cube):
    try:
        cube.coord("latitude").guess_bounds()
        cube.coord("longitude").guess_bounds()
    except ValueError:
        pass
    grid_areas = iris.analysis.cartography.area_weights(cube)
    return cube.collapsed(
        ["latitude", "longitude"], iris.analysis.MEAN, weights=grid_areas
    )


def calculate_rmse(dat, obs):
    cube = dat - obs
    try:
        cube.coord("latitude").guess_bounds()
        cube.coord("longitude").guess_bounds()
    except ValueError:
        pass


def calculate_stat_zonal_area(cube, stat, min_lat, max_lat):
    subcube = cube.extract(iris.Constraint(latitude=lambda lat: min_lat < lat < max_lat))
    try:
        subcube.coord("latitude").guess_bounds()
        subcube.coord("longitude").guess_bounds()
    except ValueError:
        pass
    grid_areas = iris.analysis.cartography.area_weights(subcube)
    return subcube.collapsed(
        ["latitude", "longitude"], stat, weights=grid_areas
    )


def calculate_mae(dat, obs):
    diff = dat - obs
    cube_data = np.abs(diff.data)
    cube = diff
    cube.data = cube_data
    try:
        cube.coord("latitude").guess_bounds()
        cube.coord("longitude").guess_bounds()
    except ValueError:
        pass
    return cube


def extract_independent_variable(epe_row, independent_variable_name):
    match independent_variable_name:
        case "reff":
            reff_liq = epe_row["cloud_simple_nml"]["reff_liq"]
            return reff_liq / 14.
        case "rhc":
            rhcsfc = epe_row["large_scale_cloud_nml"]["rhcsfc"]
            return rhcsfc / 0.95
        case "rh_trig":
            if epe_row["idealized_moist_phys_nml"]["convection_scheme"] == "RAS":
                return epe_row["ras_nml"]["rh_trig"] / 0.35
            else:
                return None
        case "alm_min":
            if epe_row["idealized_moist_phys_nml"]["convection_scheme"] == "RAS":
                return epe_row["ras_nml"]["alm_min"] / 0.25
            else:
                return None
        case "tau_bm":
            if epe_row["idealized_moist_phys_nml"]["convection_scheme"] == "SIMPLE_BETTS_MILLER":
                return epe_row["qe_moist_convection_nml"]["tau_bm"] / 7200.
            else:
                return None
        case "rhbm":
            if epe_row["idealized_moist_phys_nml"]["convection_scheme"] == "SIMPLE_BETTS_MILLER":
                return epe_row["qe_moist_convection_nml"]["rhbm"] / 0.8
            else:
                return None
        case _:
            raise ValueError("unrecognised option")


def fix_units(cube, varname):
    match varname:
        case "precipitation":
            cube.units = "mm / year"
        case "soc_olr":
            cube.units = "W/m^2"
        case "soc_toa_sw_up":
            cube.units = "W/m^2"
    return cube


def main(ctrl_file, expt_file):
    # we want to know how much papillon changes global-mean biases
    # so, let's calculate global mean bias for a list of variables,
    # for all ensemble members, and then plot diffs in RMSE/MAE vs sorting
    # metrics of interest (e.g. initial bias, epe params, etc.)
    root_dir = Path(__file__).parents[2]
    out_dir = root_dir.joinpath("data")
    plots_dir = out_dir.joinpath("plots", "metrics")
    plots_dir.mkdir(exist_ok=True)
    isca_means_dir = out_dir.joinpath("isca_means")
    ctrl_file = isca_means_dir / ctrl_file
    expt_file = isca_means_dir / expt_file
    prec_clim_file = out_dir.joinpath("GPCP/IMERG-Final.CLIM.200006-202305.V07B.coarse.nc")
    ceres_file = out_dir.joinpath("CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_CLIM01-CLIM12.nc")
    sbm_ctrl_file = isca_means_dir.joinpath("papillon_control_with_clouds_0060.nc")
    ras_ctrl_file = isca_means_dir.joinpath("papillon_control_ras_0060.nc")
    epe_conf_file = out_dir.joinpath("epe.json")
    with open(epe_conf_file, "r") as f:
        epe_conf = json.load(f)

    units = {}

    # load benchmark data
    benchmark_data = {}
    example_isca_data = iris.load_cube(isca_means_dir.joinpath("epe000_0060.nc"), "soc_toa_sw_up")
    benchmark_data["precipitation"] = iris.load_cube(prec_clim_file, "precipitation")
    benchmark_data["soc_toa_sw_up"] = coarsen_ceres_data(iris.load_cube(ceres_file, "toa_sw_all_clim"),
                                                         example_isca_data)
    benchmark_data["soc_olr"] = coarsen_ceres_data(iris.load_cube(ceres_file, "toa_lw_all_clim"), example_isca_data)
    benchmark_data["tot_cld_amt"] = coarsen_ceres_data(iris.load_cube(ceres_file, "cldarea_total_daynight_clim"),
                                                       example_isca_data)

    mae_vars = {
        "precipitation": 0.0,
        "soc_olr": 0.0,
        "soc_toa_sw_up": 0.0,
    }
    _diff_vars = {
        "precipitation",
        "soc_olr",
        "soc_toa_sw_up",
        "low_cld_amt",
        "mid_cld_amt",
        "high_cld_amt",
        "tot_cld_amt",
        "flux_lhe",
        "flux_t",
        "convection_rain",
        "t_surf",
    }
    diff_vars = {k: [] for k in _diff_vars}
    benchmark_cubes = iris.cube.CubeList([benchmark_data[var_] for var_ in mae_vars])

    if not ctrl_file.exists() or not expt_file.exists():
        print(f"skipping file due to expt or ctrl file not existing")
        return
    expt_cubes = iris.load_cubes(expt_file, mae_vars)
    ctrl_cubes = iris.load_cubes(ctrl_file, mae_vars)
    for expt_cube, ctrl_cube, obs_cube in zip(expt_cubes, ctrl_cubes, benchmark_cubes):
        if expt_cube.var_name == "precipitation":
            expt_cube = convert_precip_units_to_mm_per_year(expt_cube)
            ctrl_cube = convert_precip_units_to_mm_per_year(ctrl_cube)
        norm_diff = np.linalg.norm(ctrl_cube.data[0] - expt_cube.data[0])
        var_name = ctrl_cube.var_name
        units[var_name] = str(expt_cube.units)
        mae_vars[expt_cube.var_name] = f"{float(norm_diff):.0f}"
    return f"{ctrl_file.stem} v {expt_file.stem}: {mae_vars}", mae_vars
    return
    for var_ in mae_vars.keys():
        plot_mae_diff_map(mae_vars[var_], f"bias_{var_}", plots_dir, "mean_bias")
    # ----------------
    # start diff plots
    epe_size = 25
    for i_epe in range(epe_size):
        ctrl_file = isca_means_dir.joinpath(f"epe{i_epe:03d}{suffix}.nc")
        expt_file = isca_means_dir.joinpath(f"epe_papillon{i_epe:03d}{suffix}.nc")
        if not ctrl_file.exists() or not expt_file.exists():
            print(f"skipping file {i_epe} due to expt or ctrl file not existing")
            continue
        expt_cubes = iris.load_cubes(expt_file, diff_vars)
        ctrl_cubes = iris.load_cubes(ctrl_file, diff_vars)
        for expt_cube, ctrl_cube in zip(expt_cubes, ctrl_cubes):
            if expt_cube.var_name == "precipitation":
                expt_cube = convert_precip_units_to_mm_per_year(expt_cube)
                ctrl_cube = convert_precip_units_to_mm_per_year(ctrl_cube)
            diff_cube = expt_cube - ctrl_cube
            diff_vars[expt_cube.var_name].append(diff_cube)
    for var_ in diff_vars:
        average_diff_map = sum(diff_vars[var_]) / len(diff_vars[var_])
        average_diff_map = average_diff_map[0]
        plot_mae_diff_map(average_diff_map, var_, plots_dir, "mean")
        sd_map = average_diff_map
        sd_map.data = np.std(np.asarray([map_.data for map_ in diff_vars[var_]]), axis=0)[0]
        plot_mae_diff_map(sd_map, var_, plots_dir, "sd")


def plot_mae_diff_map(data, varname, plots_dir, operation):
    match operation:
        case "mean_bias":
            title = f"{varname} average change in bias\ndue to PAPILLON"
            vmax = np.max(data)
            vmin = np.min(data)
            vmax = np.max([abs(vmax), abs(vmin)])
            vmin = -vmax
            cmap = "bwr"
        case "sd_bias":
            title = f"{varname} standard deviation of bias\ndue to PAPILLON"
            vmin = 0
            vmax = np.max(data)
            cmap = "viridis"
        case "mean":
            title = f"{varname} average change\ndue to PAPILLON"
            vmax = np.max(data)
            vmin = np.min(data)
            vmax = np.max([abs(vmax), abs(vmin)])
            vmin = -vmax
            cmap = "PiYG"
            if varname in ["precipitation", "convection_rain"]:
                cmap = "BrBG"
        case "sd":
            title = f"{varname} standard deviation of change\ndue to PAPILLON"
            vmin = 0
            vmax = np.max(data)
            cmap = "viridis"
        case _:
            title = None
            cmap = None
            vmin = None
            vmax = None
    data = fix_units(data, varname)
    qplot.pcolormesh(data, cmap=cmap)
    plt.clim(vmin=vmin, vmax=vmax)
    plt.gca().coastlines()
    plt.title(title)
    plt.savefig(plots_dir.joinpath(f"{operation}_{varname}.png"))
    plt.close()


def plot_teracota_curves(data):
    root_dir = Path(__file__).parents[2]
    out_dir = root_dir.joinpath("data")
    plots_dir = out_dir.joinpath("plots", "metrics", "teracota", "frobenius")
    plots_dir.mkdir(exist_ok=True, parents=True)
    vars = {
        "precipitation",
        "soc_olr",
        "soc_toa_sw_up",
    }
    for var_ in vars:
        points = {}
        for control, expt_dict in data.items():
            control_name = re.search(r"teracota_control_([a-z]*)_0120", control)
            if control_name is None:
                continue
            control_name = control_name.group(1)
            if control_name not in points:
                points[control_name] = []
            for experiment, norm in expt_dict.items():
                hour = (re.search(r"teracota_ar1_([0-9]*)h_0120",experiment))
                hour = float(hour.group(1)) if hour is not None else None
                if hour is None:
                    continue
                points[control_name].append((hour,float(norm[var_])))
        for control, point_data in points.items():
            point_data = np.asarray(point_data).T
            plt.plot(point_data[0], point_data[1], label=f"compared to {control} control")
        plt.xlabel("ar1 timescale (hours)")
        plt.ylabel(f"frobenius distance")
        plt.title(f"{var_}")
        plt.legend()
        plt.savefig(plots_dir / f"{var_}.png")
        plt.close()
if __name__ == "__main__":
    files = [
        ["teracota_control_ras_0120.nc", "teracota_control_sbm_0120.nc"],
        ["teracota_control_ras_0120.nc", "teracota_ar1_1h_0120.nc"],
        ["teracota_control_sbm_0120.nc", "teracota_ar1_1h_0120.nc"],

        ["teracota_control_ras_0120.nc", "teracota_ar1_3h_0120.nc"],
        ["teracota_control_sbm_0120.nc", "teracota_ar1_3h_0120.nc"],

        ["teracota_control_ras_0120.nc", "teracota_ar1_6h_0120.nc"],
        ["teracota_control_sbm_0120.nc", "teracota_ar1_6h_0120.nc"],

        ["teracota_control_ras_0120.nc", "teracota_ar1_9h_0120.nc"],
        ["teracota_control_sbm_0120.nc", "teracota_ar1_9h_0120.nc"],

        ["teracota_control_ras_0120.nc", "teracota_ar1_11h_0120.nc"],
        ["teracota_control_sbm_0120.nc", "teracota_ar1_11h_0120.nc"],

        ["teracota_control_ras_0120.nc", "teracota_ar1_27h_0120.nc"],
        ["teracota_control_sbm_0120.nc", "teracota_ar1_27h_0120.nc"],
    ]
    outputs = []
    norms = {}
    for file1, file2 in files:
        with warnings.catch_warnings(action="ignore"):
            output, norm = main(file1, file2)
            outputs.append(output)
            if file1 not in norms:
                norms[file1] = {file2:norm}
            else:
                norms[file1][file2] = norm
    for output in outputs:
        print(output)
    plot_teracota_curves(norms)