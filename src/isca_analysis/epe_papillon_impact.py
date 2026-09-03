import json
from pathlib import Path
import iris
import numpy as np
import scipy
from matplotlib import pyplot as plt

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
    cube.units = "mm / yr"
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
    print(dat)
    print(obs)
    cube = dat-obs
    try:
        cube.coord("latitude").guess_bounds()
        cube.coord("longitude").guess_bounds()
    except ValueError:
        pass
    grid_areas = iris.analysis.cartography.area_weights(cube)
    return cube.collapsed(
        ["latitude", "longitude"], iris.analysis.RMS, weights=grid_areas
    )

def calculate_mae(dat, obs):
    cube = dat-obs
    try:
        cube.coord("latitude").guess_bounds()
        cube.coord("longitude").guess_bounds()
    except ValueError:
        pass
    grid_areas = iris.analysis.cartography.area_weights(cube)
    return cube.collapsed(
        ["latitude", "longitude"], iris.analysis.MEAN, weights=grid_areas
    )

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

def main():
    # we want to know how much papillon changes global-mean biases
    # so, let's calculate global mean bias for a list of variables,
    # for all ensemble members, and then plot diffs in RMSE/MAE vs sorting
    # metrics of interest (e.g. initial bias, epe params, etc.)
    root_dir = Path(__file__).parents[2]
    out_dir = root_dir.joinpath("data")
    isca_means_dir = out_dir.joinpath("isca_means")
    plots_dir = out_dir.joinpath("plots", "epe_papillon")
    plots_dir.mkdir(exist_ok=True)
    prec_clim_file = out_dir.joinpath("GPCP/IMERG-Final.CLIM.200006-202305.V07B.coarse.nc")
    ceres_file = out_dir.joinpath("CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_CLIM01-CLIM12.nc")
    sbm_ctrl_file = isca_means_dir.joinpath("papillon_control_with_clouds_0060.nc")
    ras_ctrl_file = isca_means_dir.joinpath("papillon_control_ras_0060.nc")
    epe_conf_file = out_dir.joinpath("epe.json")
    with open(epe_conf_file, "r") as f:
        epe_conf = json.load(f)

    # load benchmark data
    benchmark_data = {}
    example_isca_data = iris.load_cube(isca_means_dir.joinpath("epe000_0060.nc"),"soc_toa_sw_up")
    benchmark_data["precipitation"] = iris.load_cube(prec_clim_file, "precipitation")
    benchmark_data["soc_toa_sw_up"] = coarsen_ceres_data(iris.load_cube(ceres_file, "toa_sw_all_clim"), example_isca_data)
    benchmark_data["soc_olr"] = coarsen_ceres_data(iris.load_cube(ceres_file, "toa_lw_all_clim"), example_isca_data)
    benchmark_data["tot_cld_amt"] = coarsen_ceres_data(iris.load_cube(ceres_file, "cldarea_total_daynight_clim"), example_isca_data)

    global_mean_raws = {
        "precipitation": {},
        "soc_olr":{},
        "soc_toa_sw_up":{},
    }
    benchmark_cubes = iris.cube.CubeList([benchmark_data[var_] for var_ in global_mean_raws])

    units = {}

    epe_size = 25
    for i_epe in range(epe_size):
        ctrl_file = isca_means_dir.joinpath(f"epe{i_epe:03d}_0060.nc")
        expt_file = isca_means_dir.joinpath(f"epe_papillon{i_epe:03d}_0060.nc")
        if not ctrl_file.exists() or not expt_file.exists():
            continue
        expt_cubes = iris.load_cubes(expt_file, global_mean_raws)
        ctrl_cubes = iris.load_cubes(ctrl_file, global_mean_raws)
        epe_row = epe_conf[i_epe]
        for expt_cube, ctrl_cube, obs_cube in zip(expt_cubes,ctrl_cubes, benchmark_cubes):
            if expt_cube.var_name == "precipitation":
                expt_cube = convert_precip_units_to_mm_per_year(expt_cube)
                ctrl_cube = convert_precip_units_to_mm_per_year(ctrl_cube)
            rmse_expt = calculate_rmse(expt_cube, obs_cube)
            rmse_ctrl = calculate_rmse(ctrl_cube, obs_cube)
            rmse_diff = rmse_expt - rmse_ctrl
            mae_expt = calculate_mae(expt_cube, obs_cube)
            mae_ctrl = calculate_mae(ctrl_cube, obs_cube)
            mae_diff = mae_expt - mae_ctrl
            gm_expt = calculate_global_mean(expt_cube)
            gm_ctrl = calculate_global_mean(ctrl_cube)
            gm_obs = calculate_global_mean(obs_cube)
            gm_diff_expt = gm_expt - gm_obs
            gm_diff_ctrl = gm_ctrl - gm_obs
            gm_diff_expt_minus_ctrl = gm_expt - gm_ctrl
            var_name = ctrl_cube.var_name
            units[var_name] = gm_expt.units
            for k in ["rmse_expt", "mae_expt", "gm_expt", "gm_diff_expt", "rmse_ctrl", "mae_ctrl", "gm_ctrl", "gm_diff_ctrl", "gm_diff_expt_minus_ctrl", "rmse_diff", "mae_diff"]:
                if k not in global_mean_raws[var_name]:
                    global_mean_raws[var_name][k] = []
            global_mean_raws[var_name]["rmse_expt"].append(rmse_expt.data[0])
            global_mean_raws[var_name]["mae_expt"].append(mae_expt.data[0])
            global_mean_raws[var_name]["gm_expt"].append(gm_expt.data[0])
            global_mean_raws[var_name]["gm_diff_expt"].append(gm_diff_expt.data[0])
            global_mean_raws[var_name]["rmse_ctrl"].append(rmse_ctrl.data[0])
            global_mean_raws[var_name]["mae_ctrl"].append(mae_ctrl.data[0])
            global_mean_raws[var_name]["gm_ctrl"].append(gm_ctrl.data[0])
            global_mean_raws[var_name]["gm_diff_ctrl"].append(gm_diff_ctrl.data[0])
            global_mean_raws[var_name]["gm_diff_expt_minus_ctrl"].append(gm_diff_expt_minus_ctrl.data[0])
            global_mean_raws[var_name]["rmse_diff"].append(rmse_diff.data[0])
            global_mean_raws[var_name]["mae_diff"].append(mae_diff.data[0])
            for epe_param, epe_param_val in epe_row.items():
                if epe_param not in global_mean_raws[var_name]:
                    global_mean_raws[var_name][epe_param] = []
                global_mean_raws[var_name][epe_param].append(epe_param_val)
    print(global_mean_raws)

    for k, v in global_mean_raws.items():
        independent_dependent_variables = [
            ("gm_ctrl", "rmse_diff", {"mean_line", "zero_line"}),
            ("gm_ctrl", "mae_diff", {"mean_line", "zero_line"}),
            ("rmse_ctrl", "rmse_diff", {"mean_line", "zero_line"}),
            ("rmse_ctrl", "mae_diff", {"mean_line", "zero_line"}),
            ("rmse_ctrl", "rmse_expt", {"one_to_one_line", "lock_axes"}),
            ("rmse_ctrl", "rmse_expt", {"one_to_one_line", "lock_axes"}),
            ("mae_ctrl", "mae_expt", {"one_to_one_line", "lock_axes"}),
            ("mae_ctrl", "mae_expt", {"one_to_one_line", "lock_axes"}),
        ]
        for iv_name, dv_name, opts in independent_dependent_variables:
            iv = v[iv_name]
            dv = v[dv_name]

            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(iv, dv)
            r2 = r_value**2
            best_fit = slope*np.asarray(iv)+intercept
            plt.scatter(iv, dv)
            title = k
            if "mean_line" in opts:
                mean = np.mean(dv)
                means = [mean for _ in dv]
                plt.plot(iv, means, color="violet", linestyle="-.")
                title = f"{title}, mean = {mean:.2f}"
            if "best_fit_line" in opts:
                plt.plot(iv, best_fit, color="orange", linestyle="--")
                title = f"{title}, slope = {slope:.2f}, r2 = {r2:.2f}"
            if "one_to_one_line" in opts:
                plt.plot(iv, iv, color="forestgreen", linestyle="solid")
            if "lock_axes" in opts:
                plt.axis("equal")
            if "zero_line" in opts:
                plt.plot(iv, [0.0 for _ in iv], color="black", linestyle="solid")
            plt.xlabel(f"{iv_name} ({units[k]})")
            plt.ylabel(f"{k} {dv_name} ({units[k]})")
            plt.title(title)
            save_path = plots_dir.joinpath(f"{dv_name}_vs_{iv_name}_{k}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"saved plot at {save_path}")

if __name__ == "__main__":
    main()