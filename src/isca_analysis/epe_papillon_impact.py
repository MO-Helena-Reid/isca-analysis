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

def calculate_wasserstein_distance(dat, obs):
    try:
        wd = scipy.stats.wasserstein_distance_nd(dat.data[0],obs.data)
    except ValueError:
        wd = scipy.stats.wasserstein_distance_nd(dat.data[0],obs.data.T)
    return wd

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

def main(regen_gm_raw_file=False):
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

    units = {}

    units_file = plots_dir.joinpath("units.json")
    gm_raw_file = plots_dir.joinpath("gm_raw.json")
    if regen_gm_raw_file:
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
                gm_diff_diff = gm_expt - gm_ctrl

                wasserstein_distance_expt = calculate_wasserstein_distance(expt_cube, obs_cube)
                wasserstein_distance_ctrl = calculate_wasserstein_distance(ctrl_cube, obs_cube)
                wasserstein_distance_diff = wasserstein_distance_expt - wasserstein_distance_ctrl

                rmse_tropics_expt = calculate_stat_zonal_area(expt_cube-obs_cube, iris.analysis.RMS, -23.4,23.4)
                rmse_tropics_ctrl = calculate_stat_zonal_area(ctrl_cube-obs_cube, iris.analysis.RMS, -23.4,23.4)
                rmse_tropics_diff = rmse_tropics_expt - rmse_tropics_ctrl
                rmse_n_subtropics_expt = calculate_stat_zonal_area(expt_cube-obs_cube, iris.analysis.RMS, 23.4,35)
                rmse_n_subtropics_ctrl = calculate_stat_zonal_area(ctrl_cube-obs_cube, iris.analysis.RMS, 23.4,35)
                rmse_n_subtropics_diff = rmse_n_subtropics_expt - rmse_n_subtropics_ctrl
                rmse_s_subtropics_expt = calculate_stat_zonal_area(expt_cube-obs_cube, iris.analysis.RMS, -35,-23.4)
                rmse_s_subtropics_ctrl = calculate_stat_zonal_area(ctrl_cube-obs_cube, iris.analysis.RMS, -35,-23.4)
                rmse_s_subtropics_diff = rmse_s_subtropics_expt - rmse_s_subtropics_ctrl

                rmse_n_midlat_expt = calculate_stat_zonal_area(expt_cube-obs_cube, iris.analysis.RMS, 35,66)
                rmse_n_midlat_ctrl = calculate_stat_zonal_area(ctrl_cube-obs_cube, iris.analysis.RMS, 35,66)
                rmse_n_midlat_diff = rmse_n_midlat_expt - rmse_n_midlat_ctrl
                rmse_s_midlat_expt = calculate_stat_zonal_area(expt_cube-obs_cube, iris.analysis.RMS, -66,-35)
                rmse_s_midlat_ctrl = calculate_stat_zonal_area(ctrl_cube-obs_cube, iris.analysis.RMS, -66,-35)
                rmse_s_midlat_diff = rmse_s_midlat_expt - rmse_s_midlat_ctrl

                rmse_n_polar_expt = calculate_stat_zonal_area(expt_cube-obs_cube, iris.analysis.RMS, 66,90)
                rmse_n_polar_ctrl = calculate_stat_zonal_area(ctrl_cube-obs_cube, iris.analysis.RMS, 66,90)
                rmse_n_polar_diff = rmse_n_polar_expt - rmse_n_polar_ctrl
                rmse_s_polar_expt = calculate_stat_zonal_area(expt_cube-obs_cube, iris.analysis.RMS, -90,-66)
                rmse_s_polar_ctrl = calculate_stat_zonal_area(ctrl_cube-obs_cube, iris.analysis.RMS, -90,-66)
                rmse_s_polar_diff = rmse_s_polar_expt - rmse_s_polar_ctrl

                var_name = ctrl_cube.var_name
                units[var_name] = str(gm_expt.units)
                for k in ["rmse", "mae", "gm", "gm_diff", "wasserstein_distance", "rmse_tropics", "rmse_n_subtropics", "rmse_s_subtropics", "rmse_n_midlat", "rmse_s_midlat", "rmse_n_polar", "rmse_s_polar",]:
                    for k2 in [f"{k}_expt", f"{k}_ctrl", f"{k}_diff"]:
                        if k2 not in global_mean_raws[var_name]:
                            global_mean_raws[var_name][k2] = []
                global_mean_raws[var_name]["gm_clim"] = float(gm_obs.data)
                global_mean_raws[var_name]["rmse_expt"].append(rmse_expt.data[0])
                global_mean_raws[var_name]["mae_expt"].append(mae_expt.data[0])
                global_mean_raws[var_name]["gm_expt"].append(gm_expt.data[0])
                global_mean_raws[var_name]["gm_diff_expt"].append(gm_diff_expt.data[0])
                global_mean_raws[var_name]["rmse_ctrl"].append(rmse_ctrl.data[0])
                global_mean_raws[var_name]["mae_ctrl"].append(mae_ctrl.data[0])
                global_mean_raws[var_name]["gm_ctrl"].append(gm_ctrl.data[0])
                global_mean_raws[var_name]["gm_diff_ctrl"].append(gm_diff_ctrl.data[0])
                global_mean_raws[var_name]["gm_diff_diff"].append(gm_diff_diff.data[0])
                global_mean_raws[var_name]["rmse_diff"].append(rmse_diff.data[0])
                global_mean_raws[var_name]["mae_diff"].append(mae_diff.data[0])
                global_mean_raws[var_name]["wasserstein_distance_ctrl"].append(wasserstein_distance_ctrl)
                global_mean_raws[var_name]["wasserstein_distance_expt"].append(wasserstein_distance_expt)
                global_mean_raws[var_name]["wasserstein_distance_diff"].append(wasserstein_distance_diff)
                global_mean_raws[var_name]["rmse_tropics_expt"].append(rmse_tropics_expt.data[0])
                global_mean_raws[var_name]["rmse_tropics_ctrl"].append(rmse_tropics_ctrl.data[0])
                global_mean_raws[var_name]["rmse_tropics_diff"].append(rmse_tropics_diff.data[0])
                global_mean_raws[var_name]["rmse_n_subtropics_expt"].append(rmse_n_subtropics_expt.data[0])
                global_mean_raws[var_name]["rmse_n_subtropics_ctrl"].append(rmse_n_subtropics_ctrl.data[0])
                global_mean_raws[var_name]["rmse_n_subtropics_diff"].append(rmse_n_subtropics_diff.data[0])
                global_mean_raws[var_name]["rmse_s_subtropics_expt"].append(rmse_s_subtropics_expt.data[0])
                global_mean_raws[var_name]["rmse_s_subtropics_ctrl"].append(rmse_s_subtropics_ctrl.data[0])
                global_mean_raws[var_name]["rmse_s_subtropics_diff"].append(rmse_s_subtropics_diff.data[0])
                global_mean_raws[var_name]["rmse_n_midlat_expt"].append(rmse_n_midlat_expt.data[0])
                global_mean_raws[var_name]["rmse_n_midlat_ctrl"].append(rmse_n_midlat_ctrl.data[0])
                global_mean_raws[var_name]["rmse_n_midlat_diff"].append(rmse_n_midlat_diff.data[0])
                global_mean_raws[var_name]["rmse_s_midlat_expt"].append(rmse_s_midlat_expt.data[0])
                global_mean_raws[var_name]["rmse_s_midlat_ctrl"].append(rmse_s_midlat_ctrl.data[0])
                global_mean_raws[var_name]["rmse_s_midlat_diff"].append(rmse_s_midlat_diff.data[0])
                global_mean_raws[var_name]["rmse_n_polar_expt"].append(rmse_n_polar_expt.data[0])
                global_mean_raws[var_name]["rmse_n_polar_ctrl"].append(rmse_n_polar_ctrl.data[0])
                global_mean_raws[var_name]["rmse_n_polar_diff"].append(rmse_n_polar_diff.data[0])
                global_mean_raws[var_name]["rmse_s_polar_expt"].append(rmse_s_polar_expt.data[0])
                global_mean_raws[var_name]["rmse_s_polar_ctrl"].append(rmse_s_polar_ctrl.data[0])
                global_mean_raws[var_name]["rmse_s_polar_diff"].append(rmse_s_polar_diff.data[0])
                for epe_param, epe_param_val in epe_row.items():
                    if epe_param not in global_mean_raws[var_name]:
                        global_mean_raws[var_name][epe_param] = []
                    global_mean_raws[var_name][epe_param].append(epe_param_val)
        print(global_mean_raws)

        with open(gm_raw_file, "w") as f:
            json.dump(global_mean_raws, f, indent=1)
        with open(units_file, "w") as f:
            json.dump(units, f, indent=1)
    with open(gm_raw_file, "r") as f:
        global_mean_raws = json.load(f)
    with open(units_file, "r") as f:
        units = json.load(f)
    for k, v in global_mean_raws.items():
        independent_dependent_variables = [
            ("gm_ctrl", "rmse_diff", {"mean_line", "median_line", "zero_line", "gm_clim"}),
            ("gm_ctrl", "rmse_tropics_diff", {"mean_line", "median_line", "zero_line"}),
            ("gm_ctrl", "mae_diff", {"mean_line", "median_line", "zero_line"}),
            ("gm_diff_diff", "rmse_diff", {"best_fit_line"}),
            ("rmse_ctrl", "rmse_diff", {"mean_line", "zero_line"}),
            ("mae_expt", "mae_ctrl", {"mean_line", "zero_line", "best_fit_line"}),
            ("rmse_ctrl", "mae_diff", {"mean_line", "zero_line"}),
            ("rmse_ctrl", "rmse_expt", {"one_to_one_line", "lock_axes"}),
            ("rmse_ctrl", "rmse_expt", {"one_to_one_line", "lock_axes"}),
            ("mae_ctrl", "mae_expt", {"one_to_one_line", "lock_axes"}),
            ("mae_ctrl", "mae_expt", {"one_to_one_line", "lock_axes"}),
            ("rmse_diff", "rmse_n_subtropics_diff", {"zero_line", "mean_line", "median_line", "lock_axes"}),
            ("rmse_diff", "rmse_s_subtropics_diff", {"zero_line", "mean_line", "median_line", "lock_axes"}),
            ("rmse_diff", "rmse_n_midlat_diff", {"zero_line", "mean_line", "median_line", "lock_axes"}),
            ("rmse_diff", "rmse_s_midlat_diff", {"zero_line", "mean_line", "median_line", "lock_axes"}),
            ("rmse_diff", "rmse_n_polar_diff", {"zero_line", "mean_line", "median_line", "lock_axes"}),
            ("rmse_diff", "rmse_s_polar_diff", {"zero_line", "mean_line", "median_line", "lock_axes"}),
            ("rhcsfc", "rmse_diff", {"best_fit_line", "mean_line", "median_line", "zero_line"}),
            ("reff_liq", "rmse_diff", {"best_fit_line", "mean_line", "median_line", "zero_line"}),
            ("wasserstein_distance_ctrl", "wasserstein_distance_expt", {"one_to_one_line", "lock_axes"}),
            ("wasserstein_distance_ctrl", "wasserstein_distance_diff", {"one_to_one_line", "lock_axes"}),
        ]
        for iv_name, dv_name, opts in independent_dependent_variables:
            dv = v[dv_name]
            if iv_name == "rhcsfc":
                iv = [v["large_scale_cloud_nml"][i]["rhcsfc"] for i in range(len(dv))]
            elif iv_name == "reff_liq":
                iv = [v["cloud_simple_nml"][i]["reff_liq"] for i in range(len(dv))]
            else:
                iv = v[iv_name]
            opts.add("number_points")
            opts.add("conv_scheme_symbols")
            opts.add("cloud_scheme_markers")
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(iv, dv)
            r2 = r_value**2
            best_fit = slope*np.asarray(sorted(iv))+intercept
            title = k
            markers = "."
            annotations = ["" for _ in range(len(iv))]
            if "mean_line" in opts:
                mean = np.mean(dv)
                means = [mean for _ in dv]
                X, Y = sorted(iv), sorted(means)
                plt.plot(X, Y, color="violet", linestyle="-.", label="mean")
                title = f"{title}, mean = {mean:.2f}"
            if "gm_clim" in opts:
                climatology = v["gm_clim"]
                clims = [climatology for _ in iv]
                X, Y = clims, sorted(dv)
                plt.plot(X, Y, color="olive", linestyle="dotted", label="climatology")
            if "median_line" in opts:
                median = np.median(dv)
                medians = [median for _ in dv]
                X, Y = sorted(iv), sorted(medians)
                plt.plot(X, Y, color="purple", linestyle="--", label="median")
                title = f"{title}, median = {median:.2f}"
            if "best_fit_line" in opts:
                X, Y = sorted(iv), best_fit
                plt.plot(X, Y, color="red", linestyle="--", label="best fit")
                title = f"{title}, slope = {slope:.2f}, r2 = {r2:.2f}"
            if "one_to_one_line" in opts:
                X, Y = sorted(iv), sorted(iv)
                plt.plot(X, Y, color="forestgreen", linestyle="solid", label="1:1 line")
            if "lock_axes" in opts:
                plt.axis("equal")
            if "zero_line" in opts:
                plt.plot(sorted(iv), [0.0 for _ in iv], color="black", linestyle="solid", label="zero")
            if "number_points" in opts:
                for i, _ in enumerate(iv):
                    annotations[i] = f"{annotations[i]}{i}"
            if "conv_scheme_symbols" in opts and "cloud_scheme_markers" in opts:
                conv_scheme_ras_mask = np.asarray([v["idealized_moist_phys_nml"][i]["convection_scheme"] == "RAS" for i in
                                        range(len(iv))])
                cs_sbm_mask = np.asarray([not ras for ras in conv_scheme_ras_mask])
                sc_schemes = []
                j = -1
                for i in range(len(iv)):
                    j = j+1 if "do_add_stratocumulus" not in v["cloud_simple_nml"][i] else j
                    sc_schemes.append("NONE" if "do_add_stratocumulus" in v["cloud_simple_nml"][i] else v["marine_strat_cloud_nml"][j]["sc_diag_method"])
                iv = np.asarray(iv)
                dv = np.asarray(dv)
                for scheme, symbol in [("NONE", "x"),("PARK_ELF", "p"),("EIS_WOOD","*"),("SLINGO", "s"),("ECTEI","^"),("LTS","D")]:
                    mask = [sc_scheme == scheme for sc_scheme in sc_schemes]
                    total_ras_mask = conv_scheme_ras_mask & np.asarray(mask)
                    total_sbm_mask = cs_sbm_mask & np.asarray(mask)
                    plt.scatter(iv[total_ras_mask], dv[total_ras_mask], marker=symbol, color="blue", label=f"{scheme}_RAS")
                    plt.scatter(iv[total_sbm_mask], dv[total_sbm_mask], marker=symbol, color="darkorange", label=f"{scheme}_SBM")

            elif "conv_scheme_symbols" in opts:
                conv_scheme_ras_mask = [v["idealized_moist_phys_nml"][i]["convection_scheme"] == "RAS" for i in range(len(iv))]
                cs_sbm_mask = [not ras for ras in conv_scheme_ras_mask]
                iv = np.asarray(iv)
                dv = np.asarray(dv)
                plt.scatter(iv[conv_scheme_ras_mask], dv[conv_scheme_ras_mask], marker="d")
                plt.scatter(iv[cs_sbm_mask], dv[cs_sbm_mask], marker="s")
            else:
                plt.scatter(iv, dv, marker=markers)
            xwidth = np.abs(plt.xlim()[1]-plt.xlim()[0])
            dx = xwidth / 60.
            for i, label in enumerate(annotations):
                plt.annotate(label, (iv[i]+dx, dv[i]))
            plt.xlabel(f"{iv_name} ({units[k]})")
            plt.ylabel(f"{k} {dv_name} ({units[k]})")
            plt.title(title)
            plt.legend(loc="upper left", bbox_to_anchor=(1.05,1.),borderaxespad=0.0)
            save_path = plots_dir.joinpath(f"{dv_name}_vs_{iv_name}_{k}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"saved plot at {save_path}")

if __name__ == "__main__":
    main(regen_gm_raw_file=False)