import json
from pathlib import Path
import iris
import numpy as np
import scipy
from matplotlib import pyplot as plt


def calculate_global_mean(cube):
    cube.coord("latitude").guess_bounds()
    cube.coord("longitude").guess_bounds()
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
    # we want to know how much a given parameter explains variability in a given variable
    root_dir = Path(__file__).parents[2]
    out_dir = root_dir.joinpath("data")
    isca_means_dir = out_dir.joinpath("isca_means")
    plots_dir = out_dir.joinpath("plots", "epe")
    plots_dir.mkdir(exist_ok=True)
    prec_clim_file = out_dir.joinpath("GPCP/IMERG-Final.CLIM.200006-202305.V07B.coarse.nc")
    ceres_file = out_dir.joinpath("CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_CLIM01-CLIM12.nc")
    sbm_ctrl_file = isca_means_dir.joinpath("papillon_control_with_clouds_0060.nc")
    ras_ctrl_file = isca_means_dir.joinpath("papillon_control_ras_0060.nc")
    epe_conf_file = out_dir.joinpath("epe.json")
    with open(epe_conf_file, "r") as f:
        epe_conf = json.load(f)

    global_mean_raws = {
        "precipitation":[],
        "low_cld_amt":[],
        "mid_cld_amt":[],
        "high_cld_amt":[],
        "soc_olr":[],
        "soc_toa_sw_up":[],
        "tot_cld_amt":[],
    }

    independent_variables = {
        "reff":[],
        "rhc":[],
        # "rh_trig":[],
        # "alm_min":[],
        # "tau_bm":[],
        # "rhbm":[],
    }

    units = {}

    epe_size = 25
    for i_epe in range(epe_size):
        if i_epe == 13 or i_epe == 18:
            continue
        expt_file = isca_means_dir.joinpath(f"epe{i_epe:03d}_0060.nc")
        cubes = iris.load_cubes(expt_file, global_mean_raws)
        epe_row = epe_conf[i_epe]
        for cube in cubes:
            gm = calculate_global_mean(cube)
            units[gm.var_name] = gm.units
            global_mean_raws[gm.var_name].append(gm.data[0])
        for iv_key in independent_variables.keys():
            independent_variables[iv_key].append(extract_independent_variable(epe_row, iv_key))
    print(global_mean_raws)
    print(independent_variables)

    for iv_name, iv in independent_variables.items():
        for dv_name, dv in global_mean_raws.items():
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(iv, dv)
            r2 = r_value**2
            best_fit = slope*np.asarray(iv)+intercept
            plt.scatter(iv, dv)
            plt.plot(iv, best_fit, color="red")
            plt.xlabel(f"{iv_name} (normalised units)")
            plt.ylabel(f"{dv_name} ({units[dv_name]})")
            plt.title(f"slope = {slope:.2f}, r2 = {r2:.2f}")
            save_path = plots_dir.joinpath(f"{dv_name}_vs_{iv_name}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"saved plot at {save_path}")
    prec_clim = iris.load_cube(prec_clim_file, "precipitation")
    print("loaded prec_clim")

if __name__ == "__main__":
    main()