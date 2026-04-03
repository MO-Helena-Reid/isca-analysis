from pathlib import Path
import iris
import numpy as np
from iris.analysis.cartography import area_weights
import matplotlib.pyplot as plt
import iris.quickplot as qplt


def main():
    print("running analysis")
    root_dir = Path(__file__).parents[2]
    out_dir = root_dir.joinpath("out")
    print(out_dir)
    feedbacks = {}
    for filename in out_dir.iterdir():
        if filename.stem[-7:] != "pdc_164":
            continue
        scheme = filename.stem[:-8]
        filename_p4k = filename.with_stem(f"{filename.stem[:-7]}p4k_164")
        tcc = iris.load_cube(filename, "tot_cld_amt")
        lcc = iris.load_cube(filename, "low_cld_amt")
        mcc = iris.load_cube(filename, "mid_cld_amt")
        hcc = iris.load_cube(filename, "high_cld_amt")
        toa_sw = iris.load_cube(filename, "soc_toa_sw")
        toa_sw_clr = iris.load_cube(filename, "soc_toa_sw_clr")
        toa_lw = iris.load_cube(filename, "soc_olr")
        toa_lw_clr = iris.load_cube(filename, "soc_olr_clr")

        tcc_p4k = iris.load_cube(filename_p4k, "tot_cld_amt")
        lcc_p4k = iris.load_cube(filename_p4k, "low_cld_amt")
        mcc_p4k = iris.load_cube(filename_p4k, "mid_cld_amt")
        hcc_p4k = iris.load_cube(filename_p4k, "high_cld_amt")
        toa_sw_p4k = iris.load_cube(filename_p4k, "soc_toa_sw")
        toa_sw_clr_p4k = iris.load_cube(filename_p4k, "soc_toa_sw_clr")
        toa_lw_p4k = iris.load_cube(filename_p4k, "soc_olr")
        toa_lw_clr_p4k = iris.load_cube(filename_p4k, "soc_olr_clr")
        # print(f"{tcc}\n{lcc}\n{mcc}\n{hcc}")
        t_surf = iris.load_cube(filename, "t_surf")
        t_surf_p4k = iris.load_cube(filename_p4k, "t_surf")
        t_surf.coords("latitude")[0].guess_bounds()
        t_surf.coords("longitude")[0].guess_bounds()
        weights = area_weights(t_surf)
        weights = weights/np.mean(weights)
        weighted_t_surf = t_surf.data*weights
        weighted_t_surf_p4k = t_surf_p4k.data*weights
        global_surface_t_change = np.mean(weighted_t_surf_p4k) - np.mean(weighted_t_surf)
        print(f"{global_surface_t_change=}")

        lw_cre = - (toa_lw - toa_lw_clr)
        sw_cre = toa_sw - toa_sw_clr
        sw_cre_p4k = toa_sw_p4k - toa_sw_clr_p4k
        lw_cre_p4k = - (toa_lw_p4k - toa_lw_clr_p4k)
        net_cre = sw_cre + lw_cre
        net_cre_p4k = sw_cre_p4k + lw_cre_p4k
        net_toa = toa_sw - toa_lw
        net_toa_p4k = toa_sw_p4k - toa_lw_p4k
        cloud_vars = [
            # ("tcc", tcc, tcc_p4k),
            # ("lcc", lcc, lcc_p4k),
            # ("mcc", mcc, mcc_p4k),
            # ("hcc", hcc, hcc_p4k),
            # ("net", net_toa, net_toa_p4k),
            # ("sw", toa_sw, toa_sw_p4k),
            # ("lw", toa_lw, toa_lw_p4k),
            # ("clear-sky sw", toa_sw_clr, toa_sw_clr_p4k),
            # ("clear-sky lw", toa_lw_clr, toa_lw_clr_p4k),
            ("sw_cre", sw_cre, sw_cre_p4k),
            ("lw_cre", lw_cre, lw_cre_p4k),
            ("net_cre", net_cre, net_cre_p4k),
        ]
        print(f"=== PROCESSING {scheme} ===")
        feedbacks[scheme] = {}
        for varname, cube, cube_p4k in cloud_vars:
            print(f"PROCESSING VAR {str.upper(varname)}")
            feedbacks[scheme][varname] = {}
            non_aw_mean = np.mean(cube.data)
            non_aw_p4k_mean = np.mean(cube_p4k.data)
            aw_mean = np.mean(cube.data * weights)
            aw_p4k_mean = np.mean(cube_p4k.data * weights)
            # print(f"mean {varname} pdc: {non_aw_mean :.1f}")
            # print(f"p4k mean {varname}: {non_aw_p4k_mean :.1f}")
            # print(f"mean aw {varname} pdc: {aw_mean :.1f}")
            # print(f"p4k mean aw {varname}: {aw_p4k_mean :.1f}")
            diff = cube_p4k - cube
            area_weighted_diff = diff.data * weights
            # diff_per_global_t_surf_change = np.mean(area_weighted_diff)
            # print(f"{varname} diff per {global_surface_t_change:.2f}K: {diff_per_global_t_surf_change}")
            feedback = np.mean(area_weighted_diff / global_surface_t_change)
            feedbacks[scheme][varname]["non_aw_pdc_mean"] = non_aw_mean
            feedbacks[scheme][varname]["non_aw_p4k_mean"] = non_aw_p4k_mean
            feedbacks[scheme][varname]["aw_pdc_mean"] = aw_mean
            feedbacks[scheme][varname]["aw_p4k_mean"] = aw_p4k_mean
            feedbacks[scheme][varname]["non_aw_diff"] = diff
            feedbacks[scheme][varname]["aw_diff"] = area_weighted_diff
            feedbacks[scheme][varname]["feedback"] = feedback
    print(f"====== DONE PROCESSING ======")
    for scheme in feedbacks:
        print(f"=== {scheme} ===")
        for varname in feedbacks[scheme]:
            # for variable in ["aw_pdc_mean", "aw_p4k_mean", "feedback"]:
            print(f"{varname} pdc: {feedbacks[scheme][varname]["aw_pdc_mean"]:.1f}\tp4k: {feedbacks[scheme][varname]["aw_p4k_mean"]:.1f}\tfeedback: {feedbacks[scheme][varname]["feedback"]:.2f}")
    cloud_vars = ["sw_cre", "lw_cre", "net_cre"]
    vars_ = ["aw_pdc_mean", "aw_p4k_mean", "feedback"]
    print(f"#table(\n\tcolumns: {len(vars_)+2},\n\t[*scheme*]\t[*{cloud_vars[0]} (pdc)*],\t[*{cloud_vars[1]} (pdc)*],\t[*{cloud_vars[2]} (pdc)*],\t[*feedback*]")
    for scheme in feedbacks:
            # for variable in ["aw_pdc_mean", "aw_p4k_mean", "feedback"]:
            print(f"\t{scheme.replace("_no_sc", "")},\t{feedbacks[scheme][cloud_vars[0]]["aw_pdc_mean"]:.1f},\t{feedbacks[scheme][cloud_vars[1]]["aw_pdc_mean"]:.1f},\t{feedbacks[scheme][cloud_vars[2]]["aw_pdc_mean"]:.1f},\t{feedbacks[scheme][varname]["feedback"]:.2f},")
    print(")")
if __name__=="__main__":
    main()
