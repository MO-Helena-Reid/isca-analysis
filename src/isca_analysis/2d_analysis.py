import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import iris
from pathlib import Path
import iris.quickplot as qplt
from iris.analysis.cartography import area_weights


def main():
    print("analysing")
    root_dir = Path(__file__).parents[2]
    out_dir = root_dir.joinpath("out")
    plots_dir = out_dir.joinpath("plots", "feedbacks")
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(out_dir)
    clims = {}
    weights = None
    for filename in out_dir.iterdir():
        if filename.stem[-7:] != "pdc_164":
            continue
        scheme = filename.stem[:-8]
        filename_p4k = filename.with_stem(f"{filename.stem[:-7]}p4k_164")

        variables = {
            "tot_cld_amt",
            "low_cld_amt",
            "mid_cld_amt",
            "high_cld_amt",
            "soc_toa_sw",
            "soc_toa_sw_clr",
            "soc_olr_clr",
            "soc_olr",
            "t_surf",
        }
        cubes = {variable: iris.load_cube(filename, variable) for variable in variables}
        cubes_p4k = {
            variable: iris.load_cube(filename_p4k, variable) for variable in variables
        }
        for state, cl in {"pdc": cubes, "p4k": cubes_p4k}.items():
            cl["lw_cre"] = -(cl["soc_olr"] - cl["soc_olr_clr"])
            print(f"{scheme}: mean lw cre for {state}: {np.mean(cl["lw_cre"].data[:]):.2f}")
            cl["sw_cre"] = cl["soc_toa_sw"] - cl["soc_toa_sw_clr"]
            print(f"{scheme}: mean sw cre for {state}: {np.mean(cl["sw_cre"].data[:]):.2f}")
            cl["net_cre"] = cl["sw_cre"] + cl["lw_cre"]
            print(f"{scheme}: mean net cre for {state}: {np.mean(cl["net_cre"].data[:]):.2f}")
            cl["net_toa"] = cl["soc_toa_sw"] - cl["soc_olr"]
        cubes["t_surf"].coords("latitude")[0].guess_bounds()
        cubes["t_surf"].coords("longitude")[0].guess_bounds()
        weights = (
            area_weights(cubes["t_surf"]) / np.mean(area_weights(cubes["t_surf"]))
            if weights is None
            else weights
        )
        global_surface_t_change = np.mean(
            (cubes_p4k["t_surf"] - cubes["t_surf"]).data * weights
        )
        for variable, cube in cubes.items():
            feedback = (cubes_p4k[variable][0] - cube[0]) / global_surface_t_change
            mean_feedback = np.mean(feedback.data[:] * weights)
            if variable not in clims:
                clims[variable] = {}
                lim = np.max(
                    (np.abs(np.min(feedback.data[:])), np.max(feedback.data[:]))
                )
                clims[variable]["vmin"] = -lim
                clims[variable]["vmax"] = lim
            vmin = clims[variable]["vmin"]
            vmax = clims[variable]["vmax"]
            iris.plot.pcolormesh(feedback, vmin=vmin, vmax=vmax, cmap="PiYG")
            plt.colorbar(orientation="horizontal")
            plt.gca().coastlines()
            plt.title(
                f"{scheme}: {variable} feedback\n aw mean feedback: {mean_feedback:.2f}"
            )
            plot_filename = f"{scheme}_{variable}_feedback.png"
            plot_dir = plots_dir.joinpath(variable)
            plot_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(plot_dir.joinpath(plot_filename))
            print(f"saved plot {plot_filename}")
            plt.close()


if __name__ == "__main__":
    main()
