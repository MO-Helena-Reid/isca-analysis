from pathlib import Path
import iris
import numpy as np
from iris.analysis.cartography import area_weights
import matplotlib.pyplot as plt
import iris.quickplot as qplt


def papillon_first_analysis():
    print("running analysis")
    root_dir = Path(__file__).parents[2]
    out_dir = root_dir.joinpath("out")
    plots_dir = out_dir.joinpath("plots","pdc")
    plots_dir.mkdir(exist_ok=True)
    print(out_dir)
    expt_file = out_dir.joinpath("papillon_fast_pdc_152.nc")
    ctrl_file = out_dir.joinpath("linear_park_pdc_152.nc")
    prec_clim_file = Path("/data/users/helena.reid/GPCP/IMERG-Final.CLIM.200006-202305.V07B.nc4")
    prec_clim_file = prec_clim_file.parent.joinpath("IMERG-Final.CLIM.200006-202305.V07B.coarse.nc")
    prec_clim = iris.load_cube(prec_clim_file,"precipitation")
    print("loaded prec_clim")
    prec_expt = iris.load_cube(expt_file,"precipitation")
    prec_ctrl = iris.load_cube(ctrl_file,"precipitation")
    print("loaded prec_expt")
    ceres_file = Path("/data/users/helena.reid/CERES/CERES_EBAF-TOA_Ed4.2.1_Subset_CLIM01-CLIM12.nc")
    ceres_toa_sw_file = ceres_file.parent.joinpath("CERES_toa_sw_coarse.nc")
    ceres_toa_lw_file = ceres_file.parent.joinpath("CERES_toa_lw_coarse.nc")
    ceres_cc_file = ceres_file.parent.joinpath("CERES_cc_coarse.nc")
    var_dict = {
        "toa_sw_all_clim":
            ["soc_toa_sw",50,350,"PiYG",-250,250,"bwr",-40,40],
        "toa_lw_all_clim":
            ["soc_olr",140,340,"PiYG",-70,70,"bwr",-30,30],
        "cldarea_total_daynight_clim":
            ["tot_cld_amt",0,100,"PiYG",-50,50,"bwr",-20,20],
        "toa_net_all_clim":
        ["toa_net",-120,120,"PiYG",-70,70,"bwr",-30,30],
    }
    for var_,info in var_dict.items():
        soc_var = info[0]
        bias_cmap = info[3]
        bias_vmin = info[4]
        bias_vmax = info[5]
        dbias_cmap = info[6]
        dbias_vmin = info[7]
        dbias_vmax = info[8]

        clim_fine = iris.load_cube(ceres_file,var_)
        print(f"loaded climatology for {var_}")
        if not soc_var == "toa_net":
            expt = iris.load_cube(expt_file,soc_var)
            ctrl = iris.load_cube(ctrl_file,soc_var)
        else:
            expt_toa_sw = iris.load_cube(expt_file,"soc_toa_sw")
            expt_toa_lw = iris.load_cube(expt_file,"soc_olr")
            ctrl_toa_sw = iris.load_cube(ctrl_file,"soc_toa_sw")
            ctrl_toa_lw = iris.load_cube(ctrl_file,"soc_olr")
            expt = expt_toa_sw - expt_toa_lw
            ctrl = ctrl_toa_sw - ctrl_toa_lw
        clim_fine.coord("latitude").guess_bounds()
        clim_fine.coord("longitude").guess_bounds()
        expt.coord("latitude").guess_bounds()
        expt.coord("longitude").guess_bounds()
        clim = clim_fine.regrid(expt,iris.analysis.AreaWeighted())
        clim = clim.collapsed("Climatological Monthly Means Based on 07/2005 to 06/2015",iris.analysis.MEAN)
        print("successfully regridded")
        expt.units = clim.units
        ctrl.units = clim.units
        vmin = info[1]
        vmax = info[2]
        qplt.pcolormesh(expt[0])
        plt.gca().coastlines()
        plt.title(f"experiment {soc_var}")
        plt.clim(vmin=vmin, vmax=vmax)
        plt.savefig(plots_dir.joinpath(f"{soc_var}_experiment.png"))
        plt.close()
        qplt.pcolormesh(ctrl[0])
        plt.gca().coastlines()
        plt.title(f"control {soc_var}")
        plt.clim(vmin=vmin, vmax=vmax)
        plt.savefig(plots_dir.joinpath(f"{soc_var}_control.png"))
        plt.close()
        qplt.pcolormesh(clim)
        plt.gca().coastlines()
        plt.title(f"climatology {var_}")
        plt.clim(vmin=vmin, vmax=vmax)
        plt.savefig(plots_dir.joinpath(f"{soc_var}_climatology.png"))
        plt.close()

        expt_bias = expt[0]-clim
        ctrl_bias = ctrl[0]-clim

        qplt.pcolormesh(expt_bias, cmap=bias_cmap)
        plt.gca().coastlines()
        plt.title(f"experiment - climatology for {soc_var}")
        plt.clim(vmin=bias_vmin, vmax=bias_vmax)
        plt.savefig(plots_dir.joinpath(f"{soc_var}_bias_experiment.png"))
        plt.close()
        qplt.pcolormesh(ctrl_bias, cmap=bias_cmap)
        plt.gca().coastlines()
        plt.title(f"control - climatology for {soc_var}")
        plt.clim(vmin=bias_vmin, vmax=bias_vmax)
        plt.savefig(plots_dir.joinpath(f"{soc_var}_bias_control.png"))
        plt.close()
        expt_ctrl_diff = expt-ctrl
        ecd_vlim = np.max([np.abs(np.min(expt_ctrl_diff.data)),np.abs(np.max(expt_ctrl_diff.data))])
        qplt.pcolormesh(expt_ctrl_diff[0], cmap=bias_cmap)
        plt.gca().coastlines()
        plt.title(f"experiment - control for {soc_var}")
        plt.clim(vmin=-ecd_vlim, vmax=ecd_vlim)
        plt.savefig(plots_dir.joinpath(f"{soc_var}_experiment_minus_control.png"))
        plt.close()

        bias_change = expt_bias - ctrl_bias
        qplt.pcolormesh(bias_change, cmap=dbias_cmap)
        plt.clim(vmin=dbias_vmin, vmax=dbias_vmax)
        plt.gca().coastlines()
        plt.title(f"bias change (expt-ctrl) for {soc_var}")
        plt.savefig(plots_dir.joinpath(f"{soc_var}_bias_delta.png"))
        plt.close()

        abs_bias_change = expt_bias - ctrl_bias
        abs_bias_change.data = np.abs(expt_bias.data) - np.abs(ctrl_bias.data)
        qplt.pcolormesh(abs_bias_change, cmap="bwr")
        plt.gca().coastlines()
        plt.clim(vmin=dbias_vmin, vmax=dbias_vmax)
        plt.title(f"abs bias change (expt-ctrl) for {soc_var}")
        plt.savefig(plots_dir.joinpath(f"{soc_var}_bias_abs_delta.png"))
        plt.close()

        # if var_ == "toa_sw_all_clim":
        #     iris.save(clim_coarse,ceres_toa_sw_file)
        # elif var_ == "toa_lw_all_clim":
        #     iris.save(clim_coarse,ceres_toa_lw_file)
        # elif var_ == "cldarea_total_daynight_clim":
        #     iris.save(clim_coarse,ceres_cc_file)
    # prec_clim.coord("latitude").guess_bounds()
    # prec_expt.coord("latitude").guess_bounds()
    # prec_clim.coord("longitude").guess_bounds()
    # prec_expt.coord("longitude").guess_bounds()
    # prec_clim_coarse = prec_clim.regrid(prec_expt, iris.analysis.AreaWeighted())
    # print("regridded")
    # iris.save(prec_clim_coarse,prec_clim_file.parent.joinpath("IMERG-Final.CLIM.200006-202305.V07B.coarse.nc"))
    s_per_year = 3600*24*365.25
    prec_expt = s_per_year * prec_expt
    prec_ctrl = s_per_year * prec_ctrl
    prec_expt.units = "mm / yr"
    prec_ctrl.units = "mm / yr"


    qplt.pcolormesh(prec_expt[0])
    plt.gca().coastlines()
    plt.title("experiment")
    plt.clim(vmin=0,vmax=7000)
    plt.savefig(plots_dir.joinpath("precip_experiment.png"))
    plt.close()
    qplt.pcolormesh(prec_ctrl[0])
    plt.gca().coastlines()
    plt.title("control")
    plt.clim(vmin=0,vmax=7000)
    plt.savefig(plots_dir.joinpath("precip_control.png"))
    plt.close()
    qplt.pcolormesh(prec_clim)
    plt.gca().coastlines()
    plt.title("climatology")
    plt.clim(vmin=0,vmax=7000)
    plt.savefig(plots_dir.joinpath("precip_climatology.png"))
    plt.close()

    expt_bias = prec_expt - prec_clim
    ctrl_bias = prec_ctrl - prec_clim

    qplt.pcolormesh(expt_bias[0],cmap="BrBG")
    plt.gca().coastlines()
    plt.title("experiment - climatology")
    plt.clim(vmin=-5000,vmax=5000)
    plt.savefig(plots_dir.joinpath("precip_bias_experiment.png"))
    plt.close()
    qplt.pcolormesh(ctrl_bias[0],cmap="BrBG")
    plt.gca().coastlines()
    plt.title("control - climatology")
    plt.clim(vmin=-5000,vmax=5000)
    plt.savefig(plots_dir.joinpath("precip_bias_control.png"))
    plt.close()

    bias_change = expt_bias - ctrl_bias
    qplt.pcolormesh(bias_change[0],cmap="bwr")
    plt.clim(vmin=-2000,vmax=2000)
    plt.gca().coastlines()
    plt.title("bias change (expt-ctrl)")
    plt.savefig(plots_dir.joinpath("precip_bias_delta.png"))
    plt.close()

    abs_bias_change = expt_bias - ctrl_bias
    abs_bias_change.data = np.abs(expt_bias.data) - np.abs(ctrl_bias.data)
    qplt.pcolormesh(abs_bias_change[0],cmap="bwr")
    plt.gca().coastlines()
    plt.clim(vmin=-2000,vmax=2000)
    plt.title("abs bias change (expt-ctrl)")
    plt.savefig(plots_dir.joinpath("precip_bias_abs_delta.png"))
    plt.close()


    expt_ctrl_diff = prec_expt - prec_ctrl
    ecd_vlim = np.max([np.abs(np.min(expt_ctrl_diff.data)), np.abs(np.max(expt_ctrl_diff.data))])
    qplt.pcolormesh(expt_ctrl_diff[0], cmap="BrBG")
    plt.gca().coastlines()
    plt.title(f"experiment - control for precip")
    plt.clim(vmin=-ecd_vlim, vmax=ecd_vlim)
    plt.savefig(plots_dir.joinpath(f"precip_experiment_minus_control.png"))
    plt.close()

if __name__ == "__main__":
    papillon_first_analysis()