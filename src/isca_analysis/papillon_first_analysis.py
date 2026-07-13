#  (C) Crown Copyright, Met Office, 2026
from pathlib import Path
import iris
import numpy as np
from iris.analysis.cartography import area_weights
import matplotlib.pyplot as plt
import iris.quickplot as qplt


def papillon_first_analysis(ctrl_filename, expt_filename):
    ignore_time_coord_errors = True
    print("running analysis")
    root_dir = Path(__file__).parents[2]
    out_dir = root_dir.joinpath("out")
    # plots_dir = out_dir.joinpath("plots","pdc")
    expt_file = out_dir.joinpath(expt_filename)
    ctrl_file = out_dir.joinpath(ctrl_filename)
    plots_dir = out_dir.joinpath("plots","pdc",f"{expt_file.stem}vs_{ctrl_file.stem}")
    plots_dir.mkdir(exist_ok=True)
    print(out_dir)
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
            ["soc_toa_sw_up",0,150,"PiYG",-70,70,"bwr",-40,40],
        "toa_lw_all_clim":
            ["soc_olr",140,340,"PiYG",-70,70,"bwr",-30,30],
        "cldarea_total_daynight_clim":
            ["tot_cld_amt",0,100,"PiYG",-50,50,"bwr",-20,20],
        1:
            ["low_cld_amt", 0, 100, "PiYG", -50, 50, "bwr", -20, 20],
        2:
            ["mid_cld_amt", 0, 100, "PiYG", -50, 50, "bwr", -20, 20],
        3:
            ["high_cld_amt", 0, 100, "PiYG", -50, 50, "bwr", -20, 20],
        4:
            ["t_surf", 250, 340, "coolwarm", -10, 10, "bwr", -10, 10],
        5:
            ["soc_toa_sw_up_clr", 0, 150, "PiYG", -50, 50, "bwr", -20, 20],
        6:
            ["soc_olr_clr", 140,340, "PiYG", -50, 50, "bwr", -20, 20],
        7:
            ["flux_lhe", -200,200, "PiYG", -50, 50, "bwr", -50, 50],
        8:
            ["flux_t", -300,300, "PiYG", -50, 50, "bwr", -50, 50],
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
        obs_present = not isinstance(var_, int)
        clim_fine = iris.load_cube(ceres_file,var_) if obs_present else None
        print(f"loaded climatology for {var_}")
        if not soc_var == "toa_net":
            try:
                expt = iris.load_cube(expt_file,soc_var)
                ctrl = iris.load_cube(ctrl_file,soc_var)
            except iris.exceptions.ConstraintMismatchError:
                continue
        else:
            # too little olr, -15.9
            # too much osw, but too little over sahara, 13.2
            # olr bias bigger
            # net should be off by 2.7 too much going
            # net bias is positive (ie, isw - osw - olr > climatology)
            # we have isw=isw_clim, osw>clim, olr<<olr
            expt_toa_sw = iris.load_cube(expt_file,"soc_toa_sw")
            expt_toa_lw = iris.load_cube(expt_file,"soc_olr")
            ctrl_toa_sw = iris.load_cube(ctrl_file,"soc_toa_sw")
            ctrl_toa_lw = iris.load_cube(ctrl_file,"soc_olr")
            expt = expt_toa_sw - expt_toa_lw
            ctrl = ctrl_toa_sw - ctrl_toa_lw
        if obs_present:
            clim_fine.coord("latitude").guess_bounds()
            clim_fine.coord("longitude").guess_bounds()
        expt.coord("latitude").guess_bounds()
        expt.coord("longitude").guess_bounds()
        ctrl.coord("latitude").guess_bounds()
        ctrl.coord("longitude").guess_bounds()
        if obs_present:
            clim = clim_fine.regrid(expt,iris.analysis.AreaWeighted())
            clim = clim.collapsed("Climatological Monthly Means Based on 07/2005 to 06/2015",iris.analysis.MEAN)
            print("successfully regridded")
            expt.units = clim.units
            ctrl.units = clim.units
        vmin = info[1]
        vmax = info[2]

        grid_areas = iris.analysis.cartography.area_weights(expt)
        area_weighted_global_average = expt.collapsed(['latitude', 'longitude'],
                                                             iris.analysis.MEAN,
                                                             weights=grid_areas)
        awga = np.squeeze(area_weighted_global_average.data)
        qplt.pcolormesh(expt[0])
        plt.gca().coastlines()
        plt.title(f"experiment {soc_var}\nglobal mean {awga:.2f}")
        plt.clim(vmin=vmin, vmax=vmax)
        plt.savefig(plots_dir.joinpath(f"{soc_var}_experiment.png"))
        plt.close()
        grid_areas = iris.analysis.cartography.area_weights(ctrl)
        area_weighted_global_average = ctrl.collapsed(['latitude', 'longitude'],
                                                             iris.analysis.MEAN,
                                                             weights=grid_areas)
        awga = np.squeeze(area_weighted_global_average.data)
        qplt.pcolormesh(ctrl[0])
        plt.gca().coastlines()
        plt.title(f"control {soc_var}\nglobal mean {awga:.2f}")
        plt.clim(vmin=vmin, vmax=vmax)
        plt.savefig(plots_dir.joinpath(f"{soc_var}_control.png"))
        plt.close()
        if obs_present:
            grid_areas = iris.analysis.cartography.area_weights(clim)
            area_weighted_global_average = clim.collapsed(['latitude', 'longitude'],
                                                          iris.analysis.MEAN,
                                                          weights=grid_areas)
            awga = np.squeeze(area_weighted_global_average.data)
            qplt.pcolormesh(clim)
            plt.gca().coastlines()
            plt.title(f"climatology {var_}\nglobal mean {awga:.2f}")
            plt.clim(vmin=vmin, vmax=vmax)
            plt.savefig(plots_dir.joinpath(f"{soc_var}_climatology.png"))
            plt.close()

            expt_bias = expt[0]-clim
            ctrl_bias = ctrl[0]-clim
            grid_areas = iris.analysis.cartography.area_weights(expt_bias)
            area_weighted_global_average = expt_bias.collapsed(['latitude', 'longitude'],
                                                          iris.analysis.MEAN,
                                                          weights=grid_areas)
            awga_expt = np.squeeze(area_weighted_global_average.data)
            grid_areas = iris.analysis.cartography.area_weights(ctrl_bias)
            area_weighted_global_average = ctrl_bias.collapsed(['latitude', 'longitude'],
                                                          iris.analysis.MEAN,
                                                          weights=grid_areas)
            awga_ctrl = np.squeeze(area_weighted_global_average.data)
            qplt.pcolormesh(expt_bias, cmap=bias_cmap)
            plt.gca().coastlines()
            plt.title(f"experiment - climatology for {soc_var}\nglobal mean {awga_expt:.2f}")
            plt.clim(vmin=bias_vmin, vmax=bias_vmax)
            plt.savefig(plots_dir.joinpath(f"{soc_var}_bias_experiment.png"))
            plt.close()
            qplt.pcolormesh(ctrl_bias, cmap=bias_cmap)
            plt.gca().coastlines()
            plt.title(f"control - climatology for {soc_var}\nglobal mean {awga_ctrl:.2f}")
            plt.clim(vmin=bias_vmin, vmax=bias_vmax)
            plt.savefig(plots_dir.joinpath(f"{soc_var}_bias_control.png"))
            plt.close()
        if ignore_time_coord_errors:
            expt.remove_coord("time")
            ctrl.remove_coord("time")
        expt_ctrl_diff = expt-ctrl
        grid_areas = iris.analysis.cartography.area_weights(expt_ctrl_diff)
        area_weighted_global_average = expt_ctrl_diff.collapsed(['latitude', 'longitude'],
                                                             iris.analysis.MEAN,
                                                             weights=grid_areas)
        awga = np.squeeze(area_weighted_global_average.data)
        ecd_vlim = np.max([np.abs(np.min(expt_ctrl_diff.data)),np.abs(np.max(expt_ctrl_diff.data))])
        qplt.pcolormesh(expt_ctrl_diff[0], cmap=bias_cmap)
        plt.gca().coastlines()
        plt.title(f"experiment - control for {soc_var}\nglobal mean {awga:.2f}")
        plt.clim(vmin=-ecd_vlim, vmax=ecd_vlim)
        plt.savefig(plots_dir.joinpath(f"{soc_var}_experiment_minus_control.png"))
        plt.close()
        if obs_present:
            bias_change = expt_bias - ctrl_bias
            grid_areas = iris.analysis.cartography.area_weights(bias_change)
            area_weighted_global_average = bias_change.collapsed(['latitude', 'longitude'],
                                                                                     iris.analysis.MEAN,
                                                                                     weights=grid_areas)
            awga = np.squeeze(area_weighted_global_average.data)
            qplt.pcolormesh(bias_change, cmap=bias_cmap)
            plt.clim(vmin=dbias_vmin, vmax=dbias_vmax)
            plt.gca().coastlines()
            plt.title(f"expt-ctrl for {soc_var},\nglobal mean {awga:.2f}")
            plt.savefig(plots_dir.joinpath(f"{soc_var}_delta.png"))
            plt.close()

            abs_bias_change = expt_bias - ctrl_bias
            abs_bias_change.data = np.abs(expt_bias.data) - np.abs(ctrl_bias.data)

            # abs_bias_change.coord('latitude').guess_bounds()
            # abs_bias_change.coord('longitude').guess_bounds()
            grid_areas = iris.analysis.cartography.area_weights(abs_bias_change)
            area_weighted_global_average_abs_bias_change = abs_bias_change.collapsed(['latitude', 'longitude'],
                                                                                     iris.analysis.MEAN, weights=grid_areas)
            awgaabc = np.squeeze(area_weighted_global_average_abs_bias_change.data)
            # awgaabc = None
            qplt.pcolormesh(abs_bias_change, cmap="bwr")
            plt.gca().coastlines()
            plt.clim(vmin=dbias_vmin, vmax=dbias_vmax)
            plt.title(f"abs bias change (expt-ctrl) for {soc_var},\nglobal mean {awgaabc:.2f}")
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
    prec_ctrl.coord("latitude").guess_bounds()
    prec_ctrl.coord("longitude").guess_bounds()
    prec_expt.coord("latitude").guess_bounds()
    prec_expt.coord("longitude").guess_bounds()
    # prec_clim.coord("latitude").guess_bounds()
    # prec_clim.coord("longitude").guess_bounds()

    grid_areas = iris.analysis.cartography.area_weights(prec_expt)
    area_weighted_global_average = prec_expt.collapsed(['latitude', 'longitude'],
                                                         iris.analysis.MEAN,
                                                         weights=grid_areas)
    awga_expt = np.squeeze(area_weighted_global_average.data)
    grid_areas = iris.analysis.cartography.area_weights(prec_ctrl)
    area_weighted_global_average = prec_ctrl.collapsed(['latitude', 'longitude'],
                                                         iris.analysis.MEAN,
                                                         weights=grid_areas)
    awga_ctrl = np.squeeze(area_weighted_global_average.data)
    grid_areas = iris.analysis.cartography.area_weights(prec_clim)
    area_weighted_global_average = prec_clim.collapsed(['latitude', 'longitude'],
                                                         iris.analysis.MEAN,
                                                         weights=grid_areas)
    awga_clim = np.squeeze(area_weighted_global_average.data)
    qplt.pcolormesh(prec_expt[0])
    plt.gca().coastlines()
    plt.title(f"experiment\nglobal mean {awga_expt:.2f}")
    precip_vmax = 5000
    dprecip_vmax = 5000
    plt.clim(vmin=0, vmax=precip_vmax)
    plt.savefig(plots_dir.joinpath("precip_experiment.png"))
    plt.close()
    qplt.pcolormesh(prec_ctrl[0])
    plt.gca().coastlines()
    plt.title(f"control\nglobal mean {awga_ctrl:.2f}")
    plt.clim(vmin=0, vmax=precip_vmax)
    plt.savefig(plots_dir.joinpath("precip_control.png"))
    plt.close()
    qplt.pcolormesh(prec_clim)
    plt.gca().coastlines()
    plt.title(f"climatology\nglobal mean {awga_clim:.2f}")
    plt.clim(vmin=0, vmax=precip_vmax)
    plt.savefig(plots_dir.joinpath("precip_climatology.png"))
    plt.close()

    expt_bias = prec_expt - prec_clim
    ctrl_bias = prec_ctrl - prec_clim
    grid_areas = iris.analysis.cartography.area_weights(expt_bias)
    area_weighted_global_average = expt_bias.collapsed(['latitude', 'longitude'],
                                                         iris.analysis.MEAN,
                                                         weights=grid_areas)
    awga_expt = np.squeeze(area_weighted_global_average.data)
    grid_areas = iris.analysis.cartography.area_weights(ctrl_bias)
    area_weighted_global_average = ctrl_bias.collapsed(['latitude', 'longitude'],
                                                         iris.analysis.MEAN,
                                                         weights=grid_areas)
    awga_ctrl = np.squeeze(area_weighted_global_average.data)
    qplt.pcolormesh(expt_bias[0],cmap="BrBG")
    plt.gca().coastlines()
    plt.title(f"experiment - climatology\nglobal mean {awga_expt:.2f}")
    plt.clim(vmin=-dprecip_vmax, vmax=dprecip_vmax)
    plt.savefig(plots_dir.joinpath("precip_bias_experiment.png"))
    plt.close()
    qplt.pcolormesh(ctrl_bias[0],cmap="BrBG")
    plt.gca().coastlines()
    plt.title(f"control - climatology\nglobal mean {awga_ctrl:.2f}")
    plt.clim(vmin=-dprecip_vmax, vmax=dprecip_vmax)
    plt.savefig(plots_dir.joinpath("precip_bias_control.png"))
    plt.close()

    vmax_prec = 500
    if ignore_time_coord_errors:
        expt_bias.remove_coord("time")
        ctrl_bias.remove_coord("time")
    bias_change = expt_bias - ctrl_bias
    grid_areas = iris.analysis.cartography.area_weights(bias_change)
    area_weighted_global_average = bias_change.collapsed(['latitude', 'longitude'],
                                                         iris.analysis.MEAN,
                                                         weights=grid_areas)
    awga = np.squeeze(area_weighted_global_average.data)
    qplt.pcolormesh(bias_change[0],cmap="BrBG")
    plt.clim(vmin=-vmax_prec,vmax=vmax_prec)
    plt.gca().coastlines()
    plt.title(f"expt-ctrl\nglobal mean {awga:.2f}")
    plt.savefig(plots_dir.joinpath("precip_delta.png"))
    plt.close()


    abs_bias_change = expt_bias - ctrl_bias
    abs_bias_change.data = np.abs(expt_bias.data) - np.abs(ctrl_bias.data)
    grid_areas = iris.analysis.cartography.area_weights(abs_bias_change)
    area_weighted_global_average_abs_bias_change = abs_bias_change.collapsed(['latitude', 'longitude'],
                                                                             iris.analysis.MEAN, weights=grid_areas)
    awgaabc = np.squeeze(area_weighted_global_average_abs_bias_change.data)
    qplt.pcolormesh(abs_bias_change[0],cmap="bwr")
    plt.gca().coastlines()
    plt.clim(vmin=-vmax_prec,vmax=vmax_prec)
    plt.title(f"bias change (expt-ctrl)\nglobal mean {awgaabc:.2f}")
    plt.savefig(plots_dir.joinpath("precip_bias_abs_delta.png"))
    plt.close()

    if ignore_time_coord_errors:
        prec_expt.remove_coord("time")
        prec_ctrl.remove_coord("time")
    expt_ctrl_diff = prec_expt - prec_ctrl
    grid_areas = iris.analysis.cartography.area_weights(expt_ctrl_diff)
    area_weighted_global_average = expt_ctrl_diff.collapsed(['latitude', 'longitude'],
                                                         iris.analysis.MEAN,
                                                         weights=grid_areas)
    awga = np.squeeze(area_weighted_global_average.data)
    ecd_vlim = np.max([np.abs(np.min(expt_ctrl_diff.data)), np.abs(np.max(expt_ctrl_diff.data))])
    print(f"ecd vlim: {ecd_vlim}")
    # ecd_vlim = 100
    qplt.pcolormesh(expt_ctrl_diff[0], cmap="BrBG")
    plt.gca().coastlines()
    plt.title(f"experiment - control for precip\nglobal mean {awga:.2f}")
    plt.clim(vmin=-ecd_vlim, vmax=ecd_vlim)
    plt.savefig(plots_dir.joinpath(f"precip_experiment_minus_control.png"))
    plt.close()

if __name__ == "__main__":
    control_files = [
        # "papillon_control_with_clouds_120.nc",
        # "papillon_control_with_clouds_0060.nc",
        # "papillon_control_smith_ras_0120.nc",
        # "papillon_control_with_clouds_0036.nc",
        # "papillon_control_with_clouds_0048.nc",
        # "papillon_control_0240.nc",
        "papillon_vertical_1km_0120.nc",
        # "papillon_na_1_0_0048.nc",
    ]
    exp_files = [
                    # "papillon_random_vertical_random_time_120.nc",
                    # "papillon_random_vertical_3h_time_120.nc",
                    # "papillon_random_vertical_120.nc",
                    # "papillon_uniform_vertical_120.nc",
                    # "papillon_na_2_0_120.nc",
                    # "papillon_na_1_0_120.nc",
                    # "papillon_na_0_1_120.nc",
                    # "papillon_control_smith_0120.nc",
                    # "papillon_control_ras_0120.nc",
                    # "papillon_ml_0036.nc",
        # "papillon_smith_ras_0120.nc",
        # "papillon_control_slingo_0120.nc",
        # "papillon_vertical_1km_0120.nc",
        "papillon_vertical_1km_0240.nc",
        # "papillon_ml_0048.nc",
        # "papillon_ml_0060.nc",
                  ]
    for file1 in control_files:
        for file2 in exp_files:
            papillon_first_analysis(ctrl_filename=file1, expt_filename=file2)