import numpy as np
from pyrfu import pyrf
from .acceleration_terms import acceleration_terms
from .acceleration_terms2 import acceleration_terms2
from pyrfu import mms
import pickle
from copy import copy
from copy import deepcopy
from .calc_l import calc_l
from .temp_ratio import temp_ratio
from .inp_time2ti import inp_time2ti
from .inp_time2b import inp_time2b
from .df_tint import (load_b_field, load_n_t_4sc)
from .fpi_flux_energy import fpi_flux_energy
from .feeps_flux_energy import feeps_flux_energy
from .fit_dfs import fit_dfs
from .ps_distance_proxy import ps_distance_proxy
from .fpi_para_perp import fpi_para_perp
from .feeps_para_perp import feeps_para_perp
from pyrfu.mms import feeps_corrections
from pyrfu.mms import feeps_remove_sunlit_sectors

def epoch_analysis(tints, window_ti = 5, scale_with:str = "t_i", 
                   energies_fpi = [0.2, 1, 4, 10, 20, 27], roll_t:int =1 ,roll:int = 15, roll_f = 30, 
                   roll_feeps = 3, do_fit = True, fit_manually:bool = False, 
                   fit_path = None, energy_ranges_feeps = [[32.1, 50], [50, 200]], xs_cond = None):
    r"""
    Function to perform epoch analysis on a list of time intervals.
    The function will save the results in a pickle file.

    Parameters:
    -----------
    tints: list
        List of time intervals to perform the epoch analysis on.
    window_ti: float
        Halfsize of the window in t_i.
    scale_with: str
        Scale the time intervals with "t_i" or "b".
    energies_fpi: list
        List of energies for the FPI data at which to get omnidirectional and 
        fperp/fpara.
    roll: int
        Rolling window size for the betatron and Fermi acceleration terms.
    roll_f: int
        Rolling window size for the FPI data.
    roll_feeps: int
        Rolling window size for the FEEPS data.
    do_fit: bool
        If True, the DFs will be fitted with a hyperbolic tangent.
    fit_manually: bool
        If True, the user will be prompted to fit the DFs manually.
    fit_path: str
        Path to the file containing the fitted DFs.
    energy_ranges_feeps: list
        List of energy ranges for the FEEPS data at which to get
        omnidirectional and fperp/fpara.
        
    Returns:
    --------
    dict_ea: dict
        Dictionary containing the results of the epoch analysis.

    """

    assert scale_with in ["t_i", "b"], "scale_with must be either 't_i' or 'b'."

    if scale_with == "t_i":
        time_conv = inp_time2ti
        ss = "ti"
    else:
        time_conv = inp_time2b
        ss = "b"

    dict_ea = {it: {} for it in range(len(tints))}

    dict_fit = {it: {} for it in range(len(tints))}

    if do_fit:

        dict_fit = fit_dfs(tints, fit_manually = fit_manually, fit_path = None)

    else:
        
        with open(fit_path, "rb") as f:
            dict_fit = pickle.load(f)
    
    if xs_cond is not None:

        dict_ea_xs0 = {}
        dict_ea_xs1 = {}

        dict_fit_xs0 = {}
        dict_fit_xs1 = {}

    for it, tint in enumerate(tints):
        
        print(f"{it+1}/{len(tints)}")
        print(f"Processing df at: {tint}")

        if xs_cond is not None:

            R_E = 6378.136
            
            rs_gsm = pyrf.avg_4sc([mms.get_data(f'r_gsm_mec_srvy_l2', tint, mms_id, verbose = False) for mms_id in range(1,5)]) / R_E
            if b'x' in rs_gsm.coords["comp"]:
                xs_gsm = rs_gsm.sel(comp = b'x').mean(skipna = True)
                ys_gsm = rs_gsm.sel(comp = b'y').mean(skipna = True)
                zs_gsm = rs_gsm.sel(comp = b'z').mean(skipna = True)
            else:
                xs_gsm = rs_gsm.sel(comp = b'x  ').mean(skipna = True)
                zs_gsm = rs_gsm.sel(comp = b'z  ').mean(skipna = True)
                ys_gsm = rs_gsm.sel(comp = b'y  ').mean(skipna = True)

            if xs_gsm < xs_cond:
                dict_ea_xs0[it] = {}
                dict_ea = dict_ea_xs0
                # dict_fit_xs0[it] = {}
                # dict_fit = dict_fit_xs0
            else:
                dict_ea_xs1[it] = {}
                dict_ea = dict_ea_xs1
                # dict_fit_xs1[it] = {}
                # dict_fit = dict_fit_xs1

        window_half_width = 10
        t_df_init, b_syn, a, b, c, sigma, t_df, b_filt, pcov = dict_fit[it].values()
        # Convert b to a timedelta object
        b_timedelta = np.timedelta64(int(abs(b * 1e9)), "ns")

        tint_new = deepcopy(tint)

        while True:

            tint_start, tint_end = tint_new

            _, b_avg = load_b_field(tint)
            dbz_dt = pyrf.sliding_derivative(b_avg.sel(represent_vec_tot = "z"), t_units= "ns", method = "5ps")

            Vi_gse_part = [mms.get_data(f'partvi_gse_fpi_brst_l2', tint=tint, mms_id=id, verbose=False) for id in range(1, 5)]

            idx_energy_cut = [np.where(Vi_gse_part[i].lowest_included_energy > 1e2)[0][0] for i in range(4)]
            Vigse = [Vi_gse_part[i][:, idx_energy_cut[i], :] for i in range(4)]
            # V_i_avg = pyrf.ts_convolve(pyrf.avg_4sc(Vigse), np.ones(6)/6)
            V_i_avg = pyrf.avg_4sc(Vigse)
            V_i_avg = pyrf.resample(V_i_avg, b_avg)
            
            n_i_avg, _ = load_n_t_4sc(tint, s = "i", lowest_energy=1e2)
            # n_i_avg = pyrf.ts_convolve(n_i_avg, kernel = np.ones(6)/6)
            
            l_i = calc_l(n_i_avg, b_avg, s = "i")

            t_i = l_i * 1e-3 / pyrf.norm(V_i_avg)

            df_timescale = [str((t_df[0] - window_half_width * b_timedelta)), str((t_df[0] + window_half_width *b_timedelta))]

            if scale_with == "t_i":
                t_i_before_median = t_i.sel(time = slice(df_timescale[0], t_df[0])).median().data


                if np.datetime64(tint_start) > t_df[0] - window_ti*np.timedelta64(int(t_i_before_median* 1e9), "ns"):

                        tint_start = (t_df[0] - window_ti*np.timedelta64(int(t_i_before_median* 1e9), "ns")).astype(str)

                if np.datetime64(tint_end) < t_df[0] + window_ti*np.timedelta64(int(t_i_before_median* 1e9), "ns"):

                        tint_end = (t_df[0] + window_ti*np.timedelta64(int(t_i_before_median* 1e9), "ns")).astype(str)
            elif scale_with == "b":

                if np.datetime64(tint_start) > t_df[0] - window_ti*b_timedelta:
                    
                        tint_start = (t_df[0] - window_ti*b_timedelta).astype(str)

                if np.datetime64(tint_end) < t_df[0] + window_ti*b_timedelta:

                        tint_end = (t_df[0] + window_ti*b_timedelta).astype(str)

            tint_new = [tint_start, tint_end]

            break
        
        if tint_new != tint:

            print(f"New tint: {tint_new}")

            _, b_avg = load_b_field(tint_new)
            dbz_dt = pyrf.sliding_derivative(b_avg.sel(represent_vec_tot = "z"), t_units= "ns", method = "5ps")

            Vi_gse_part = [mms.get_data(f'partvi_gse_fpi_brst_l2', tint=tint_new, mms_id=id, verbose=False) for id in range(1, 5)]

            idx_energy_cut = [np.where(Vi_gse_part[i].lowest_included_energy > 1e2)[0][0] for i in range(4)]
            Vigse = [Vi_gse_part[i][:, idx_energy_cut[i], :] for i in range(4)]
            # V_i_avg = pyrf.ts_convolve(pyrf.avg_4sc(Vigse), np.ones(6)/6)
            V_i_avg = pyrf.avg_4sc(Vigse)
            V_i_avg = pyrf.resample(V_i_avg, b_avg)

            n_i_avg, _ = load_n_t_4sc(tint_new, s = "i", lowest_energy=1e2)
            # n_i_avg = pyrf.ts_convolve(n_i_avg, kernel = np.ones(6)/6)
            l_i = calc_l(n_i_avg, b_avg, s = "i")
            t_i = l_i * 1e-3 / pyrf.norm(V_i_avg)

        partp_e = [mms.get_data(f"partpe_gse_fpi_brst_l2", tint_new, id, verbose=False) for id in range(1, 5)]
        idx_energy_cut = [np.where(partp_e[i].lowest_included_energy > 1e2)[0][0] for i in range(4)]
        p_e = [partp_e[i][:, idx_energy_cut[i]] for i in range(4)]
        p_e_avg = pyrf.avg_4sc(p_e)

        partp_i = [mms.get_data(f"partpi_gse_fpi_brst_l2", tint_new, id, verbose=False) for id in range(1, 5)]
        idx_energy_cut = [np.where(partp_i[i].lowest_included_energy > 1e2)[0][0] for i in range(4)]
        p_i = [partp_i[i][:, idx_energy_cut[i]] for i in range(4)]
        p_i_avg = pyrf.avg_4sc(p_i)

        p_e_avg = pyrf.resample(p_e_avg, p_i_avg)
        p_tot_avg = p_e_avg + p_i_avg
        p_tot_avg.attrs = p_e_avg.attrs

        ps_distance = ps_distance_proxy(b_avg, p_tot_avg)
        ps_distance_alt = ps_distance_proxy(b_avg, p_tot_avg, alt=True)
###### ---------------------------------------------------- ######

# Load acceleration terms
        acc_terms = acceleration_terms2(tint_new, verbose=False, part_moments_en=1e2, 
                                    kappa_limit=3, B_smooth_window=5)
        
        W_betatron, W_fermi, w_betatron, w_fermi, w_betatron_cumsum, w_fermi_cumsum, kappa_roll, R_g, kappa_extra_energy = acc_terms

# Load FPI normalized flux and perp/para flux ratio
        f_en_perp_para = fpi_flux_energy(tint = tint_new, s = "e", flag = "ratio", 
                                         energies = energies_fpi)
        f_en_omni = fpi_flux_energy(tint = tint_new, s = "e", flag = "omni", 
                                    energies = energies_fpi)
        f_en_omni = [f_en_omni[i].rolling(time = roll_f, center=True).median()
                      for i in range(len(f_en_omni))]
        f_en_omni = [f_en_omni[i]/f_en_omni[i].mean(skipna=True)
                      for i in range(len(f_en_omni))]

        f_en_perp_para = [f_en_perp_para[i].rolling(time = roll_f, center=True).median() 
                          for i in range(len(f_en_perp_para))]
        
        f_en_para0, f_en_para180, f_en_perp = fpi_para_perp(tint_new, s = "e", energies = energies_fpi)
        # f_en_perp_para = [2 * f_en_perp[i]/ (f_en_para0[i] + f_en_para180[i]) for i in range(len(energies_fpi))]
        # for i in range(len(f_en_omni)):
        #     f_en_omni[i][0] = f_en_omni[i][0].rolling(time = roll_f, center=True).median()
        #     f_en_omni[i][1] = f_en_omni[i][1].rolling(time = roll_f, center=True).median()

        for i in range(len(f_en_perp)):
            f_en_para0[i] = f_en_para0[i].rolling(time = roll_f, center=True).median()
            f_en_para180[i] = f_en_para180[i].rolling(time = roll_f, center=True).median()
            f_en_perp[i] = f_en_perp[i].rolling(time = roll_f, center=True).median()

        f_en_para0 = [f_en_para0[i]/f_en_para0[i].mean(skipna=True)
                        for i in range(len(f_en_para0))]
        f_en_para180 = [f_en_para180[i]/f_en_para180[i].mean(skipna=True)
                        for i in range(len(f_en_para180))]
        f_en_perp = [f_en_perp[i]/f_en_perp[i].mean(skipna=True)
                        for i in range(len(f_en_perp))]
# Load FEEPS normalized flux and perp/para flux ratio
        feeps_energy_centroids = [(energy_range[1] - energy_range[0])/(np.log(energy_range[1]) - np.log(energy_range[0])) for energy_range in energy_ranges_feeps]

        try:
            no_feeps = False

            feeps_alle_e = [mms.get_feeps_alleyes(f"fluxe_brst_l2", tint_new, mms_id, verbose = False) for mms_id in range(1, 5)]
            feeps_e_corr_new = [feeps_corrections(deepcopy(feeps_alle_e[i])) for i in range(len(feeps_alle_e))]
            feeps_e_corr_new = [feeps_remove_sunlit_sectors(feeps_e_corr_new[i],) for i in range(len(feeps_e_corr_new))]

            feeps_en_perp_para = feeps_flux_energy(feeps_e_corr_new = feeps_e_corr_new, tint = tint_new, s = "e", flag = "ratio",
                                                energy_ranges = energy_ranges_feeps)
            feeps_en_omni = feeps_flux_energy(feeps_e_corr_new = feeps_e_corr_new, tint = tint_new, s = "e", flag = "omni",
                                            energy_ranges = energy_ranges_feeps)
            
            feeps_en_para0, feeps_en_para180, feeps_en_perp = feeps_para_perp(feeps_e_corr_new = feeps_e_corr_new, tint = tint_new, s = "e",
                                                                        energy_ranges = energy_ranges_feeps)
            
            if roll_feeps != None:
                
                feeps_en_perp_para = [feeps_en_perp_para[i].rolling(time = roll_feeps, center=True).median()
                                        for i in range(len(feeps_en_perp_para))]
                feeps_en_para0 = [feeps_en_para0[i].rolling(time = roll_feeps, center=True).median()
                                    for i in range(len(feeps_en_para0))]
                feeps_en_para180 = [feeps_en_para180[i].rolling(time = roll_feeps, center=True).median()
                                    for i in range(len(feeps_en_para180))]
                feeps_en_perp = [feeps_en_perp[i].rolling(time = roll_feeps, center=True).median()
                                    for i in range(len(feeps_en_perp))]
                
                feeps_en_omni = [feeps_en_omni[i].rolling(time = roll_feeps, center=True).median()
                                    for i in range(len(feeps_en_omni))]
                
                # for i in range(len(feeps_en_omni)):
                #     feeps_en_omni[i][0] = feeps_en_omni[i][0].rolling(time = roll_feeps, center=True).median()
                #     feeps_en_omni[i][1] = feeps_en_omni[i][1].rolling(time = roll_feeps, center=True).median()
            
            feeps_en_para0 = [feeps_en_para0[i]/feeps_en_para0[i].mean(skipna=True)
                                    for i in range(len(feeps_en_para0))]
            feeps_en_para180 = [feeps_en_para180[i]/feeps_en_para180[i].mean(skipna=True)
                                    for i in range(len(feeps_en_para180))]
            feeps_en_perp = [feeps_en_perp[i]/feeps_en_perp[i].mean(skipna=True)
                                    for i in range(len(feeps_en_perp))]
            feeps_en_omni = [feeps_en_omni[i]/feeps_en_omni[i].mean(skipna=True)
                                    for i in range(len(feeps_en_omni))]
                
        except:
            print("FEEPS data not available.")
            no_feeps = True

        try:
            no_hpca = False
            Vi_hpca = []
            for i in range(1, 5):
                try:
                    Vi_hpca.append(mms.get_data(f"vhplus_gsm_hpca_srvy_l2", tint_new, i, verbose = False))
                except:
                    print(f"HPCA data for MMS {i} not available")

            # Vi_hpca = [mms.get_data(f"vhplus_dbcs_hpca_brst_l2", tint, mms_id, verbose = False) for mms_id in range(1, 5)]
            # ancillary = []
            # for i in range(1, 5):
            #     try:
            #         ancillary.append(mms.load_ancillary(product = "defatt", tint = tint_new, mms_id = i, verbose = False))
            #     except:
            #         print(f"Ancillary data for MMS {i} not available")


        # ancillary = [mms.load_ancillary(product = "defatt", tint = tint, mms_id = mms_id, verbose = False) for mms_id in range(1, 5)]
            # Vi_hpca = [mms.dsl2gse(Vi_hpca[i], ancillary[i], direction = +1) for i in range(len(Vi_hpca))]
            # Vi_hpca = [pyrf.cotrans(Vi_hpca[i], "gse>gsm", ) for i in range(len(Vi_hpca))]
            Vi_avg_hpca = pyrf.avg_4sc(Vi_hpca)
            Vi_avg_hpca = pyrf.time_clip(Vi_avg_hpca, b_avg.time)
            

            # Vi_hpca_fac = [pyrf.convert_fac(Vi_hpca[i], Bgse[i], np.array([1, 0, 0])) for i in range(len(Vi_hpca))]
            # Vi_hpca_fac_avg = pyrf.avg_4sc(Vi_hpca_fac)
            # Vi_hpca_avg_perp_x = Vi_hpca_fac_avg.sel(comp = "x")
            # Vi_hpca_avg_perp_x = pyrf.time_clip(Vi_hpca_avg_perp_x, Bgse[0].time)
        
        except:
            print("HPCA data not available")
            no_hpca = True

        # xs_gsm = np.array([x for x in xs_gsm])
        # zs_gsm = np.array([z for z in zs_gsm])
        # ys_gsm = np.array([y for y in ys_gsm])
        # rs = np.sqrt(xs_gsm**2 + ys_gsm**2 + zs_gsm**2)

# Load FPI perp/par temperature ratio
        t_perp, t_para, t_ratio = temp_ratio(tint_new, s = "e", lowest_energy = 1e2)
        temp = np.sqrt(t_perp**2 + t_para**2)
# Load FPI electron density
        n_e_avg, _ = load_n_t_4sc(tint_new, s = "e", lowest_energy=1e2)
        # n_e_avg = pyrf.ts_convolve(n_e_avg, kernel = np.ones(6)/6)
###### ---------------------------------------------------- ######
        t_series = {}
        t_series = {"b_avg": b_avg, "dbz_dt": dbz_dt,
                    "W_betatron": W_betatron.rolling(time = roll, center=True).median(),
                    "W_fermi": W_fermi.rolling(time = roll, center=True).median(),
                    "w_betatron": w_betatron.rolling(time = roll, center=True).median(),
                    "w_fermi": w_fermi.rolling(time = roll, center=True).median(),
                    "R_g": R_g,
                    "kappa_roll": kappa_roll,
                    "kappa_extra_energy": kappa_extra_energy,
                    "t_ratio": t_ratio.rolling(time = roll_t, center=True).median(),
                    "temp": temp.rolling(time = roll_t, center=True).median(),
                    "b_syn": b_syn, "b_filt": b_filt,
                    "w_betatron_cumsum": w_betatron_cumsum.rolling(time = roll, center=True).median(),
                    "w_fermi_cumsum": w_fermi_cumsum.rolling(time = roll, center=True).median(),
                    "V_i_avg": V_i_avg,
                    "l_i": l_i,
                    "n_i_avg": n_i_avg,
                    "t_i": t_i,
                    "ps_distance": ps_distance,
                    "ps_distance_alt": ps_distance_alt,
                    "n_e_avg": n_e_avg,}
        
        t_series.update({f"f_en_perp_para_{energy}": f_en_perp_para[i] 
                         for i, energy in enumerate(energies_fpi)})
        # t_series.update({f"f_en_omni_{energy}": f_en_omni[i][0] 
                        #  for i, energy in enumerate(energies_fpi)})
        t_series.update({f"f_en_omni_{energy}_norm": f_en_omni[i] 
                         for i, energy in enumerate(energies_fpi)})
        t_series.update({f"f_en_para0_{energy}": f_en_para0[i]
                         for i, energy in enumerate(energies_fpi)})
        t_series.update({f"f_en_para180_{energy}": f_en_para180[i]
                         for i, energy in enumerate(energies_fpi)})
        t_series.update({f"f_en_perp_{energy}": f_en_perp[i]
                         for i, energy in enumerate(energies_fpi)})
        
        if no_hpca == False:
            t_series.update({"V_i_avg_hpca": Vi_avg_hpca,})

        if no_feeps == False:

            t_series.update({f"feeps_en_perp_para_{energy[0]}_{energy[-1]}": feeps_en_perp_para[i] 
                            for i, energy in enumerate(energy_ranges_feeps)})
            # t_series.update({f"feeps_en_omni_{energy[0]}_{energy[-1]}": feeps_en_omni[i][0] 
                            # for i, energy in enumerate(energy_ranges_feeps)})
            t_series.update({f"feeps_en_omni_{energy[0]}_{energy[-1]}_norm": feeps_en_omni[i] 
                            for i, energy in enumerate(energy_ranges_feeps)})
            t_series.update({f"feeps_en_para0_{energy[0]}_{energy[-1]}": feeps_en_para0[i]
                            for i, energy in enumerate(energy_ranges_feeps)})
            t_series.update({f"feeps_en_para180_{energy[0]}_{energy[-1]}": feeps_en_para180[i]
                            for i, energy in enumerate(energy_ranges_feeps)})
            t_series.update({f"feeps_en_perp_{energy[0]}_{energy[-1]}": feeps_en_perp[i]
                            for i, energy in enumerate(energy_ranges_feeps)})
###### ---------------------------------------------------- ######
        if scale_with == "t_i":
            args = {"t_df":t_df[0], "t_i":t_i, "df_tint":df_timescale, 
                    "Vi": V_i_avg, "window_ti": window_ti, "method": "same_t_i", 
                    "unit": "l_i"}
        else:
            args = {}
            args = {"t_df": t_df[0], "b": b, "window_ti": window_ti}

        for key, val in t_series.items():
            
            if key == "b_avg":
                if scale_with == "t_i":
                    val_ti, t_i_before_median, t_i_after_median = time_conv(inp = val, **args)
                else:
                    val_ti= time_conv(inp = val, **args)
            else:
                if scale_with == "t_i":
                    val_ti, _, _ = time_conv(inp = val, **args)
                else:
                    val_ti = time_conv(inp = val, **args)
            dict_ea[it][key] = val
            dict_ea[it][key + f"_{ss}"] = val_ti
###### ---------------------------------------------------- ######
        if scale_with == "b":
            _, t_i_before_median, t_i_after_median = inp_time2ti(inp = b_avg, t_df = t_df[0], t_i = t_i, 
                                                                    df_tint = df_timescale, Vi = V_i_avg, 
                                                                    window_ti =window_ti , method = "same_t_i", 
                                                                    unit = "l_i")
        
        dict_ea[it].update({"tint": tint, "tint_new":tint_new, "t_df_init": t_df_init, "t_df": t_df[0], 
                            "t_i_before_median": t_i_before_median,
                            "t_i_after_median": t_i_after_median, 
                            "df_timescale": df_timescale, "b":b, "a":a, "c":c, 
                            "sigma":sigma, "pcov": pcov, "energies_fpi": energies_fpi, 
                            "energy_ranges_feeps": energy_ranges_feeps, 
                            "feeps_energy_centroids": feeps_energy_centroids,})

    if xs_cond is not None:

        return dict_ea_xs0, dict_ea_xs1
    
    else:

        return dict_ea