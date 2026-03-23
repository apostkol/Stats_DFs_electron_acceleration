# 3rd party imports
import numpy as np
from pyrfu.pyrf import (avg_4sc, norm, c_4_grad, dot, e_vxb,
                        dec_par_perp, sliding_derivative, resample, 
                        ts_convolve)
from pyrfu.mms import (get_data, rotate_tensor)
from scipy import constants

def acceleration_terms(tint, verbose = False, part_moments_en = 1e2, kappa_limit = 3, B_smooth_window = 5):
    r"""
    Function to calculate the betatron and Fermi acceleration terms, as well as the kappa parameter.
    
    Parameters:
    -----------
    tint : list of str
        Time interval.
    verbose : bool
        Verbose mode.
    part_moments_en : float
        Lowest included energy for particle moments.
    kappa_limit : float
        Limit for kappa parameter.
    B_smooth_window : float
        Window size for smoothing the magnetic field.
    
    Returns:
    --------
    W_betatron : xarray.DataArray
        Betatron acceleration term.
    W_fermi : xarray.DataArray
        Fermi acceleration term.
    w_betatron : xarray.DataArray
        Betatron acceleration term normalized by the electron density.
    w_fermi : xarray.DataArray
        Fermi acceleration term normalized by the electron density.
    kappa_roll : xarray.DataArray
        Kappa parameter.
    """

    ### Load data
    Egse_fast = [get_data(f'e_gse_edp_fast_l2', tint=tint, mms_id=id, verbose = verbose) for id in range(1, 5)]
    Bgse = [get_data(f'b_gse_fgm_brst_l2', tint=tint, mms_id=id, verbose = verbose) for id in range(1, 5)]
    
    Ve_gse_part = [get_data(f'partve_gse_fpi_brst_l2', tint=tint, mms_id=id, verbose = verbose) for id in range(1, 5)]
    
    idx_energy_cut = [np.where(Ve_gse_part[i].lowest_included_energy > part_moments_en)[0][0] for i in range(4)]
    Ve_gse = [Ve_gse_part[i][:, idx_energy_cut[i], :] for i in range(4)]

    n_e_part = [get_data(f"partne_fpi_brst_l2", tint, id, verbose = verbose) for id in range(1, 5)]
    
    idx_energy_cut = [np.where(n_e_part[i].lowest_included_energy > part_moments_en)[0][0] for i in range(4)]
    n_e = [n_e_part[i][:, idx_energy_cut[i]] for i in range(4)]

    partt_perp_e = [get_data(f"parttperpe_fpi_brst_l2", tint, id, verbose = verbose) for id in range(1, 5)]
    idx_energy_cut = [np.where(partt_perp_e[i].lowest_included_energy > part_moments_en)[0][0] for i in range(4)]
    t_perp_e = [partt_perp_e[i][:, idx_energy_cut[i]] for i in range(4)]
    
    partt_para_e = [get_data(f"parttparae_fpi_brst_l2", tint, id, verbose = verbose) for id in range(1, 5)]
    idx_energy_cut = [np.where(partt_para_e[i].lowest_included_energy > part_moments_en)[0][0] for i in range(4)]

    r_gse = [get_data(f'r_gse_mec_srvy_l2', tint, mms_id, verbose = verbose) for mms_id in range(1,5)]
    for i in range(len(r_gse)):
        r_gse[i].coords["comp"] = np.array(["x", "y", "z"])
        
    pe_part = [get_data(f'partpe_gse_fpi_brst_l2', tint=tint, mms_id=id, verbose = verbose) for id in range(1, 5)]
    
    idx_energy_cut = [np.where(pe_part[i].lowest_included_energy > part_moments_en)[0][0] for i in range(4)]
    pe_cut = [pe_part[i][:, idx_energy_cut[i], :] for i in range(4)]
    
    pe_cut_fac = [rotate_tensor(pe_cut[i], "fac", Bgse[i], "pp") for i in range(4)]
    
    pe_cut_perp = [(pe_cut_fac[i][:,1,1]) for i in range(4)]

    ### Calculate acceleration terms
    Bgse_smoothed = [resample(Bgse[i], Egse_fast[i]) for i in range(4)]
    Bgse_smoothed = [ts_convolve(Bgse_smoothed[i], np.ones(shape = B_smooth_window)/B_smooth_window) for i in range(4)]
    
        ### Calculate betatron acceleration
    B_mag = [norm(Bgse_smoothed[i]) for i in range(4)]
    r_gse = [resample(r_gse[i], B_mag[i]) for i in range(4)]
    gradB = c_4_grad(r_gse, B_mag)
    
    t_perp_e_smoothed = [resample(t_perp_e[i], Egse_fast[i]) for i in range(4)]

    u_E = [e_vxb(Egse_fast[i], Bgse_smoothed[i], flag = "exb") for i in range(4)]
    
    db_dt_5ps = avg_4sc([sliding_derivative(B_mag[i],  t_units= "ns",method = "5ps") for i in range(4)])
    
    pe_perp_avg = avg_4sc(pe_cut_perp)
    
    u_E_avg = avg_4sc(u_E)
    B_mag_avg = avg_4sc(B_mag)
    gradB = c_4_grad(r_gse, B_mag, method = "grad")
    uE_dot_gradB_avg = dot(u_E_avg, gradB)
    
    W_betatron = resample(pe_perp_avg, u_E_avg) * (resample(db_dt_5ps, u_E_avg) + uE_dot_gradB_avg)/resample(B_mag_avg, u_E_avg) *6.24151

    w_betatron = W_betatron / resample(avg_4sc(n_e), u_E_avg)
    # W_bet_wei_units = W_bet_E_fast*1e3 

        ### Calculate Fermi acceleration
    ev2j = constants.physical_constants["electron volt-joule relationship"][0]
    
    Ve_para = [dec_par_perp(Ve_gse[i], Bgse[i])[0] for i in range(4)]
    pe_cut_para = [(pe_cut_fac[i][:,0,0]) for i in range(4)]
    curv = c_4_grad(r_gse, Bgse_smoothed,  method = "curv")
    
    pe_para_avg = resample(avg_4sc(pe_cut_para), u_E_avg)
    n_e_avg = resample(avg_4sc(n_e), u_E_avg)
    Ve_para_avg = resample(avg_4sc(Ve_para), u_E_avg)
    
    W_fermi = 1e-3 * ev2j**(-1) * (1e-15*pe_para_avg + 1e6 * constants.m_e * n_e_avg * (Ve_para_avg)**2) \
                * dot(u_E_avg, curv )
    
    w_fermi = W_fermi / resample(avg_4sc(n_e), u_E_avg)
    # W_fermi_wei_units = W_fermi*1e3 

    ### Calculate kappa 
    Rc = resample(norm(c_4_grad(r_gse, Bgse_smoothed,  method = "curv"))**(-1), avg_4sc(t_perp_e_smoothed))
    Rc_roll = Rc.rolling(time = 15, center=True).median()

    Rg = [1e-3*np.sqrt(2*t_perp_e_smoothed[i] * ev2j/constants.electron_mass) * 
          constants.electron_mass/
          (constants.e * resample(B_mag[i], t_perp_e_smoothed[i])*1e-9) 
          for i in range(4)]
    
    Rg_avg = avg_4sc(Rg)
    
    kappa_roll = np.sqrt(Rc_roll/Rg_avg)
    ### Calculate kappa for extra energy
    extra_energy = 10e4 #eV

    Rg_extra_energy = [1e-3 * constants.electron_mass/(constants.e * resample(B_mag[i], t_perp_e_smoothed[i]) * 1e-9) 
                   * np.sqrt(2 * extra_energy * ev2j / constants.electron_mass)
                    for i in range(4)]
    Rg_extra_energy_avg = avg_4sc(Rg_extra_energy)
    
    kappa_extra_energy = np.sqrt(Rc_roll / Rg_extra_energy_avg)

    ### limit acceleration terms to regions where kappa > kappa_limit
    w_fermi = w_fermi.where(kappa_roll > kappa_limit)
    w_betatron = w_betatron.where(kappa_roll > kappa_limit)
    W_fermi = W_fermi.where(kappa_roll > kappa_limit)
    W_betatron = W_betatron.where(kappa_roll > kappa_limit)
    
    w_betatron_cumsum = w_betatron.cumsum()
    w_fermi_cumsum = w_fermi.cumsum()

    return W_betatron, W_fermi, w_betatron, w_fermi, w_betatron_cumsum, w_fermi_cumsum, kappa_roll, Rg_avg, kappa_extra_energy
        