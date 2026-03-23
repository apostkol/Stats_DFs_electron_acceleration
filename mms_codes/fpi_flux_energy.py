from pyrfu import mms
from pyrfu.pyrf import avg_4sc
import numpy as np
from .calc_parapar_parperp import calc_parapar_parperp

def fpi_flux_energy(tint, s:str = "e", flag:str = "omni", energies: list = [4, 25]):

    r"""Load burst vdf data, average over the four MMS spacecraft and return the 
    omni-directional flux or the f_perp/f_parallel for chosen energies.
    
    Parameters
    ----------
    tint : list
        Time interval for the data.
    s : str
        Species, "e" for electrons and "i" for ions.
    flag : str
        "omni" for omni-directional flux and "ratio" for f_perp/f_parallel.
    energies : list
        List of energies in keV for which to calculate the ratio. The routine will find 
        the closest available energy channel.
        
    Returns
    -------
    out : xarray.DataArray
        Omni-directional flux or f_perp/f_parallel for chosen energies.
    """

    assert s in ["e", "i"], "Species must be 'e' or 'i'"
    assert flag in ["omni", "ratio"], "Flag must be 'omni' or 'ratio'"

    b_dmpa = [mms.get_data("b_dmpa_fgm_brst_l2", tint = tint, mms_id = id, verbose=False) for id in range(1, 5)]

    vdf_e = []
    for id in range(4):
        vdf = mms.get_data(f"pd{s}_fpi_brst_l2", tint, id+1, verbose=False)
        vdf_e.append(vdf)

    vdf_e = [mms.psd2def(vdf) for vdf in vdf_e]

    if flag == "ratio":
        
        pad_def_e = [mms.get_pitch_angle_dist(vdf_e[i], b_dmpa[i], tint=tint, angles = 9, verbose=False) for i in range(4)]
        pad_def_e = avg_4sc(pad_def_e)

        _, par_perp = calc_parapar_parperp(pad_def_e)
        perp_par = 1/par_perp

        out = [perp_par.sel(energy = energy * 1e3, method = "nearest") for energy in energies]
    
    else:

        vdf_e_omni = avg_4sc([mms.vdf_omni(vdf_e[i]) for i in range(4)])
        out = []
        for energy in energies:
            out_1 = vdf_e_omni.sel(energy = energy * 1e3, method = "nearest")
            # out_2 = out_1 / out_1.mean()

            out.append(out_1)
    
    return out

