from pyrfu import mms
from pyrfu.pyrf import nanavg_4sc
from pyrfu.mms import feeps_avg_4sc
import numpy as np
from .feeps_codes.feeps_corrections import feeps_corrections
from pyrfu.mms import feeps_remove_sunlit_sectors
from copy import deepcopy
import xarray as xr

def _calc_parapar_parperp(pad):
    coords = [pad.time.data,]
    dims = ["time"]

    if pad.theta.shape[0] %2 == 0:

        idx_90deg = pad.theta.shape[0]/2
        pad_perp = pad.data[:, int(idx_90deg)] + pad.data[:, int(idx_90deg) - 1]
        
    else:

        idx_90deg = int(np.floor(pad.theta.shape[0]/2))
        pad_perp = 2 * pad.data[:, idx_90deg]

    parapar = pad.data[:, 0] / pad.data[:, -1]
    parperp = (pad.data[:, 0] + pad.data[:, -1])/pad_perp

    parapar = xr.DataArray(parapar, coords=coords, dims=dims)
    parperp = xr.DataArray(parperp, coords=coords, dims=dims)

    return parapar, parperp

def feeps_flux_energy(tint, feeps_e_corr_new, s:str = "e", flag:str = "omni", energy_ranges: list = [[32.1, 50], [50, 200]]):

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

    energy_channels_sc = [ [None, 1, 1, 1], 
                       [1, 2, 2, 2], 
                       [2, 3, 3, 3], 
                       [3, 4, 4, 4], 
                       [4, 5, 5, 5], 
                       [5, 6, 6, 6], 
                       [6, 7, 7, 7], 
                       [7, 8, 8, 8], 
                       [None, 9, 9, 9], 
                       [10, 10, 10, 10], 
                       [11, 11, 11, 11], 
                       [12, 12, 12, 12], 
                       [13, 13, 13, 13], 
                       [14, 14, 14, 14], 
                       [15, 15, 15, 15]]

    combined_energies_dict = {i: {j: energy_channels_sc[i][j] for j in range(4)} for i in range(len(energy_channels_sc))}
    assert s in ["e", "i"], "Species must be 'e' or 'i'"
    assert flag in ["omni", "ratio"], "Flag must be 'omni' or 'ratio'"

    if s == "e":
        data_type = "electron"
    else:
        data_type = "ion"

    n_pabins = 9
    bin_size = 180.0 / n_pabins

    # feeps_alle_e = [mms.get_feeps_alleyes(f"flux{s}_brst_l2", tint, mms_id, verbose = False) for mms_id in range(1, 5)]
    # feeps_e_corr_new = feeps_corrections(deepcopy(feeps_alle_e))
    # feeps_e_corr_new = [feeps_remove_sunlit_sectors(feeps_e_corr_new[i],) for i in range(4)]
    energies_centr = [(energy_range[1] - energy_range[0])/(np.log(energy_range[1]) - np.log(energy_range[0])) for energy_range in energy_ranges]

    if flag == "ratio":
        
        b_bcs = [mms.get_data("b_bcs_fgm_brst_l2", tint, mms_id, verbose = False) for mms_id in range(1, 5)]
        

        feeps_pad_e_energy_ranges = []

        for i in range(4):

            feeps_pad_e_energy_ranges.append([mms.feeps_pad(feeps_e_corr_new[i], b_bcs[i], 
                                                                energy = energy_range, 
                                                                bin_size=bin_size) 
                                                                for energy_range in energy_ranges])

        feeps_pad_4sc_enranges = [feeps_avg_4sc([feeps_pad_e_energy_ranges[i_sc][i_en] for i_sc in range(4)], flag="pad") 
                                    for i_en in range(len(energy_ranges))]

        out = []
        # return feeps_pad_4sc_enranges
        for i in range(len(energy_ranges)):

            _, par_perp = _calc_parapar_parperp(feeps_pad_4sc_enranges[i])
            out.append(1/par_perp)
    
    else:
        
        feeps_omni = [mms.feeps_omni(feeps_e_corr_new[i],) for i in range(4)]
        feeps_omni_avg = feeps_avg_4sc(feeps_omni, flag="omni", combined_energies=combined_energies_dict)

        out = []

        for i, energy_range in enumerate(energy_ranges):

            out_1 = feeps_omni_avg.sel(energy = energies_centr[i], method = "nearest")
            # out_2 = out_1 / out_1.mean()

            out.append(out_1)
    
    return out


