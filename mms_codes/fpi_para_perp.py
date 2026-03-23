import xarray as xr
import numpy as np
from pyrfu import mms
from pyrfu.pyrf import avg_4sc

def calc_para_perp(pad):
    coords = [pad.time.data, pad.energy.data[0, :]]
    dims = ["time", "energy"]

    if pad.theta.shape[1] %2 == 0:

        idx_90deg = pad.theta.shape[1]/2
        pad_perp = pad.data.data[:, :, int(idx_90deg)] + pad.data.data[:, :, int(idx_90deg) - 1]
        
    else:

        idx_90deg = int(np.floor(pad.theta.shape[1]/2))
        pad_perp = 2 * pad.data.data[:, :, idx_90deg]

    para0 = xr.DataArray(pad.data.data[:, :, 0], coords=coords, dims=dims)
    para180 = xr.DataArray(pad.data.data[:, :, -1], coords=coords, dims=dims)
    perp = xr.DataArray(pad_perp, coords=coords, dims=dims)

    return para0, para180, perp


def fpi_para_perp(tint, s:str = "e", energies: list = [4, 25]):
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
    # assert flag in ["omni", "ratio"], "Flag must be 'omni' or 'ratio'"

    b_dmpa = [mms.get_data("b_dmpa_fgm_brst_l2", tint = tint, mms_id = id, verbose=False) for id in range(1, 5)]

    vdf_e = []
    # vdf_e_err = []

    for id in range(4):
        vdf = mms.get_data(f"pd{s}_fpi_brst_l2", tint, id+1, verbose=False)
        # vdf_err = mms.get_data(f"pderr{s}_fpi_brst_l2", tint, id+1, verbose=False)
        vdf_e.append(vdf)
        # vdf_e_err.append(vdf_err)

    # vdf_e_counts = [(vdf_e[i]/vdf_e_err[i])**2 for i in range(4)]

    # for i in range(4):
    #     vdf_e_counts[i].data.attrs["UNITS"] = "counts"
    #     vdf_e_counts[i]["phi"] = vdf_e[i].phi
    #     vdf_e_counts[i]["theta"] = vdf_e[i].theta
    #     vdf_e_counts[i]["energy"] = vdf_e[i].energy

        
    pad_def_e = [mms.get_pitch_angle_dist(vdf_e[i], b_dmpa[i], angles = 9, verbose=False) for i in range(4)]
    pad_def_e = avg_4sc(pad_def_e)

    para0, para180, perp = calc_para_perp(pad_def_e)

    para0_en = [para0.sel(energy = energies[i] * 1e3, method = "nearest") for i in range(len(energies))]
    para180_en = [para180.sel(energy = energies[i] * 1e3, method = "nearest") for i in range(len(energies))]
    perp_en = [perp.sel(energy = energies[i] * 1e3, method = "nearest") for i in range(len(energies))]

    return para0_en, para180_en, perp_en