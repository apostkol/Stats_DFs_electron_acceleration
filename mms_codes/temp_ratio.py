from pyrfu.pyrf import avg_4sc
from pyrfu.mms import get_data
import numpy as np

def temp_ratio(tint, s:str = "e", lowest_energy:float = 1e2):
    r"""Load burst partial temperature data, average over the four MMS spacecraft and calcuate
    the perpendicular/parallel temperature ratio.
    
    Parameters
    ----------
    tint : list
        Time interval for the data.
    s : str
        Species, "e" for electrons and "i" for ions.
    lowest_energy : float
        Lowest included energy for partial particle moments.
        
    Returns
    -------
    t_perp_4sc : xarray.DataArray
        Averaged perpendicular temperature data.
    t_para_4sc : xarray.DataArray
        Averaged parallel temperature data.
    t_ratio_4sc : xarray.DataArray
        Temperature ratio.
    """

    assert s in ["e", "i"], "Species must be 'e' or 'i'"

    partt_perp = []; partt_para = []
    for i in range(1, 5):
        try:
            partt_perp.append(get_data(f"parttperp{s}_fpi_brst_l2", tint, i, verbose=False))
            partt_para.append(get_data(f"parttpara{s}_fpi_brst_l2", tint, i, verbose=False))
        except:
            return f"Data for MMS{i} not available"
        
    idx_energy_cut = [np.where(partt_perp[i].lowest_included_energy > lowest_energy)[0][0] for i in range(4)]
    t_perp = [partt_perp[i][:, idx_energy_cut[i]] for i in range(4)]

    idx_energy_cut = [np.where(partt_para[i].lowest_included_energy > lowest_energy)[0][0] for i in range(4)]
    t_para = [partt_para[i][:, idx_energy_cut[i]] for i in range(4)]

    t_perp_4sc = avg_4sc(t_perp)
    t_para_4sc = avg_4sc(t_para)
    t_ratio_4sc = t_perp_4sc / t_para_4sc

    return t_perp_4sc, t_para_4sc, t_ratio_4sc