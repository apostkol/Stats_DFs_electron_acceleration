# 3rd party imports
from scipy import constants
import xarray as xr
from pyrfu.pyrf import resample
import numpy as np

def calc_fc(b_xyz):
    r"""
    Calculate the ion and electron cyclotron frequencies.

    Parameters:
    -----------
    b_xyz : xarray.DataArray
        Magnetic field.

    Returns:
    --------
    out : xarray.Dataset
        Dataset containing the ion and electron cyclotron frequencies (f_cp, f_ce).
    """

    q_e = constants.elementary_charge
    m_e = constants.electron_mass
    m_p = constants.proton_mass
    mp_me = m_p / m_e

    b_si = 1e-9 * np.linalg.norm(b_xyz, axis=1)

    w_ce = q_e * b_si / m_e 
    f_ce = w_ce / (2 * np.pi)
    f_cp = f_ce / mp_me

    out = xr.Dataset(data_vars = {"time": b_xyz.time, "f_cp":(["time"], f_cp), 
                           "f_ce":(["time"], f_ce)},
                    attrs = dict(description = "Ion and electron cyclotron frequencies",
                                    units = "Hz"))
    return out