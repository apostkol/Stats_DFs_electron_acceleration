# 3rd party imports
from scipy import constants
import xarray as xr
from pyrfu.pyrf import resample
import numpy as np

def calc_l(n, b_xyz, s = "i"):
    r"""
    Calculate the ion inertial length.

    Parameters:
    -----------
    n_i : xarray.DataArray
        Number density of the ions.
    b_xyz : xarray.DataArray
        Magnetic field.
    s : str
        Species, "i" or "e".

    Returns:
    --------
    out : xarray.DataArray
        DataArray containing the ion inertial length.
    """

    assert s in ["i", "e"], "s must be either 'i' or 'e'."
    q_e = constants.elementary_charge
    cel = constants.speed_of_light
    ep0 = constants.epsilon_0

    m = {"i": constants.proton_mass, "e": constants.electron_mass}
    # Resample all variables with respect to the magnetic field
    n = resample(n, b_xyz).data * 1e6  # cm^-3 -> m^-3

    w_p = np.sqrt(n * q_e**2 / (m[s] * ep0))

    l = cel / w_p

    out = xr.DataArray(data=l, coords={"time": b_xyz.time}, dims="time",
                       attrs = dict(description = "Inertial length",
                                    units = "m"))
    return out