# 3rd party imports
from scipy import constants
import xarray as xr
from pyrfu.pyrf import resample
from pyrfu.pyrf import norm
import numpy as np

def calc_rg(b, t, s = "i"):
    r"""
    Calculate the ion or electron gyroradius.

    Parameters:
    -----------
    b : xarray.DataArray
        Magnetic field in nT.
    t : xarray.DataArray
        Temperature in eV.
    s : str
        Species, "i" or "e" for ions or electrons.

    Returns:
    --------
    out : xarray.DataArray
        DataArray containing the ion or electron gyroradius.
    """

    assert s in ["i", "e"], "s must be either 'i' or 'e'."

    q_e = constants.elementary_charge
    eV_to_J = constants.physical_constants["electron volt-joule relationship"][0]

    t = resample(t, b)
    b_si = norm(b) * 1e-9  # Convert nT to T

    if s == "i" :
        m = constants.proton_mass
    else:
        m = constants.electron_mass
    # Resample all variables with respect to the magnetic field
    v_th = np.sqrt(2 * t * eV_to_J / m)  # Thermal speed in m/s

    r_g = m* v_th / (q_e * b_si)  # Gyroradius in meters

    # l = cel / w_p

    # out = xr.DataArray(data=l, coords={"time": b_xyz.time}, dims="time",
    #                    attrs = dict(description = "Inertial length",
    #                                 units = "m"))
    return r_g