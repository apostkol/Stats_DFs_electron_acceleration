import xarray as xr
import numpy as np

def feeps_dpf2psd(inp , units_out: str = "cm"):

    r"""
    Convert feeps data from dpf (cm-2 s-1 sr-1 keV-1) to psd (s+3 cm-6)
    the energy should be in keV
    
    Parameters
    ----------
    inp: xarray.dataarray or xarray.dataset
        dpf data to convert to psd
    Returns
    -------
    out: xarray.dataarray or xarray.dataset
        data in psd
    """
    attrs = inp.attrs.copy()
    energies = inp.energy.data * 1e-3
    data_dpf = inp.data * inp.energy.data

    data_psd = 1.6e-40 * data_dpf / (energies**2 * (energies + 1.02))

    if units_out == "m":
        data_psd *= 1e12
        attrs["UNITS"] = "s^3 m^-6"
    elif units_out == "cm":
        attrs["UNITS"] = "s^3 cm^-6"
    else:
        raise ValueError("units_out should be 'cm' or 'm'")

    attrs["units_name"] = "psd"

    out = xr.DataArray(data = data_psd, dims = inp.dims, coords = inp.coords, attrs = attrs)

    return out