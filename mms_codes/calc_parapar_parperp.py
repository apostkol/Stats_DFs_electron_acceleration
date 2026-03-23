import xarray as xr
import numpy as np

def calc_parapar_parperp(pad):
    coords = [pad.time.data, pad.energy.data[0, :]]
    dims = ["time", "energy"]

    if pad.theta.shape[1] %2 == 0:

        idx_90deg = pad.theta.shape[1]/2
        pad_perp = pad.data.data[:, :, int(idx_90deg)] + pad.data.data[:, :, int(idx_90deg) - 1]
        
    else:

        idx_90deg = int(np.floor(pad.theta.shape[1]/2))
        pad_perp = 2 * pad.data.data[:, :, idx_90deg]

    parapar = pad.data.data[:, :, 0] / pad.data.data[:, :, -1]
    parperp = (pad.data.data[:, :, 0] + pad.data.data[:, :, -1])/pad_perp

    parapar = xr.DataArray(parapar, coords=coords, dims=dims)
    parperp = xr.DataArray(parperp, coords=coords, dims=dims)

    return parapar, parperp