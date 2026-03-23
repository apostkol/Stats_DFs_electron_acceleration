# local imports
from mms_codes.t_window import t_window

# 3rd party imports
import xarray as xr
import numpy as np
from pyrfu import pyrf

def inp_time2b(inp, t_df, b, window_ti: float = 5):

    r"""
    Function to create a before and after data array around a specific time, in units of
    an input timescale.

    Parameters:
    -----------
    inp : xarray.DataArray
        Input data.
    t_df : xarray.DataArray
        Time around which to create the dataframe.
    b : float
        Timescale in seconds.
    window_ti : float
        Halfsize of the window in b.

    Returns:
    --------
    out : xarray.DataArray
        DataArray containing the before and after t_df data, with dimensions 
        corresponding to t_i units. 
    """

    window = t_window(t = t_df, window_size = 2 * window_ti * b, method = "center")
    time_sec = (inp.time - t_df) / np.timedelta64(1, 's')
    time2b = time_sec.sel(time=slice(window[0][0], window[1][0])) /b 
    inp_b = inp.sel(time = slice(window[0][0], window[1][0])).assign_coords(time=time2b)

    return inp_b
