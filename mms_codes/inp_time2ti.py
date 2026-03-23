# local imports
from mms_codes.t_window import t_window

# 3rd party imports
import xarray as xr
import numpy as np
from pyrfu import pyrf

def inp_time2ti(inp, t_df, t_i, df_tint, Vi = None, window_ti: float = 5, 
                method: str = "same_t_i", unit: str = "t_i"):

    r"""
    Function to create a before and after data array around a specific time, in units of
    an input timescale.

    Parameters:
    -----------
    inp : xarray.DataArray
        Input data.
    t_df : xarray.DataArray
        Time around which to create the before and after dataframe.
    t_i : str
        Time corresponding to inertial length in seconds.
    df_tint : list of str of interval
        Time interval of structure.
    window_size : float
        Halfsize of the window in t_i.

    Returns:
    --------
    out : xarray.DataArray
        DataArray containing the before and after t_df data, with dimensions 
        corresponding to t_i units. 
    """

    assert unit in ["t_i", "l_i"], "unit must be either 't_i' or 't_e'."

    message = 'method must be either "same_t_i" or "different_t_i".'
    assert method in ["same_t_i", "different_t_i"], message

    if method == "same_t_i":
        
        t_i_before = t_i.sel(time = slice(df_tint[0], t_df))

        t_i_before_median = t_i_before.median().data
        t_i_after_median = t_i_before.median().data 

        if unit == "l_i":

            Vi_mag = pyrf.norm(Vi)

            V_i_before = Vi_mag.sel(time = slice(df_tint[0], t_df)).median().data
            V_i_after = V_i_before
        
    else:

        t_i_before = t_i.sel(time = slice(df_tint[0], t_df))
        t_i_after = t_i.sel(time = slice(t_df, df_tint[1]))

        t_i_before_median = t_i_before.median().data
        t_i_after_median = t_i_after.median().data

        if unit == "l_i":
            
            Vi_mag = pyrf.norm(Vi)

            V_i_before = Vi_mag.sel(time = slice(df_tint[0], t_df)).median().data
            V_i_after = Vi_mag.sel(time = slice(t_df, df_tint[1])).median().data

    window_before = t_window(t = t_df, window_size = window_ti * t_i_before_median, method = "right")
    window_after = t_window(t = t_df, window_size = window_ti * t_i_after_median, method = "left")

    time_sec = (inp.time - t_df) / np.timedelta64(1, 's')

    t_before = slice(window_before[0][0], t_df - np.timedelta64(1, 'ns'))
    t_after = slice(t_df, window_after[1][0])
    
    if unit == "t_i":
        
        time_ti = xr.concat([time_sec.sel(time=t_before) / t_i_before_median.data, 
                            time_sec.sel(time=t_after)/ t_i_after_median.data], 
                            dim="time")
    
    else:

        l_i_before = t_i_before_median * V_i_before
        l_i_after = t_i_after_median * V_i_after

        time_ti = xr.concat([time_sec.sel(time=t_before) * V_i_before / l_i_before, 
                            time_sec.sel(time=t_after) * V_i_after / l_i_after], 
                            dim="time")

    inp_ti = inp.sel(time = slice(window_before[0][0], window_after[1][0])).assign_coords(time=time_ti)

    return inp_ti, t_i_before_median, t_i_after_median

# def inp_time2li(inp, t_df, t_i, df_tint, Vi, window_ti: float = 5, method: str = "same_t_i", ):

#     r"""
#     Function to create a before and after data array around a specific time, in units of
#     an input timescale.

#     Parameters:
#     -----------
#     inp : xarray.DataArray
#         Input data.
#     t_df : xarray.DataArray
#         Time around which to create the before and after dataframe.
#     t_i : str
#         Time corresponding to inertial length in seconds.
#     df_tint : list of str of interval
#         Time interval of structure.
#     window_size : float
#         Halfsize of the window in t_i.

#     Returns:
#     --------
#     out : xarray.DataArray
#         DataArray containing the before and after t_df data, with dimensions 
#         corresponding to t_i units. 
#     """
#     if method == "same_t_i":
        
#         t_i_before = t_i.sel(time = slice(df_tint[0], t_df))

#         t_i_before_median = t_i_before.median().data
#         t_i_after_median = t_i_before.median().data 
        
#     else:

#         t_i_before = t_i.sel(time = slice(df_tint[0], t_df))
#         t_i_after = t_i.sel(time = slice(t_df, df_tint[1]))

#         t_i_before_median = t_i_before.median().data
#         t_i_after_median = t_i_after.median().data

#     window_before = t_window(t = t_df, window_size = window_ti * t_i_before_median, method = "right")
#     window_after = t_window(t = t_df, window_size = window_ti * t_i_after_median, method = "left")

#     time_sec = (inp.time - t_df) / np.timedelta64(1, 's')
    # time_ti = xr.concat([time_sec.sel(time=slice(window_before[0][0], t_df - np.timedelta64(1, 'ns'))) / t_i_before_median.data, 
#                         time_sec.sel(time=slice(t_df, window_after[1][0])) / t_i_after_median.data], dim="time")
    
#     inp_ti = inp.sel(time = slice(window_before[0][0], window_after[1][0])).assign_coords(time=time_ti)

#     return inp_ti, t_i_before_median, t_i_after_median