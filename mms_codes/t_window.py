import numpy as np
from pyrfu import pyrf

def t_window(t, window_size, method: str = "center"):
    
    """
    Input:
    t: timestamp of window center: datetime64
    window_size: size of time window (width) in s
    Output:
    window: list of dattime64 corresponding to start, end of window
    """
    if not isinstance(t, np.datetime64):

        t = np.datetime64(t.data)

    
    t_ns = pyrf.datetime642ttns(t)
    window_size_ns = window_size * 1e9

    if method == "center":
        window_ns = [t_ns - window_size_ns/2, t_ns + window_size_ns/2]
    elif method == "right":
        window_ns = [t_ns - window_size_ns, t_ns]
    elif method == "left":
        window_ns = [t_ns, t_ns + window_size_ns]

    window = [pyrf.ttns2datetime64(window_ns[i]) for i in range(2)]
    
    return window