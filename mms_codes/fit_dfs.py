import pickle
from .df_tint import find_df_timescale

def fit_dfs(tints, fit_manually,fit_path = None,):
    r"""
    Function to fit the DFs manually.

    Parameters:
    -----------
    tints : list of str
        Time intervals.
    fit_manually : bool
        If True, the DFs are fitted "manually".
    fit_path : str
        Path to save the fit dictionary.
    
    Returns:
    --------
    dict_fit : dict
        Dictionary with the fit parameters.

    """

    dict_fit = {it: {} for it in range(len(tints))}

    for it, tint in enumerate(tints):
        
        print(f"{it+1}/{len(tints)}")
        print(f"Doing the fit for tint: {tint}")
        t_df_init, _, b_syn, a, b, c, sigma, t_df, b_filt, pcov = find_df_timescale(tint, fit_manually = fit_manually)
        dict_fit[it] = {"t_df_init": t_df_init, "b_syn": b_syn, "a": a, "b": b, "c": c, 
                        "sigma": sigma, "t_df": t_df, "b_filt": b_filt, "pcov": pcov, "tint": tint}
    
    if fit_path is not None:

        with open(fit_path, "wb") as f:
            pickle.dump(dict_fit, f, protocol=pickle.HIGHEST_PROTOCOL)

        return dict_fit
    else:
        return dict_fit

