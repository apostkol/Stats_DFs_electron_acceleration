# 3rd party imports
import numpy as np
from scipy import optimize, signal
from pyrfu import pyrf
from pyrfu.pyrf import (ts_scalar, avg_4sc,
                        ts_convolve, sliding_derivative)
from pyrfu.plot import (plot_line, make_labels)
from matplotlib import pyplot as plt
from IPython import display
from .calc_fc import calc_fc
from pyrfu.mms import get_data

def fit_df(b_gsm, t_df_init, t_span: list = [-20, 10]):
    r"""Fit North-South magnetic field using the method described in [1]_
    in order to identify dipolarization fronts in the time series. 
    Credits: Louis Richard.

    Parameters
    ----------
    b_gsm : xarray.DataArray
        Time series of the magnetic field in GSM coordinates.

    Returns
    -------
    t_df : np.datetime64
        Time of the candidate DF.
    b_syn : xarray.DataArray
        Time series of the fit of the North-South magnetic field in GSM
        coordinates.
    a : float
        Amplitude of the hyperbolic tangent [nT].
    b : float
        Time scale of DF [s].
    sigma : float
        Root-mean square of the fit residual [nT].

    References
    ----------
    .. [1]  Fu, H. S., Y. V. Khotyaintsev, A. Vaivads, M. André, and S. Y.
            Huang (2012), Occurrence rate of earthward-propagating
            dipolarization fronts, Geophys. Res. Lett., 39, L10101,
            doi:10.1029/2012GL051784.

    """

    df_start, df_end = t_span

    tint_df = [(t_df_init - np.timedelta64(abs(df_start), "s")).astype(str),
            (t_df_init + np.timedelta64(abs(df_end), "s")).astype(str)]
    
    # n_i, _ = load_n_t_4sc(tint_df, s = "i", lowest_energy = 1e2)

    b_df = pyrf.time_clip(b_gsm, tint_df)
    
    # b_filt = pyrf.ts_scalar(b_df.time.data,
    #                 signal.savgol_filter(b_df.data[:, 2], int(2*pyrf.calc_fs(n_i)), 5))
    # b_df = b_df.rolling(time = 20, center = True).mean()
    
    b_z = b_df.data[:, 2]
    p_, pcov = optimize.curve_fit(bz_model,
                                np.linspace(df_start, df_end, len(b_z)),
                                b_z, bounds = ([4, 0, -20, -20], [np.inf, 8, 20, 20]), 
                                nan_policy="omit")
    a, b, c, t_df = p_
    # Estimate the RMS residual
    b_syn = bz_model(np.linspace(df_start, df_end, len(b_z)), a, b, c, t_df)

    t_df = ((pyrf.datetime642ttns(np.datetime64(t_df_init))*1e-9 + t_df) * 1e9).astype(str)
    t_df = pyrf.ttns2datetime64(t_df.astype(float))

    sigma = np.sqrt(np.nanmean((b_z - b_syn) ** 2))

    b_syn = ts_scalar(b_df.time.data, b_syn)
    
    return b_syn, a, b, c, sigma, t_df, ts_scalar(b_df.time.data, b_z), pcov

def bz_model(t, a, b, c, t_df):
    r"""Model of the dipolarization front North-South magnetic field as an
    hyperbolic tangent. Credits: Louis Richard.

    Parameters
    ----------
    delta_t : np.ndarray
        Time line to fit the magnetic field.
    a : float
        Amplitude of the dipolarization front.
    b : float
        Time scale of the dipolarization front.
    c : float
        Offset of the dipolarization front.

    Returns
    -------
    b_mod : np.ndarray
        Synthetic dipolarization front magnetic field.

    """

    b_mod = (0.5 * a) * np.tanh((t - t_df) / (0.5 * b)) + (c + a / 2)

    return b_mod

def load_b_field(tint):
    r"""Load magnetic field data, resample to E_fast resolution and average 
    over the MMS spacecraft
    
    Parameters
    ----------
    tint : list
        Time interval for the data.
        
    Returns
    -------
    b_gsm : list
        List of magnetic field data in GSM coordinates for each MMS spacecraft.
    b_avg : xarray.DataArray
        Averaged magnetic field data in GSM coordinates.
    
    Raises
    ------
    ValueError
        If the data is not available.
    
    """

    b_gsm = []
    e_gse = []
    for i in range(1, 5):
        try:
            b_gsm.append(get_data("b_gsm_fgm_brst_l2", tint, i, verbose=False))
            e_gse.append(get_data("e_gse_edp_fast_l2", tint, i, verbose=False))
        except:
            return f"Data for MMS{i} not available"
    
    # e_gsm = [cotrans(e, "gse", "gsm") for e in e_gse]
    # b_gsm = [resample(b, e) for b, e in zip(b_gsm, e_gsm)]
    # b_gsm = [ts_convolve(b, np.ones(shape=5)/5) for b in b_gsm]
    b_avg = avg_4sc(b_gsm)

    return b_gsm, b_avg

def load_n_t_4sc(tint, s = "e", lowest_energy = 1e2):
    r"""Load particle data, density and temperature, and average over the four MMS spacecraft.
    
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
    n_4sc : xarray.DataArray
        Averaged density data.
    t_4sc : xarray.DataArray
        Averaged temperature data.

    Raises
    ------
    ValueError
        If the data is not available.
    """

    assert s in ["e", "i"], "Species must be 'e' or 'i'"

    partt = []; partn = []
    
    for i in range(1, 5):
        try:
            partt.append(get_data(f"partt{s}_gse_fpi_brst_l2", tint, i, verbose=False))
            partn.append(get_data(f"partn{s}_fpi_brst_l2", tint, i, verbose=False))
        except:
            return f"Data for MMS{i} not available"
        
    idx_energy_cut = [np.where(partt[i].lowest_included_energy > lowest_energy)[0][0] for i in range(4)]
    t = [partt[i][:, idx_energy_cut[i]] for i in range(4)]
    t_4sc = pyrf.avg_4sc(t)

    idx_energy_cut = [np.where(partn[i].lowest_included_energy > lowest_energy)[0][0] for i in range(4)]
    n = [partn[i][:, idx_energy_cut[i]] for i in range(4)]
    n_4sc = pyrf.avg_4sc(n)

    return n_4sc, t_4sc

def find_df_timescale(tint, fit_manually: bool = False):
    r"""Find the time scale of dipolarization fronts.
    
    Parameters
    ----------
    tint : list
        Time interval for the data.
    fit_manually : bool
        If True, the DFs are fitted "manually".
        
    Returns
    -------
    t_df : np.datetime64
        Time of the candidate DF.
    b_syn : xarray.DataArray
        Time series of the fit of the North-South magnetic field in GSM
        coordinates.
    a : float
        Amplitude of the hyperbolic tangent [nT].
    b : float
        Time scale of DF [s].
    sigma : float
        Root-mean square of the fit residual [nT].
    
    """

    _, b_avg = load_b_field(tint)

    # f_ce = calc_fc(b_avg).f_ce
    # n_pts = int(0.5 * f_ce.median())
    n_pts = 5
 #    f_ce = calc_fc(b_avg).f_cp
    # n_pts = int(1/(2 * f_ce.median()))
    b_avg_filt = ts_convolve(b_avg, kernel = np.ones(n_pts)/n_pts)

    dbz_dt = sliding_derivative(b_avg_filt.sel(represent_vec_tot = "z"),  t_units= "ns",method = "5ps")
    t_max_bz = b_avg_filt.sel(represent_vec_tot = "z").idxmax()
    t_df_init = dbz_dt.where((dbz_dt.time < t_max_bz) & (dbz_dt.time > t_max_bz - np.timedelta64(30 , "s"))).idxmax()

    if fit_manually:

        b_syn, a, b, c, sigma, t_df, b_filt, pcov = fit_df(b_avg_filt, t_df_init.data)
        pcov_std = np.sqrt(np.diag(pcov))
        plt.close("all")
        fig, axs = plt.subplots(2, sharex= "all", figsize = (9, 7))
        plot_line(axs[0], b_avg_filt.sel(represent_vec_tot = "z"), label = "Bz")
        plot_line(axs[0], b_syn, label = "Fit")
        axs[0].set_ylabel("B [nT]")
        
        plot_line(axs[1], dbz_dt)
        axs[1].set_ylabel("dBz/dt [nT/s]")

        axs[0].text(s = f"b = ({np.around(b, 2)} $\pm$ {np.around(pcov_std[1], 2)}) $s$", x = 0.1, y = 0.7, transform = axs[0].transAxes)
        axs[0].text(s = f"a = ({np.around(a, 2)} $\pm$ {np.around(pcov_std[0], 2)}) $nT$", x = 0.1, y = 0.8, transform = axs[0].transAxes)
        axs[0].text(s = f"c = ({np.around(c, 2)} $\pm$ {np.around(pcov_std[2], 2)}) $nT$", x = 0.1, y = 0.6, transform = axs[0].transAxes)
        axs[0].text(s = f"$\sigma$ = {np.around(sigma, 2)} $nT$", x = 0.1, y = 0.5, transform = axs[0].transAxes)
        axs[0].text(s = f"a/2 = {np.around(a/2, 2)} $nT$", x = 0.1, y = 0.4, transform = axs[0].transAxes)
        [ax.axvline(t_df_init.data, color = "k", linestyle = "--", label = "$t_{DF, init}$") for ax in axs]
        [ax.axvline(t_df, color = "r", linestyle = "--", label = "$t_{DF, new}$") for ax in axs]
        [ax.axvline(t_df_init.data - np.timedelta64(20, "s"), color = "blue", linestyle = "--") for ax in axs]
        [ax.axvline(t_df_init.data + np.timedelta64(10, "s"), color = "blue", linestyle = "--") for ax in axs]

        axs[1].legend()
        make_labels(axs, pos = [0.03, 0.07])
        display.display(plt.gcf())

        while True:
            cont = input("Do you want to fit the dipolarization front manually? (y/n): ").lower()

            if cont == "y":
                t_span = list(input("Enter the time span for the fit in the format t_start, t_end, where t_start and t_end are in seconds: ").split(","))

                try:
                    t_span = [int(t) for t in t_span]
                except:
                    print("Invalid input, please enter the time span in seconds.")
                    continue

                b_syn, a, b, c, sigma, t_df, b_filt, pcov = fit_df(b_avg_filt, t_df_init.data, t_span=t_span)
                plt.cla()
                display.clear_output(wait=True)
                [ax.clear() for ax in axs]

                plot_line(axs[0], b_avg_filt.sel(represent_vec_tot = "z"), label = "Bz")
                plot_line(axs[0], b_syn, label = "Fit")
                axs[0].set_ylabel("B [nT]")

                plot_line(axs[1], dbz_dt)
                axs[1].set_ylabel("dBz/dt [nT/s]")

                axs[0].text(s = f"b = ({np.around(b, 2)} $\pm$ {np.around(pcov_std[1], 2)}) $s$", x = 0.1, y = 0.7, transform = axs[0].transAxes)
                axs[0].text(s = f"a = ({np.around(a, 2)} $\pm$ {np.around(pcov_std[0], 2)}) $nT$", x = 0.1, y = 0.8, transform = axs[0].transAxes)
                axs[0].text(s = f"c = ({np.around(c, 2)} $\pm$ {np.around(pcov_std[2], 2)}) $nT$", x = 0.1, y = 0.6, transform = axs[0].transAxes)
                axs[0].text(s = f"$\sigma$ = {np.around(sigma, 2)} $nT$", x = 0.1, y = 0.5, transform = axs[0].transAxes)
                axs[0].text(s = f"a/2 = {np.around(a/2, 2)} $nT$", x = 0.1, y = 0.4, transform = axs[0].transAxes)
                [ax.axvline(t_df_init.data, color = "k", linestyle = "--", label = "$t_{DF, init}$") for ax in axs]
                [ax.axvline(t_df, color = "r", linestyle = "--", label = "$t_{DF, new}$") for ax in axs]
                [ax.axvline(t_df_init.data - np.timedelta64(abs(t_span[0]), "s"), color = "blue", linestyle = "--") for ax in axs]
                [ax.axvline(t_df_init.data + np.timedelta64(abs(t_span[1]), "s"), color = "blue", linestyle = "--") for ax in axs]

                axs[1].legend()
                make_labels(axs, pos = [0.03, 0.07])
                display.display(plt.gcf())
            elif cont == "n":
                plt.close(fig)
                break
            else:
                print("Invalid input, answer 'y' or 'n'")
                continue

    else:

        b_syn, a, b, c, sigma, t_df, b_filt, pcov = fit_df(b_avg_filt, t_df_init.data)

    return t_df_init, t_max_bz, b_syn, a, b, c, sigma, t_df, b_filt, pcov