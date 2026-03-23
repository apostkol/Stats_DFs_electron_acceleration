from pyrfu import pyrf
import numpy as np

def ps_distance_proxy(B_xyz, p_xyz, alt=False):

    r"""
    Calculate the distance to the plasma sheet using the following equation:
    .. math::
        d = \frac{B_{xy}}{\sqrt{1 + \beta}} = \frac{B_{xy}}{\sqrt{1 + \frac{p}{B^2}}}
    where :math:`B_{xy} = B_x^2 + B_y^2` and :math:`\beta = \frac{p}{B^2}`.
    Parameters
    ----------
    B_xyz : xr.DataArray
        Time series of Magnetic field vector.
    p_xyz : xr.DataArray
        Time series of plasma pressure tensor.
    alt : bool
        If True, use the alternative equation for the distance to the plasma sheet.
        Default is False.
    Returns
    -------
    ps_distance : xr.DataArray
        Distance to the plasma sheet proxy.
    """
    
    B_mag = pyrf.norm(B_xyz)
    beta = pyrf.plasma_beta(B_xyz, p_xyz)*np.float64(p_xyz.attrs["SI_CONVERSION"][:6])
    Bext = np.sqrt(1 + beta) * B_mag

    if alt:
        ps_distance = B_xyz[:, 0]/Bext
    else:
        ps_distance = np.sqrt(B_xyz[:, 0]**2 + B_xyz[:, 1]**2)/Bext
    
    return ps_distance