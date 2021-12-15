"""
Last updated on Tue Dec 14

@authors: Anne Denton, David Schwarz, Rahul Gomes

License information:
https://opensource.org/licenses/GPL-3.0
"""
import math

import numpy as np

def slope(dem_data):
    agg_window_len = 2**dem_data.num_aggre
    inv_xx = 12. / (agg_window_len**2 - 1) 
    xz = dem_data.xz()
    yz = dem_data.yz()

    with np.errstate(invalid='ignore'):
        slope = (np.sqrt(xz**2 + yz**2) * inv_xx )
        # when (xz == yz == 0) result is NaN
        slope[np.isnan(slope)] = 0
        slope = np.arctan(slope)*180/math.pi

    return slope

def profile(dem_data):
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, xxz, yyz, xyz = dem_data.arrays()

    xx = (agg_window_len**2 - 1) / 12
    xxxx_xx2 = (agg_window_len**4 - 5 * agg_window_len**2 + 4) / 180
    p = xz/xx
    q = yz/xx
    r = 2 * (xxz - (xx * z)) / xxxx_xx2
    s = xyz / (xx**2)
    t = 2 * (yyz - (xx * z)) / xxxx_xx2

    with np.errstate(invalid='ignore'):
        profile = -100*np.divide((p*p*r + 2*p*q*s + q*q*t),(p*p + q*q))
        profile[np.isnan(profile)] = 0

    return profile

def proper_profile(dem_data):
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, xxz, yyz, xyz = dem_data.arrays()

    xx = (agg_window_len**2 - 1) / 12
    xxxx_xx2 = (agg_window_len**4 - 5 * agg_window_len**2 + 4) / 180
    p = xz/xx
    q = yz/xx
    r = 2 * (xxz - (xx * z)) / xxxx_xx2
    s = xyz / (xx**2)
    t = 2 * (yyz - (xx * z)) / xxxx_xx2

    with np.errstate(invalid='ignore'):
        proper_profile = -100*np.divide((p*p*r + 2*p*q*s + q*q*t),(p*p + q*q)*np.power(np.sqrt(1+p*p+q*q),3))
        proper_profile[np.isnan(proper_profile)] = 0

    return proper_profile

def tangential(dem_data):
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, xxz, yyz, xyz = dem_data.arrays()
    
    xx = (agg_window_len**2 - 1) / 12
    xxxx_xx2 = (agg_window_len**4 - 5 * agg_window_len**2 + 4) / 180
    p = xz/xx
    q = yz/xx
    r = 2*(xxz-xx*z)/xxxx_xx2
    s = xyz/xx**2
    t = 2*(yyz-xx*z)/xxxx_xx2

    with np.errstate(invalid='ignore'):
        tangential = -100*(q*q*r-2*p*q*s+p*p*t)/(p*p+q*q)
        tangential[np.isnan(tangential)] = 0

    return tangential

def proper_tangential(dem_data):
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, xxz, yyz, xyz = dem_data.arrays()
    
    xx = (agg_window_len**2 - 1) / 12
    xxxx_xx2 = (agg_window_len**4 - 5 * agg_window_len**2 + 4) / 180
    p = xz/xx
    q = yz/xx
    r = 2*(xxz-xx*z)/xxxx_xx2
    s = xyz/xx**2
    t = 2*(yyz-xx*z)/xxxx_xx2

    with np.errstate(invalid='ignore'):
        proper_tangential = -100*(q*q*r-2*p*q*s+p*p*t)/((p*p+q*q) * np.sqrt(1+p*p+q*q))
        # when (xz == yz == 0) result is NaN
        proper_tangential[np.isnan(proper_tangential)] = 0

    return proper_tangential

def contour(dem_data):
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, xxz, yyz, xyz = dem_data.arrays()
    
    xx = (agg_window_len**2 - 1) / 12
    xxxx_xx2 = (agg_window_len**4 - 5 * agg_window_len**2 + 4) / 180
    p = xz/xx
    q = yz/xx
    r = 2*(xxz-xx*z)/xxxx_xx2
    s = xyz/xx**2
    t = 2*(yyz-xx*z)/xxxx_xx2

    with np.errstate(invalid='ignore'):
        contour = -100*(q*q*r-2*p*q*s+p*p*t)/np.power(np.sqrt(p*p+q*q),3)
        # when (xz == yz == 0) result is NaN
        contour[np.isnan(contour)] = 0

    return contour

