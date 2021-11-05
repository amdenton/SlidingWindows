from windowagg.dem_data import Dem_data

import math

import numpy as np

def slope_simple(dem_data):
    print('dem_data')
    print(dem_data)
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

def profile_simple(dem_data):
#    w = pixel_width
#    h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, xxz, yyz, xyz = dem_data.arrays()

#    xxxx = ((3 * agg_window_len**4) - (10 * agg_window_len**2) + 7) / 240
    xx = (agg_window_len**2 - 1) / 12
    xxxx_xx2 = (agg_window_len**4 - 5 * agg_window_len**2 + 4) / 180
    p = xz/xx
    q = yz/xx
    r = 2 * (xxz - (xx * z)) / xxxx_xx2
    s = xyz / (xx**2)
    t = 2 * (yyz - (xx * z)) / xxxx_xx2
#    a00 = (xxz - (xx * z)) / xxxx_xx2
#    a10 = xyz / (2 * xx**2)
#    a11 = (yyz - (xx * z)) / xxxx_xx2

    with np.errstate(invalid='ignore'):
        profile = -np.divide((p*p*r + 2*p*q*s + q*q*t),(p*p + q*q)*np.power(np.sqrt(1+p*p+q*q),3))
#        profile = -np.divide((2 * ((xz**2 * a00) + (2 * xz * yz * a10) + (yz**2 * a11))),(xz**2 + yz**2))
#            ((xz**2 + yz**2) * np.sqrt(((w**2 * xz**2) + (h**2 * yz**2)) / (xz**2 + yz**2)))
        # when (xz == yz == 0) result is NaN
        profile[np.isnan(profile)] = 0

    return profile

def horizontal_simple(dem_data):
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, xxz, yyz, xyz = dem_data.arrays()
    
    xx = (agg_window_len**2 - 1) / 12
    xxxx_xx2 = (agg_window_len**4 - 5 * agg_window_len**2 + 4) / 180
    p = xz/xx
    q = yz/xx
    r = 2*(xxz-xx*z)/xxxx_xx2
    s = xyz/xx**2
    t = 2*(yyz-xx*z)/xxxx_xx2
    #a00 = (xxz - (xx * z)) / xxxx_xx2
    #a10 = xyz / (2 * xx**2)
    #a11 = (yyz - (xx * z)) / xxxx_xx2

    with np.errstate(invalid='ignore'):
        horizontal = -(q*q*r-2*p*q*s+p*p*t)/((p*p+q*q) * np.sqrt(1+p*p+q*q))
        #planform = ((2 * ((xz**2 * a11) - (2 * xz * yz * a10) + (yz**2 * a00)) / (xz**2 + yz**2)))
#            /((xz**2 + yz**2) * np.sqrt(((w**2 * xz**2) + (h**2 * yz**2)) / (xz**2 + yz**2))))
        # when (xz == yz == 0) result is NaN
        horizontal[np.isnan(horizontal)] = 0


    return planform

def combination_simple(dem_data):
#    w = pixel_width
#    h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, xxz, yyz, xyz = dem_data.arrays()

#    xxxx = ((3 * agg_window_len**4) - (10 * agg_window_len**2) + 7) / 240
    xx = (agg_window_len**2 - 1) / 12
    inv_xx = 1/xx
    inv_xxxx_xx2 = 180/(agg_window_len**4 - 5 * agg_window_len**2 + 4) 
    p = xz*inv_xx
    q = yz*inv_xx
    r = 2*(xxz-xx*z)*inv_xxxx_xx2
    s = xyz*inv_xx**2
    t = 2*(yyz-xx*z)*inv_xxxx_xx2
    #a00 = (xxz - (xx * z)) / xxxx_xx2
    #a10 = xyz / (2 * xx**2)
    #a11 = (yyz - (xx * z)) / xxxx_xx2

    with np.errstate(invalid='ignore'):
        slope = (np.sqrt(xz**2 + yz**2) * inv_xx )
        # when (xz == yz == 0) result is NaN
        slope[np.isnan(slope)] = 0
        slope = np.arctan(slope)*180/math.pi
        profile = -np.divide((p*p*r + 2*p*q*s + q*q*t),(p*p + q*q)*np.power(np.sqrt(1+p*p+q*q),3))
        profile[np.isnan(profile)] = 0
        horizontal = -(q*q*r-2*p*q*s+p*p*t)/((p*p+q*q) * np.sqrt(1+p*p+q*q))
        horizontal[np.isnan(horizontal)] = 0

    return slope, profile, horizontal


# return array of aggregated slope values
def slope(dem_data, pixel_width, pixel_height):
    w = pixel_width
    h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    xx = (agg_window_len**2 - 1) / 12
    xz = dem_data.xz()
    yz = dem_data.yz()

    with np.errstate(invalid='ignore'):
        slope = (
            np.sqrt(xz**2 + yz**2)
            /
            (xx * np.sqrt(((w**2 * xz**2) + (h**2 * yz**2)) / (xz**2 + yz**2)))
        )
        # when (xz == yz == 0) result is NaN
        slope[np.isnan(slope)] = 0
        slope = np.arctan(slope)*180/math.pi

    return slope

# return array of aggregated angle of steepest descent, calculated as clockwise angle from north
def aspect(dem_data):
    xz = dem_data.xz()
    yz = dem_data.yz()
    dtype = xz.dtype

    aspect = (np.arctan2(-yz, -xz, dtype=dtype) + (math.pi / 2)) % (2 * math.pi)
    # TODO change this to something more appropriate for nodata
    # handle different noData in dtype conversion
    aspect[(xz == 0) & (yz == 0)] = 0

    return aspect

# return array of aggregated profile curvature, second derivative parallel to steepest descent
def profile(dem_data, pixel_width=1, pixel_height=1):
#    w = pixel_width
#    h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, xxz, yyz, xyz = dem_data.arrays()

#    xxxx = ((3 * agg_window_len**4) - (10 * agg_window_len**2) + 7) / 240
    xx = (agg_window_len**2 - 1) / 12
    xxxx_xx2 = (agg_window_len**4 - 5 * agg_window_len**2 + 4) / 180
    p = xz/xx
    q = yz/xx
    r = 2 * (xxz - (xx * z)) / xxxx_xx2
    s = xyz / (xx**2)
    t = 2 * (yyz - (xx * z)) / xxxx_xx2
#    a00 = (xxz - (xx * z)) / xxxx_xx2
#    a10 = xyz / (2 * xx**2)
#    a11 = (yyz - (xx * z)) / xxxx_xx2

    with np.errstate(invalid='ignore'):
        profile = -np.divide((p*p*r + 2*p*q*s + q*q*t),(p*p + q*q)*np.power(np.sqrt(1+p*p+q*q),3))
#        profile = -np.divide((2 * ((xz**2 * a00) + (2 * xz * yz * a10) + (yz**2 * a11))),(xz**2 + yz**2))
#            ((xz**2 + yz**2) * np.sqrt(((w**2 * xz**2) + (h**2 * yz**2)) / (xz**2 + yz**2)))
        # when (xz == yz == 0) result is NaN
        profile[np.isnan(profile)] = 0

#    print('profile')
#    print(profile)
    return profile

# return array of aggregated planform curvature, second derivative perpendicular to steepest descent
def planform(dem_data, pixel_width=1, pixel_height=1):
    #w = pixel_width
    #h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, xxz, yyz, xyz = dem_data.arrays()
    
    #xxxx = ((3 * agg_window_len**4) - (10 * agg_window_len**2) + 7) / 240
    xx = (agg_window_len**2 - 1) / 12
    xxxx_xx2 = (agg_window_len**4 - 5 * agg_window_len**2 + 4) / 180
    p = xz/xx
    q = yz/xx
    r = 2*(xxz-xx*z)/xxxx_xx2
    s = xyz/xx**2
    t = 2*(yyz-xx*z)/xxxx_xx2
    #a00 = (xxz - (xx * z)) / xxxx_xx2
    #a10 = xyz / (2 * xx**2)
    #a11 = (yyz - (xx * z)) / xxxx_xx2

    with np.errstate(invalid='ignore'):
        planform = -(q*q*r-2*p*q*s+p*p*t)/np.power(np.sqrt(p*p+q*q),3)
        #planform = ((2 * ((xz**2 * a11) - (2 * xz * yz * a10) + (yz**2 * a00)) / (xz**2 + yz**2)))
#            /((xz**2 + yz**2) * np.sqrt(((w**2 * xz**2) + (h**2 * yz**2)) / (xz**2 + yz**2))))
        # when (xz == yz == 0) result is NaN
        planform[np.isnan(planform)] = 0

    return planform

# return array of aggregated planform curvature, second derivative perpendicular to steepest descent
def horizontal(dem_data, pixel_width=1, pixel_height=1):
    #w = pixel_width
    #h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, xxz, yyz, xyz = dem_data.arrays()
    
    #xxxx = ((3 * agg_window_len**4) - (10 * agg_window_len**2) + 7) / 240
    xx = (agg_window_len**2 - 1) / 12
    xxxx_xx2 = (agg_window_len**4 - 5 * agg_window_len**2 + 4) / 180
    p = xz/xx
    q = yz/xx
    r = 2*(xxz-xx*z)/xxxx_xx2
    s = xyz/xx**2
    t = 2*(yyz-xx*z)/xxxx_xx2
    #a00 = (xxz - (xx * z)) / xxxx_xx2
    #a10 = xyz / (2 * xx**2)
    #a11 = (yyz - (xx * z)) / xxxx_xx2

    with np.errstate(invalid='ignore'):
        horizontal = -(q*q*r-2*p*q*s+p*p*t)/((p*p+q*q) * np.sqrt(1+p*p+q*q))
        #planform = ((2 * ((xz**2 * a11) - (2 * xz * yz * a10) + (yz**2 * a00)) / (xz**2 + yz**2)))
#            /((xz**2 + yz**2) * np.sqrt(((w**2 * xz**2) + (h**2 * yz**2)) / (xz**2 + yz**2))))
        # when (xz == yz == 0) result is NaN
        horizontal[np.isnan(horizontal)] = 0

    return horizontal


# return array of aggregated standard curvature
def standard(dem_data, pixel_width=1, pixel_height=1):
#    w = pixel_width
#    h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, xxz, yyz, xyz = dem_data.arrays()
    
#    xxxx = ((3 * agg_window_len**4) - (10 * agg_window_len**2) + 7) / 240
    xx = (agg_window_len**2 - 1) / 12
    xxxx_xx = (agg_window_len**4 - 5 * agg_window_len**2 + 4) / 180
#    a00 = (xxz - (xx * z)) / (xxxx - xx**2)
#    a11 = (yyz - (xx * z)) / (xxxx - xx**2)

    with np.errstate(invalid='ignore'):
#       The factor of two compensates the factor of 1/2 in the Taylor expansion
        standard = 2* (0.5*(xxz+yyz) - xx*z) / xxxx_xx
#            np.sqrt(((w**2 * xz**2) + (h**2 * yz**2)) / (xz**2 + yz**2))
#        )
        # when (xz == yz == 0) result is NaN
        standard[np.isnan(standard)] = 0

    return standard