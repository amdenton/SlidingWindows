from windowagg.dem_data import Dem_data

import math

import numpy as np

# return array of aggregated slope values
def slope(dem_data, pixel_width=1, pixel_height=1):
    w = pixel_width
    h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    xz = dem_data.xz()
    yz = dem_data.yz()

    xx = (agg_window_len**2 - 1) / 12
    b0 = xz / xx
    b1 = yz / xx

    # directional derivative of the following equation
    # in the direction of the positive gradient, derived in mathematica
    # a00(x*w)**2 + 2a10(x*w)(y*h) + a11(y*h)**2 + b0(x*w) + b1(y*h) + cc
    slope = np.sqrt((b1*h)**2 + (b0*w)**2)

    return slope

# return array of aggregated slope values
def slope_angle(dem_data, pixel_width=1, pixel_height=1):
    w = pixel_width
    h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    xz = dem_data.xz()
    yz = dem_data.yz()

    slope_x = (xz * 12) / (agg_window_len**2 - 1)
    slope_y = (yz * 12) / (agg_window_len**2 - 1)
    mag = np.sqrt(np.power(slope_x, 2) + np.power(slope_y, 2))
    x_unit = slope_x / mag
    y_unit = slope_y / mag
    len_opp = (x_unit * slope_x) + (y_unit * slope_y)
    len_adj = np.sqrt( (x_unit * w)**2 + (y_unit * h)**2 )
    slope_angle = np.arctan(len_opp / len_adj)

    return slope_angle

# return array of aggregated angle of steepest descent, calculated as clockwise angle from north
def aspect(dem_data):
    xz = dem_data.xz()
    yz = dem_data.yz()
    
    aspect = (-np.arctan(xz / yz) - (np.sign(yz) * math.pi / 2) + (math.pi / 2)) % (2 * math.pi)
    return aspect

# return array of aggregated profile curvature, second derivative parallel to steepest descent
def profile(dem_data, pixel_width=1, pixel_height=1):
    w = pixel_width
    h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, yyz, xxz, xyz = dem_data.arrays()

    xxxxminusxx2 = (agg_window_len**4 - (5 * agg_window_len**2) + 4) / 180
    xx = (agg_window_len**2 - 1) / 12
    a00 = (xxz - (xx * z)) / xxxxminusxx2
    a10 = xyz / (2 * xx**2)
    a11 = (yyz - (xx * z)) / xxxxminusxx2
    b0 = xz / xx
    b1 = yz / xx

    # directional derivative of the slope of the following equation
    # in the direction of the slope, derived in mathematica
    # a00(x*w)**2 + 2a10(x*w)(y*h) + a11(y*h)**2 + b0(x*w) + b1(y*h) + cc
    profile = (2*(a11*(b1**2)*(h**4) + b0*(w**2)*(2*a10*b1*(h**2) + a00*b0*(w**2)))) / ((b1*h)**2 + (b0*w)**2)

    return profile

# return array of aggregated planform curvature, second derivative perpendicular to steepest descent
def planform(dem_data, pixel_width=1, pixel_height=1):
    w = pixel_width
    h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, yyz, xxz, xyz = dem_data.arrays()
    
    xxxxminusxx2 = (agg_window_len**4 - (5 * agg_window_len**2) + 4) / 180
    xx = (agg_window_len**2 - 1) / 12
    a00 = (xxz - (xx * z)) / xxxxminusxx2
    a10 = xyz / (2 * xx**2)
    a11 = (yyz - (xx * z)) / xxxxminusxx2
    b0 = xz / xx
    b1 = yz / xx

    
    # directional derivative of the slope of the following equation
    # in the direction perpendicular to slope, derived in mathematica
    # a00(x*w)**2 + 2a10(x*w)(y*h) + a11(y*h)**2 + b0(x*w) + b1(y*h) + cc
    planform = (2*h*w*(a11*b0*b1*(h**2) - a10*(b1**2)*(h**2) + a10*(b0**2)*(w**2) - a00*b0*b1*(w**2))) / ((b1*h)**2 + (b0*w)**2)

    return planform

# return array of aggregated standard curvature
def standard(dem_data, pixel_width=1, pixel_height=1):
    w = pixel_width
    h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, yyz, xxz, xyz = dem_data.arrays()
    
    xxxxminusxx2 = (agg_window_len**4 - (5 * agg_window_len**2) + 4) / 180
    xx = (agg_window_len**2 - 1) / 12
    a00 = (xxz - (xx * z)) / xxxxminusxx2
    a10 = xyz / (2 * (xx**2))
    a11 = (yyz - (xx * z)) / xxxxminusxx2
    b0 = xz / xx
    b1 = yz / xx

    # (profile + planform) / 2
    # derived in mathematica
    standard = (a00*b0*(w**3)*(-b1*h + b0*w) + a11*b1*(h**3)*(b1*h + b0*w) + a10*h*w*((-b1**2)*(h**2) + 2*b0*b1*h*w + (b0**2)*(w**2))) / ((b1*h)**2 + (b0*w)**2)

    return standard