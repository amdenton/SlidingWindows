from windowagg.dem_data import Dem_data

import math

import numpy as np

# TODO What is the scale of the elevation, width and height of the image?
# should I take in scale as a parameter? Or derive it from metadata?

# return array of aggregated slope values
def slope(dem_data, pixel_width, pixel_height):
    agg_window_len = 2**dem_data.num_aggre
    xx = (agg_window_len**2 - 1) / 12
    xz = dem_data.xz()
    yz = dem_data.yz()

    return (
        ( np.power(xz, 2) + np.power(yz, 2) )
        /
        ( xx * ( (np.absolute(xz) * pixel_width) + (np.absolute(yz) * pixel_height) ) )
    )

# return array of aggregated slope values
def slope_angle(dem_data, pixel_width, pixel_height):
    agg_window_len = 2**dem_data.num_aggre
    xx = (agg_window_len**2 - 1) / 12
    xz = dem_data.xz()
    yz = dem_data.yz()

    return np.arctan(
        ( np.power(xz, 2) + np.power(yz, 2) )
        /
        ( xx * ( (np.absolute(xz) * pixel_width) + (np.absolute(yz) * pixel_height) ) )
    )

# return array of aggregated angle of steepest descent, calculated as clockwise angle from north
def aspect(dem_data):
    xz = dem_data.xz()
    yz = dem_data.yz()
    return np.arctan(xz / yz) + (-np.sign(xz) * math.pi / 2)

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