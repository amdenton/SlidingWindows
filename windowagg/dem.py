from windowagg.dem_data import Dem_data

import math

import numpy as np

# TODO What is the scale of the elevation, width and height of the image?
# should I take in scale as a parameter? Or derive it from metadata?

# return array of aggregated slope values
def slope(dem_data, pixel_width, pixel_height):
    w = pixel_width
    h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    xx = (agg_window_len**2 - 1) / 12
    xz = dem_data.xz()
    yz = dem_data.yz()
    dir_mag = np.sqrt(xz**2 + yz**2)

    return (
        ( np.power(xz, 2) + np.power(yz, 2) )
        /
        (xx * (np.abs((w * xz) / dir_mag) + np.abs((h * yz) / dir_mag)))
    )

# return array of aggregated slope values
def slope_angle(dem_data, pixel_width, pixel_height):
    w = pixel_width
    h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    xx = (agg_window_len**2 - 1) / 12
    xz = dem_data.xz()
    yz = dem_data.yz()
    dir_mag = np.sqrt(xz**2 + yz**2)

    return np.arctan(
        ( np.power(xz, 2) + np.power(yz, 2) )
        /
        (xx * (np.abs((w * xz) / dir_mag) + np.abs((h * yz) / dir_mag)))
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

    xxxx = ((3 * agg_window_len**4) - (10 * agg_window_len**2) + 7) / 240
    xx = (agg_window_len**2 - 1) / 12
    a00 = (xxz - (xx * z)) / (xxxx - xx**2)
    a10 = xyz / (2 * xx**2)
    a11 = (yyz - (xx * z)) / (xxxx - xx**2)
    dir_mag = np.sqrt(xz**2 + yz**2)

    profile = (
        (2 * ((xz**2 * a00) + (2 * xz * yz * a10) + (yz**2 * a11)))
        /
        ((xz**2 + yz**2) * (np.abs((w * xz) / dir_mag) + np.abs((h * yz) / dir_mag)))
    )

    return profile

# return array of aggregated planform curvature, second derivative perpendicular to steepest descent
def planform(dem_data, pixel_width=1, pixel_height=1):
    w = pixel_width
    h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, yyz, xxz, xyz = dem_data.arrays()
    
    xxxx = ((3 * agg_window_len**4) - (10 * agg_window_len**2) + 7) / 240
    xx = (agg_window_len**2 - 1) / 12
    a00 = (xxz - (xx * z)) / (xxxx - xx**2)
    a10 = xyz / (2 * xx**2)
    a11 = (yyz - (xx * z)) / (xxxx - xx**2)
    dir_mag = np.sqrt(xz**2 + yz**2)

    planform = (
        (2 * ((xz**2 * a00) - (2 * xz * yz * a10) + (yz**2 * a11)))
        /
        ((xz**2 + yz**2) * (np.abs((w * xz) / dir_mag) + np.abs((h * yz) / dir_mag)))
    )

    return planform

# return array of aggregated standard curvature
def standard(dem_data, pixel_width=1, pixel_height=1):
    w = pixel_width
    h = pixel_height
    agg_window_len = 2**dem_data.num_aggre
    z, xz, yz, yyz, xxz, _xyz = dem_data.arrays()
    
    xxxx = ((3 * agg_window_len**4) - (10 * agg_window_len**2) + 7) / 240
    xx = (agg_window_len**2 - 1) / 12
    a00 = (xxz - (xx * z)) / (xxxx - xx**2)
    a11 = (yyz - (xx * z)) / (xxxx - xx**2)
    dir_mag = np.sqrt(xz**2 + yz**2)

    standard = (
        (2 * ((xz**2 * a00) + (yz**2 * a11)))
        /
        ((xz**2 + yz**2) * (np.abs((w * xz) / dir_mag) + np.abs((h * yz) / dir_mag)))
    )

    return standard