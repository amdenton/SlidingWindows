from windowagg.agg_ops import Agg_ops
from windowagg.dem_data import Dem_data
import numpy as np
import math

# TODO Do the input arrays have to be converted to float?

# non-vectorized aggregation method
# very slow
# returns floating point array
def aggregate_brute(arr_in, operation, num_aggre=1, num_prev_aggre=0):
    if (len(arr_in.shape) != 2):
        raise ValueError('Array must be 2 dimensional')
    if (not isinstance(operation, Agg_ops)):
        raise ValueError('operation must be of type Agg_ops')

    x_max = arr_in.shape[1]
    y_max = arr_in.shape[0]
    removal_num = 2**(num_prev_aggre + num_aggre) - 2**num_prev_aggre
    if ((removal_num > x_max) or (removal_num > y_max)):
        raise ValueError('Image size is too small to aggregate ' + str(num_aggre) + ' times')
    arr_out = arr_in
    delta = 2**num_prev_aggre

    for i in range(num_aggre):
        delta = 2**i
        y_max -= delta
        x_max -= delta
        arr = np.empty([y_max, x_max], dtype=arr_out.dtype)

        # iterate through pixels
        for j in range (y_max):
            for i in range (x_max):
                if (operation == Agg_ops.add_all):
                    arr[j, i] = arr_out[j, i] + arr_out[j, (i + delta)] + arr_out[(j + delta), i] + arr_out[(j + delta), (i + delta)]
                if (operation == Agg_ops.add_bottom):
                    arr[j, i] = -arr_out[j, i] - arr_out[j, (i + delta)] + arr_out[(j + delta), i] + arr_out[(j + delta), (i + delta)]
                if (operation == Agg_ops.add_right):
                    arr[j, i] = -arr_out[j, i] + arr_out[j, (i + delta)] - arr_out[(j + delta), i] + arr_out[(j + delta), (i + delta)]
                if (operation == Agg_ops.add_main_diag):
                    arr[j, i] = arr_out[j, i] - arr_out[j, (i + delta)] - arr_out[j+delta, i] + arr_out[(j + delta), (i + delta)]
                elif (operation == Agg_ops.maximum):
                    arr[j, i] = max(max(max(arr_out[j, i], arr_out[j, (i + delta)]), arr_out[(j + delta), i]), arr_out[(j + delta), (i + delta)])
                elif (operation == Agg_ops.minimum):
                    arr[j, i] = min(min(min(arr_out[j, i], arr_out[j, (i + delta)]), arr_out[(j + delta), i]), arr_out[(j + delta), (i + delta)])
        arr_out = arr
    return arr_out

# Aggregate arr_in
# Assuming 2*prev_aggre size windows are already aggregated
# returns floating point array
def aggregate(arr_in, operation, num_aggre=1, num_prev_aggre=0):
    if (len(arr_in.shape) != 2):
        raise ValueError('Array must be 2 dimensional')
    if (not isinstance(operation, Agg_ops)):
        raise ValueError('operation must be of type Agg_ops')

    y_max = arr_in.shape[0]
    x_max = arr_in.shape[1]
    removal_num = 2**(num_prev_aggre + num_aggre) - 2**num_prev_aggre
    if ((removal_num > x_max) or (removal_num > y_max)):
        raise ValueError('Image size is too small to aggregate ' + str(num_aggre) + ' times')
    arr_out = arr_in.flatten()
    delta = 2**num_prev_aggre
    
    # iterate through sliding window sizes
    for _ in range(num_aggre):
        size = arr_out.size

        # create offset slices of the array to aggregate elements
        top_left = arr_out[0 : (size - ((delta * x_max) + delta))]
        top_right = arr_out[delta: size - (x_max * delta)]
        bottom_left = arr_out[(delta * x_max) : (size - delta)]
        bottom_right = arr_out[((delta * x_max) + delta) : size]

        if operation == Agg_ops.add_all:
            arr_out = top_left + top_right + bottom_left + bottom_right
        elif operation == Agg_ops.add_bottom:
            arr_out = -top_left - top_right + bottom_left + bottom_right
        elif operation == Agg_ops.add_right:
            arr_out = -top_left + top_right - bottom_left + bottom_right
        elif operation == Agg_ops.add_main_diag:
            arr_out = top_left - top_right - bottom_left + bottom_right
        elif operation == Agg_ops.maximum:
            arr_out = np.maximum(np.maximum(np.maximum(top_left, top_right), bottom_left), bottom_right)
        elif operation == Agg_ops.minimum:
            arr_out = np.minimum(np.minimum(np.minimum(top_left, top_right), bottom_left), bottom_right)
        
        delta *= 2
    
    if (removal_num > 0):
        arr_out = np.pad(arr_out, (0, removal_num), 'constant')
        arr_out = np.reshape(arr_out, [(y_max - removal_num), x_max])
        arr_out = np.delete(arr_out, np.s_[-removal_num::], 1)
    else:
        arr_out = np.reshape(arr_out, [y_max, x_max])

    return arr_out

def aggregate_dem(dem_data, num_aggre=1):
    if (not isinstance(dem_data, Dem_data)):
        raise ValueError('dem_data must be of type Dem_data')

    z, xz, yz, xxz, yyz, xyz = dem_data.arrays()
    num_prev_aggre = dem_data.num_aggre
    window_size = 2**num_prev_aggre

    for _ in range(num_aggre):
        window_size *= 2

        z_sum_all = aggregate(z, Agg_ops.add_all, 1, num_prev_aggre)

        new_z = 0.25 * z_sum_all

        xxz_sum_all = aggregate(xxz, Agg_ops.add_all, 1, num_prev_aggre)
        xz_sum_right = aggregate(xz, Agg_ops.add_right, 1, num_prev_aggre)
        new_xxz = 0.25 * (xxz_sum_all + (.5 * window_size * xz_sum_right) + (0.0625 * window_size**2 * z_sum_all))
        del xxz_sum_all
        del xz_sum_right

        yyz_sum_all = aggregate(yyz, Agg_ops.add_all, 1, num_prev_aggre)
        yz_sum_bottom = aggregate(yz, Agg_ops.add_bottom, 1, num_prev_aggre)
        new_yyz = 0.25 * (yyz_sum_all + (.5 * window_size * yz_sum_bottom) + (0.0625 * window_size**2 * z_sum_all))
        del yyz_sum_all
        del yz_sum_bottom

        del z_sum_all

        xz_sum_all = aggregate(xz, Agg_ops.add_all, 1, num_prev_aggre)
        z_sum_right = aggregate(z, Agg_ops.add_right, 1, num_prev_aggre)
        new_xz = 0.25 * (xz_sum_all + (0.25 * window_size * z_sum_right))
        del xz_sum_all
        del z_sum_right
        
        yz_sum_all = aggregate(yz, Agg_ops.add_all, 1, num_prev_aggre)
        z_sum_bottom = aggregate(z, Agg_ops.add_bottom, 1, num_prev_aggre)
        new_yz = 0.25 * (yz_sum_all + (0.25 * window_size * z_sum_bottom))
        del yz_sum_all
        del z_sum_bottom

        xyz_sum_all = aggregate(xyz, Agg_ops.add_all, 1, num_prev_aggre)
        xz_sum_bottom = aggregate(xz, Agg_ops.add_bottom, 1, num_prev_aggre)
        yz_sum_right = aggregate(yz, Agg_ops.add_right, 1, num_prev_aggre)
        z_sum_main_diag = aggregate(z, Agg_ops.add_main_diag, 1, num_prev_aggre)
        new_xyz = 0.25 * (xyz_sum_all + (0.25 * window_size * (xz_sum_bottom + yz_sum_right)) + (0.0625 * window_size**2 * z_sum_main_diag))
        del xyz_sum_all
        del xz_sum_bottom
        del yz_sum_right
        del z_sum_main_diag

        z = new_z
        xz = new_xz
        yz = new_yz
        xxz = new_xxz
        yyz = new_yyz
        xyz = new_xyz
        num_prev_aggre += 1

    dem_data.set_arrays(z, xz, yz, xxz, yyz, xyz)
    dem_data.num_aggre = num_prev_aggre

def aggregate_z_brute(z, num_aggre=1):
    y_max = z.shape[0]
    x_max = z.shape[1]
    removal_num = 2**num_aggre - 1
    if ((removal_num > x_max) or (removal_num > y_max)):
        raise ValueError('Image size is too small to aggregate ' + str(num_aggre) + ' times')

    z_average = np.zeros([x_max - removal_num, y_max - removal_num], dtype=z.dtype)

    for y in range(z_average.shape[0]):
        for x in range(z_average.shape[1]):
            for y_agg in range(2**num_aggre):
                for x_agg in range(2**num_aggre):
                    z_average[y, x] += z[(y + y_agg), (x + x_agg)]

            z_average[y, x] /= (2**num_aggre)**2

    return z_average

def aggregate_xz_brute(z, num_aggre=1):
    y_max = z.shape[0]
    x_max = z.shape[1]
    extremum = 2**(num_aggre - 1) - .5
    removal_num = 2**num_aggre - 1
    if ((removal_num > x_max) or (removal_num > y_max)):
        raise ValueError('Image size is too small to aggregate ' + str(num_aggre) + ' times')

    xz = np.zeros([x_max - removal_num, y_max - removal_num], dtype=z.dtype)

    for y in range(xz.shape[0]):
        for x in range(xz.shape[1]):
            for y_agg in range(2**num_aggre):
                x_val = -extremum

                for x_agg in range(2**num_aggre):
                    xz[y, x] += z[(y + y_agg), (x + x_agg)] * x_val
                    x_val += 1

            xz[y, x] /= (2**num_aggre)**2

    return xz

def aggregate_yz_brute(z, num_aggre=1):
    y_max = z.shape[0]
    x_max = z.shape[1]
    extremum = 2**(num_aggre - 1) - .5
    removal_num = 2**num_aggre - 1
    if ((removal_num > x_max) or (removal_num > y_max)):
        raise ValueError('Image size is too small to aggregate ' + str(num_aggre) + ' times')

    yz = np.zeros([x_max - removal_num, y_max - removal_num], dtype=z.dtype)

    for y in range(yz.shape[0]):
        for x in range(yz.shape[1]):
            for x_agg in range(2**num_aggre):
                y_val = -extremum

                for y_agg in range(2**num_aggre):
                    yz[y, x] += z[(y + y_agg), (x + x_agg)] * y_val
                    y_val += 1
                    
            yz[y, x] /= (2**num_aggre)**2

    return yz
        
def aggregate_xxz_brute(z, num_aggre=1):
    y_max = z.shape[0]
    x_max = z.shape[1]
    extremum = 2**(num_aggre - 1) - .5
    removal_num = 2**num_aggre - 1
    if ((removal_num > x_max) or (removal_num > y_max)):
        raise ValueError('Image size is too small to aggregate ' + str(num_aggre) + ' times')

    xxz = np.zeros([x_max - removal_num, y_max - removal_num], dtype=z.dtype)

    for y in range(xxz.shape[0]):
        for x in range(xxz.shape[1]):
            for y_agg in range(2**num_aggre):
                x_val = -extremum

                for x_agg in range(2**num_aggre):
                    xxz[y, x] += z[(y + y_agg), (x + x_agg)] * x_val**2
                    x_val += 1
                    
            xxz[y, x] /= (2**num_aggre)**2

    return xxz

def aggregate_yyz_brute(z, num_aggre=1):
    y_max = z.shape[0]
    x_max = z.shape[1]
    extremum = 2**(num_aggre - 1) - .5
    removal_num = 2**num_aggre - 1
    if ((removal_num > x_max) or (removal_num > y_max)):
        raise ValueError('Image size is too small to aggregate ' + str(num_aggre) + ' times')

    yyz = np.zeros([x_max - removal_num, y_max - removal_num], dtype=z.dtype)

    for y in range(yyz.shape[0]):
        for x in range(yyz.shape[1]):
            for x_agg in range(2**num_aggre):
                y_val = -extremum

                for y_agg in range(2**num_aggre):
                    yyz[y, x] += z[(y + y_agg), (x + x_agg)] * y_val**2
                    y_val += 1
                    
            yyz[y, x] /= (2**num_aggre)**2

    return yyz

def aggregate_xyz_brute(z, num_aggre=1):
    y_max = z.shape[0]
    x_max = z.shape[1]
    extremum = 2**(num_aggre - 1) - .5
    removal_num = 2**num_aggre - 1
    if ((removal_num > x_max) or (removal_num > y_max)):
        raise ValueError('Image size is too small to aggregate ' + str(num_aggre) + ' times')

    xyz = np.zeros([x_max - removal_num, y_max - removal_num], dtype=z.dtype)

    for y in range(xyz.shape[0]):
        for x in range(xyz.shape[1]):
            y_val = -extremum

            for y_agg in range(2**num_aggre):
                x_val = -extremum

                for x_agg in range(2**num_aggre):
                    xyz[y, x] += z[(y + y_agg), (x + x_agg)] * x_val * y_val
                    x_val += 1

                y_val += 1
                    
            xyz[y, x] /= (2**num_aggre)**2

    return xyz