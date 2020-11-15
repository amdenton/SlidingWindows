from windowagg.agg_ops import Agg_ops
from windowagg.dem_data import Dem_data
import numpy as np
import math

# TODO Do the input arrays have to be converted to float?

# non-vectorized aggregation method
# very slow
# returns floating point array
def aggregate_brute(arr_in, operation, num_aggre=1):
    if (not isinstance(operation, Agg_ops)):
        raise ValueError('operation must be of type Agg_ops')

    arr_in = arr_in.astype(float)
    x_max = arr_in.shape[1]
    y_max = arr_in.shape[0]
    arr_out = np.array(arr_in)

    # iterate through window sizes
    for i in range(num_aggre):
        delta = 2**i
        y_max -= delta
        x_max -= delta
        arr = np.empty([y_max, x_max])

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

    # remove last removal_num rows and columns, they are not aggregate pixels
    removal_num = (2**(num_prev_aggre + num_aggre)) - (2**num_prev_aggre)
    
    if (removal_num > 0):
        arr_out = np.pad(arr_out, (0, removal_num), 'constant')
        arr_out = np.reshape(arr_out, [(y_max - removal_num), x_max])
        arr_out = np.delete(arr_out, np.s_[-removal_num::], 1)
    else:
        arr_out = np.reshape(arr_out, [y_max, x_max])

    return arr_out

def aggregate_dem(dem_data, num_aggre=1):
    if (not isinstance(dem_data, Dem_data)):
        raise ValueError('den_data must be of type Dem_data')

    z, xz, yz, xxz, yyz, xyz = dem_data.arrays()
    num_prev_aggre = dem_data.num_aggre
    window_size = 2**num_prev_aggre

    for _ in range(num_aggre):

        z_sum_all = aggregate(z, Agg_ops.add_all, 1, num_prev_aggre)
        z_sum_bottom = aggregate(z, Agg_ops.add_bottom, 1, num_prev_aggre)
        z_sum_right = aggregate(z, Agg_ops.add_right, 1, num_prev_aggre)
        z_sum_main_diag = aggregate(z, Agg_ops.add_main_diag, 1, num_prev_aggre)

        xz_sum_all = aggregate(xz, Agg_ops.add_all, 1, num_prev_aggre)
        xz_sum_bottom = aggregate(xz, Agg_ops.add_bottom, 1, num_prev_aggre)
        xz_sum_right = aggregate(xz, Agg_ops.add_right, 1, num_prev_aggre)

        yz_sum_all = aggregate(yz, Agg_ops.add_all, 1, num_prev_aggre)
        yz_sum_bottom = aggregate(yz, Agg_ops.add_bottom, 1, num_prev_aggre)
        yz_sum_right = aggregate(yz, Agg_ops.add_right, 1, num_prev_aggre)

        xxz_sum_all = aggregate(xxz, Agg_ops.add_all, 1, num_prev_aggre)

        yyz_sum_all = aggregate(yyz, Agg_ops.add_all, 1, num_prev_aggre)

        xyz_sum_all = aggregate(xyz, Agg_ops.add_all, 1, num_prev_aggre)

        window_size *= 2
        num_prev_aggre += 1

        z = 0.25 * z_sum_all
        xz = 0.25 * (xz_sum_all + (0.25 * window_size * z_sum_right))
        yz = 0.25 * (yz_sum_all + (0.25 * window_size * z_sum_bottom))
        xxz = 0.25 * (xxz_sum_all + (.5 * window_size * xz_sum_right) + (0.0625 * window_size**2 * z_sum_all))
        yyz = 0.25 * (yyz_sum_all + (.5 * window_size * yz_sum_bottom) + (0.0625 * window_size**2 * z_sum_all))
        xyz = 0.25 * (xyz_sum_all + (0.25 * window_size * (xz_sum_bottom + yz_sum_right)) + (0.0625 * window_size**2 * z_sum_main_diag))

    dem_data.set_arrays(z, xz, yz, xxz, yyz, xyz)
    dem_data.num_aggre = num_prev_aggre

def aggregate_dem_brute(dem_data, num_aggre=1):
    if (not isinstance(dem_data, Dem_data)):
        raise ValueError('den_data must be of type Dem_data')

    z, xz, yz, xxz, yyz, xyz = dem_data.get_arrays()
    delta = 2**dem_data.num_aggre
    
    for _ in range(num_aggre):
        x_max = z.shape[1] - delta
        y_max = z.shape[0] - delta
        agg_window_len = delta * 2
        for y in range (y_max):
            for x in range (x_max):
                z_sum_all = z[y, x] + z[y, (x + delta)] + z[(y + delta), x] + z[(y + delta), (x + delta)]
                z_sum_bottom = -z[y, x] - z[y, (x + delta)] + z[(y + delta), x] + z[(y + delta), (x + delta)]
                z_sum_right = -z[y, x] + z[y, (x + delta)] - z[(y + delta), x] + z[(y + delta), (x + delta)]
                z_sum_main_diag = z[y,x] - z[y, (x + delta)] - z[(y + delta), x] + z[(y + delta), (x + delta)]

                xz_sum_all = xz[y, x] + xz[y, (x + delta)] + xz[(y + delta), x] + xz[(y + delta), (x + delta)]
                xz_sum_bottom = -xz[y, x] - xz[y, (x + delta)] + xz[(y + delta), x] + xz[(y + delta), (x + delta)]
                xz_sum_right = -xz[y, x] + xz[y, (x + delta)] - xz[(y + delta), x] + xz[(y + delta), (x + delta)]

                yz_sum_all = yz[y, x] + yz[y, (x + delta)] + yz[(y + delta), x] + yz[(y + delta), (x + delta)]
                yz_sum_bottom = -yz[y, x] - yz[y, (x + delta)] + yz[(y + delta), x] + yz[(y + delta), (x + delta)]
                yz_sum_right = -yz[y, x] + yz[y, (x + delta)] - yz[(y + delta), x] + yz[(y + delta), (x + delta)]

                xxz_sum_all = xxz[y, x] + xxz[y, (x + delta)] + xxz[(y + delta), x] + xxz[(y + delta), (x + delta)]

                yyz_sum_all = yyz[y, x] + yyz[y, (x + delta)] + yyz[(y + delta), x] + yyz[(y + delta), (x + delta)]

                xyz_sum_all = xyz[y, x] + xyz[y, (x + delta)] + xyz[(y + delta), x] + xyz[(y + delta), (x + delta)]

                xz[y, x] = 0.25 * (xz_sum_all + (0.25 * agg_window_len * z_sum_right))
                yz[y, x] = 0.25 * (yz_sum_all + (0.25 * agg_window_len * z_sum_bottom))
                xxz[y, x] = 0.25 * (xxz_sum_all + (.5 * agg_window_len * xz_sum_right) + (0.0625 * agg_window_len**2 * z_sum_all))
                yyz[y, x] = 0.25 * (yyz_sum_all + (.5 * agg_window_len * yz_sum_bottom) + (0.0625 * agg_window_len**2 * z_sum_all))
                xyz[y, x] = 0.25 * (xyz_sum_all + (0.25 * agg_window_len * (xz_sum_bottom + yz_sum_right)) + (0.0625 * agg_window_len**2 * z_sum_main_diag))
                z[y, x] = 0.25 * z_sum_all

        delta *= 2
    
    dem_data.set_arrays(z, xz, yz, xxz, yyz, xyz)
    dem_data.num_aggre += num_aggre
