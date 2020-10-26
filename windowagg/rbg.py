from windowagg.agg_ops import Agg_ops
import windowagg.aggregation as aggregation
import windowagg.helper as helper

import numpy as np

# i.e. Normalized Difference Vegetation Index
# for viewing live green vegetation
# requires red and infrared bands
# returns floating point array
def ndvi(red_arr, ir_arr):
    red_arr = red_arr.astype(float)
    ir_arr = ir_arr.astype(float)
    return ( (ir_arr - red_arr) / (ir_arr + red_arr) )

# create black and white image
# values greater than or equal to threshold percentage will be white
# threshold: percent in decimal of maximum
# returns array of same data type
# TODO can I assume minimum is always 0, how would I handle it otherwise?
def binary(arr, threshold):
    if (threshold < 0 or threshold > 1):
        raise ValueError('threshold must be between 0 and 1')
    dtype = arr.dtype
    maximum = helper.dtype_max(dtype)
    return np.where(arr < (threshold * maximum), 0, maximum).astype(dtype)

# Do num_aggre aggregations and return the regression slope between two bands
# returns floating point array
def regression(arr_a, arr_b, num_aggre):
    arr_a = arr_a.astype(float)
    arr_b = arr_b.astype(float)
    arr_aa = arr_a**2
    arr_ab = (arr_a * arr_b)

    arr_a = aggregation.aggregate(arr_a, Agg_ops.add_all, num_aggre)
    arr_b = aggregation.aggregate(arr_b, Agg_ops.add_all, num_aggre)
    arr_aa = aggregation.aggregate(arr_aa, Agg_ops.add_all, num_aggre)
    arr_ab = aggregation.aggregate(arr_ab, Agg_ops.add_all, num_aggre)

    # total input pixels aggregated per output pixel
    count = (2**num_aggre)**2

    # regression coefficient, i.e. slope of best fit line
    numerator = count * arr_ab - arr_a * arr_b
    denominator = count * arr_aa - arr_a**2
    # avoid division by zero
    # TODO is this required? Zero only occurs when there is no variance in the a band
    denominator = np.maximum(denominator, 1)
    arr_m = numerator/denominator

    return arr_m

# Do num_aggre aggregations and return the regression slope between two bands
# returns floating point array
def pearson(arr_a, arr_b, num_aggre):
    arr_a = arr_a.astype(float)
    arr_b = arr_b.astype(float)
    arr_aa = arr_a**2
    arr_bb = arr_b**2
    arr_ab = (arr_a * arr_b)

    arr_a = aggregation.aggregate(arr_a, Agg_ops.add_all, num_aggre)
    arr_b = aggregation.aggregate(arr_b, Agg_ops.add_all, num_aggre)
    arr_aa = aggregation.aggregate(arr_aa, Agg_ops.add_all, num_aggre)
    arr_bb = aggregation.aggregate(arr_bb, Agg_ops.add_all, num_aggre)
    arr_ab = aggregation.aggregate(arr_ab, Agg_ops.add_all, num_aggre)

    # total input pixels aggregated per output pixel
    count = (2**num_aggre)**2

    # pearson correlation
    numerator = (count * arr_ab) - (arr_a * arr_b)
    denominator = np.sqrt((count * arr_aa) - arr_a**2) * np.sqrt((count * arr_bb) - arr_b**2)
    # avoid division by zero
    # TODO is this required? Zero only occurs when there is no variance in the a or b bands
    denominator = np.maximum(denominator, 1)
    arr_r = numerator / denominator
    
    return arr_r

# Do num_aggre aggregations and return the regression slope between two bands
# non-vectorized using numpy's polyfit method
# returns floating point array
def regression_brute(arr_a, arr_b, num_aggre):
    arr_a = arr_a.astype(float)
    arr_b = arr_b.astype(float)
    w_out = 2**num_aggre
    y_max =  arr_a.shape[0] - (w_out - 1)
    x_max = arr_a.shape[1] - (w_out - 1)
    arr_m = np.empty([x_max, y_max])
    
    for j in range (y_max):
        for i in range (x_max):
            arr_1 = arr_a[j:(j + w_out), i:(i + w_out)].flatten()
            arr_2 = arr_b[j:(j + w_out), i:(i + w_out)].flatten()
            arr_coef = np.polynomial.polynomial.polyfit(arr_1, arr_2, 1)
            arr_m[j][i] = arr_coef[1]

    return arr_m

# check if an image is black and white or not
# i.e. only contains values of dtype.min and dtype.max
# TODO should min value be arr_in.dtype.min or 0?
# TODO maybe remove this later
def is_binary(arr_in):
    max_val = np.amax(arr_in)
    return ((arr_in==0) | (arr_in==max_val)).all()

# Compute fractal dimension on 2**power_target wide pixel areas
def fractal(arr_in, threshold, num_aggre):
    arr_binary = binary(arr_in, threshold)
    removal_num = (2**num_aggre - 1)
    y_max = arr_binary.shape[0] - removal_num
    x_max = arr_binary.shape[1] - removal_num
    denom_regress = np.empty(num_aggre)
    num_regress = np.empty([(num_aggre), (x_max * y_max)])
    
    for i in range(num_aggre):
        if (i > 0):
            arr_binary = aggregation.aggregate(arr_binary, Agg_ops.maximum, 1, i - 1)

        arr_sum = aggregation.aggregate(arr_binary, Agg_ops.add_all, num_aggre - i, i)

        arr_sum = np.log2(arr_sum)
        denom_regress[i] = i
        num_regress[i, ] = arr_sum.flatten()

    arr_slope = np.polynomial.polynomial.Polynomial.fit(denom_regress, num_regress, 1)[1]
    arr_out = np.reshape(arr_slope, (y_max, x_max))
    return arr_out

# This is for the 3D fractal dimension that is between 2 and 3, but it isn't tested yet
def _boxed_array(arr_in, power_target):
    arr_min = np.amin(arr_in)
    arr_max = np.amax(arr_in)
    arr_out = np.zeros(arr_in.size)
    if (arr_max > arr_min):
        n_boxes = 2**power_target - 1
        buffer = (arr_in - arr_min) / (arr_max - arr_min)
        arr_out = np.floor(n_boxes * buffer)
    return arr_out

# TODO does this need to be binary too? probably not?
# TODO should this have a power_start?
def fractal_3d(arr_in, num_aggre):
    if (num_aggre <= 1):
        raise ValueError('number of aggregations must be greater than one')
    y_max = arr_in.shape[0] - (2**num_aggre - 1)
    x_max = arr_in.shape[1] - (2**num_aggre - 1)
    arr_box = _boxed_array(arr_in, num_aggre).astype(float)
    arr_min = np.array(arr_box)
    arr_max = np.array(arr_box)
    denom_regress = np.empty(num_aggre - 1)
    num_regress = np.empty([(num_aggre - 1), (x_max * y_max)])
    
    # TODO is this supposed to start at 1?
    for i in range(1, num_aggre):
        arr_min = aggregation.aggregate(arr_min, Agg_ops.minimum, 1, (i-1))
        arr_max = aggregation.aggregate(arr_max, Agg_ops.maximum, 1, (i-1))
        arr_sum = aggregation.aggregate((arr_max - arr_min + 1), i, num_aggre, Agg_ops.add_all)
        arr_num = np.log2(arr_sum)
        denom_regress[i - 1] = num_aggre - i
        num_regress[(i - 1), ] = arr_num.flatten()

        # TODO why do we divide by two?
        arr_min /= 2
        arr_max /= 2

    arr_slope = np.polynomial.polynomial.polyfit(denom_regress, num_regress, 1)[1]
    arr_out = np.reshape(arr_slope, (y_max, x_max))
    return arr_out