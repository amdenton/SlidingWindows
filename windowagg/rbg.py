from windowagg.agg_ops import Agg_ops
import windowagg.aggregation as aggregation
import windowagg.helper as helper
import windowagg.config as config

import numpy as np

# Normalized Difference Vegetation Index
def ndvi(arr_red, arr_ir):
    return ( (arr_ir - arr_red) / (arr_ir + arr_red) )

# Black and white image
# TODO should this handle negative numbers?
def binary(arr, threshold):
    if (threshold < 0 or threshold > 1):
        raise ValueError('threshold must be between 0 and 1')

    dtype = arr.dtype
    dtype_maximum = helper.dtype_max(dtype)
    maximum = np.amax(arr)
    minimum = np.amin(arr)
    threshold_val = (threshold * (maximum - minimum)) + minimum
    return np.where(arr < threshold_val, 0, dtype_maximum)

# Do num_aggre aggregations and return the regression slope between two arrays
def regression(arr_x, arr_y, num_aggre):
    arr_x = arr_x.astype(config.work_dtype)
    arr_y = arr_y.astype(config.work_dtype)
    arr_xx = arr_x**2
    arr_xy = (arr_x * arr_y)

    arr_x = aggregation.aggregate(arr_x, Agg_ops.add_all, num_aggre)
    arr_y = aggregation.aggregate(arr_y, Agg_ops.add_all, num_aggre)
    arr_xx = aggregation.aggregate(arr_xx, Agg_ops.add_all, num_aggre)
    arr_xy = aggregation.aggregate(arr_xy, Agg_ops.add_all, num_aggre)

    pixel_count = (2**num_aggre)**2

    # regression coefficient, i.e. slope of best fit line
    numerator = pixel_count * arr_xy - arr_x * arr_y
    denominator = pixel_count * arr_xx - arr_x**2

    with np.errstate(divide='ignore'):
        arr_m = numerator/denominator
        # TODO how should infinity be handled?
        arr_m[np.isinf(arr_m)] = helper.dtype_max(arr_m.dtype)

    return arr_m

# Do num_aggre aggregations and return the Pearson Correlation coefficient between two bands
def pearson(arr_x, arr_y, num_aggre):
    arr_x = arr_x.astype(config.work_dtype)
    arr_y = arr_y.astype(config.work_dtype)
    arr_xx = arr_x**2
    arr_yy = arr_y**2
    arr_xy = (arr_x * arr_y)

    arr_x = aggregation.aggregate(arr_x, Agg_ops.add_all, num_aggre)
    arr_y = aggregation.aggregate(arr_y, Agg_ops.add_all, num_aggre)
    arr_xx = aggregation.aggregate(arr_xx, Agg_ops.add_all, num_aggre)
    arr_yy = aggregation.aggregate(arr_yy, Agg_ops.add_all, num_aggre)
    arr_xy = aggregation.aggregate(arr_xy, Agg_ops.add_all, num_aggre)

    # total input pixels aggregated per output pixel
    pixel_count = (2**num_aggre)**2

    # pearson correlation
    numerator = (pixel_count * arr_xy) - (arr_x * arr_y)
    denominator = np.sqrt((pixel_count * arr_xx) - arr_x**2) * np.sqrt((pixel_count * arr_yy) - arr_y**2)

    with np.errstate(divide='ignore'):
        arr_r = numerator/denominator
        # TODO how should infinity be handled?
        arr_r[np.isinf(arr_r)] = helper.dtype_max(arr_r.dtype)
    
    return arr_r

# Do num_aggre aggregations and return the regression slope between two bands
def regression_brute(arr_x, arr_y, num_aggre):
    arr_x = arr_x.astype(config.work_dtype)
    arr_y = arr_y.astype(config.work_dtype)
    delta = 2**num_aggre
    y_max =  arr_x.shape[0] - (delta - 1)
    x_max = arr_x.shape[1] - (delta - 1)
    arr_m = np.empty([y_max, x_max])
    
    for y in range (y_max):
        for x in range (x_max):
            x_slice = arr_x[y:(y + delta), x:(x + delta)].flatten()
            y_slice = arr_y[y:(y + delta), x:(x + delta)].flatten()

            arr_coef = np.polynomial.polynomial.polyfit(x_slice, y_slice, 1)
            arr_m[y][x] = arr_coef[1]

    return arr_m

# Compute fractal dimension on 2**num_aggre wide pixel areas
def fractal(arr_in, threshold, num_aggre):
    arr_binary = binary(arr_in, threshold)
    arr_binary = arr_binary.astype(config.work_dtype)
    removal_num = (2**num_aggre - 1)
    y_max = arr_binary.shape[0] - removal_num
    x_max = arr_binary.shape[1] - removal_num
    denom_regress = np.empty(num_aggre)
    num_regress = np.empty([num_aggre, (x_max * y_max)])
    
    for i in range(num_aggre):
        if (i > 0):
            arr_binary = aggregation.aggregate(arr_binary, Agg_ops.maximum, 1, (i - 1))

        arr_sum = aggregation.aggregate(arr_binary, Agg_ops.add_all, (num_aggre - i), i)

        arr_sum[arr_sum == 0] = 1
        arr_sum = np.log2(arr_sum)
        denom_regress[i] = -i
        num_regress[i, ] = arr_sum.flatten()

    arr_fractal_dim = np.polynomial.polynomial.polyfit(denom_regress, num_regress, 1)[1]
    arr_fractal_dim = np.reshape(arr_fractal_dim, (y_max, x_max))
    return arr_fractal_dim

def _boxed_array(arr_in, num_aggre):
    arr_min = np.amin(arr_in)
    arr_max = np.amax(arr_in)
    arr_out = np.zeros(arr_in.size, dtype=arr_in.dtype)
    if (arr_max > arr_min):
        n_boxes = 2**num_aggre
        buffer = (arr_in - arr_min) / (arr_max - arr_min)
        arr_out = np.floor(n_boxes * buffer)
    return arr_out

def fractal_3d(arr_in, num_aggre):
    if (num_aggre <= 1):
        raise ValueError('number of aggregations must be greater than one')
    arr_in = arr_in.astype(config.work_dtype)
    y_max = arr_in.shape[0] - (2**num_aggre - 1)
    x_max = arr_in.shape[1] - (2**num_aggre - 1)
    arr_box = _boxed_array(arr_in, num_aggre)
    arr_min = np.array(arr_box, dtype=arr_in.dtype)
    arr_max = np.array(arr_box, dtype=arr_in.dtype)
    denom_regress = np.empty(num_aggre)
    num_regress = np.empty([num_aggre, (x_max * y_max)])
    
    for i in range(num_aggre):
        if (i > 0):
            arr_min = aggregation.aggregate(arr_min, Agg_ops.minimum, 1, (i-1))
            arr_max = aggregation.aggregate(arr_max, Agg_ops.maximum, 1, (i-1))
            arr_min /= 2
            arr_max /= 2

        arr_sum = aggregation.aggregate((arr_max - arr_min + 1), Agg_ops.add_all, (num_aggre - i), i)

        arr_num = np.log2(arr_sum)
        denom_regress[i] = -i
        num_regress[i, ] = arr_num.flatten()

    arr_fractal_dim = np.polynomial.polynomial.polyfit(denom_regress, num_regress, 1)[1]
    arr_fractal_dim = np.reshape(arr_fractal_dim, (y_max, x_max))
    return arr_fractal_dim