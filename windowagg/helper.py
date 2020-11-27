import numpy as np
import affine
import rasterio

# get max and min of numpy data type
# returns tuple (max, min)
def dtype_max_min(dtype):
    max_val = 0
    min_val = 0
    if (np.issubdtype(dtype, np.floating)):
        max_val = np.finfo(dtype).max
        min_val = np.finfo(dtype).min
    else:
        max_val = np.iinfo(dtype).max
        min_val = np.iinfo(dtype).min

    return (max_val, min_val)

# get max and min of numpy data type
# returns tuple (max, min)
def dtype_max(dtype):
    max_val = 0
    if (np.issubdtype(dtype, np.floating)):
        max_val = np.finfo(dtype).max
    else:
        max_val = np.iinfo(dtype).max

    return max_val

# get max and min of numpy data type
# returns tuple (max, min)
def dtype_min(dtype):
    min_val = 0
    if (np.issubdtype(dtype, np.floating)):
        min_val = np.finfo(dtype).min
    else:
        min_val = np.iinfo(dtype).min

    return min_val

# create tif with array of numpy arrays representing image bands
# adjust geoTransform according to how many pixels were aggregated
def create_tif(arr_in, file_name, profile=None, num_aggre=0):
    if (type(arr_in) != list):
        arr_in = [arr_in]

    dtype = arr_in[0].dtype
    shape = arr_in[0].shape
    for i in range(1, len(arr_in)):
        if (arr_in[i].dtype != dtype):
            raise ValueError('arrays must have the same dtype')
        if (arr_in[i].shape != shape):
            raise ValueError('arrays must have the same shape')

    if (profile == None):
        transform = affine.Affine(1, 0, 1, 0, -1, 1)
        # TODO should nodata be 0?
        # TODO is the crs appropriate?
        profile = {
            'nodata': 0,
            'driver': 'GTiff',
            'crs': '+proj=latlong',
            'transform': transform,
            'dtype': dtype,
            'count': len(arr_in),
            'height': len(arr_in[0]),
            'width': len(arr_in[0][0])
        }
    else:
        transform = profile["transform"]

    num_trunc = (2**num_aggre - 1)
    img_offset = num_trunc / 2

    x = transform[2] + ((transform[0] + transform[1]) * img_offset)
    y = transform[5] + ((transform[3] + transform[4]) * img_offset)
    transform = affine.Affine(transform[0], transform[1], x, transform[3] , transform[4], y)

    big_tiff = 'NO'
    n_bytes = 0
    gigabyte = 1024**3
    for i in range(len(arr_in)):
        n_bytes += arr_in[i].nbytes
    if (n_bytes > (2 * gigabyte)):
        big_tiff = 'YES'

    # TODO should nodata be 0?
    profile.update(
        nodata=0,
        dtype=dtype,
        count=len(arr_in),
        height=len(arr_in[0]),
        width=len(arr_in[0][0])
        )
        
    with rasterio.open(file_name, 'w', **profile, BIGTIFF=big_tiff) as dst:
        for i in range(len(arr_in)): 
            dst.write(arr_in[i], i + 1)

# TODO fix later, not the best way to do this
# arr_in: array to be converted
# dtype: numpy type to convert to
def arr_dtype_conversion(arr_in, dtype=np.uint16, low_bound=None, high_bound=None):
    arr_max = np.amax(arr_in)
    arr_min = np.amin(arr_in)
    if (low_bound == None):
        low_bound = arr_min
    else:
        if (arr_min < low_bound):
            raise ValueError('Lower bound must be smaller than all values')
    if (high_bound == None):
        high_bound = arr_max
    else:
        if (arr_max > high_bound):
            raise ValueError('Upper bound must be greater than all values')

    original_max = dtype_max(arr_in.dtype)
    new_max = dtype_max(dtype)
    if (high_bound == low_bound):
        original_max = dtype_max(arr_in.dtype)
        arr_out = arr_in / original_max * new_max
    else:
        arr_out = ((arr_in - low_bound) / (high_bound - low_bound) * new_max).astype(dtype)
    
    return arr_out