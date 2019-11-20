import numpy as np
import affine
import rasterio
import math

class _Utilities:

    # TODO fix later, not the best way to do this
    # arr_in: array to be converted
    # dtype: numpy type to convert to
    @staticmethod
    def _arr_dtype_conversion(arr_in, dtype):
        arr_max = np.amax(arr_in)
        arr_min = np.amin(arr_in)
        dtype_max = _Utilities._get_max_min(dtype)[0]
        arr_out = ((arr_in - arr_min)/(arr_max - arr_min)*dtype_max).astype(dtype) 
        return arr_out

    # get max and min of numpy data type
    # returns tuple (max, min)
    @staticmethod
    def _get_max_min(dtype):
        max_val = 0
        min_val = 0
        if (np.issubdtype(dtype, np.floating)):
            max_val = np.finfo(dtype).max
            min_val = np.finfo(dtype).min
        else:
            max_val = np.iinfo(dtype).max
            min_val = np.iinfo(dtype).min

        return (max_val, min_val)

    # check if an image is black and white or not
    # i.e. only contains values of dtype.min and dtype.max
    # TODO should min value be arr_in.dtype.min or 0?
    @staticmethod
    def _is_binary(arr_in):
        max_val = np.amax(arr_in)
        return ((arr_in==0) | (arr_in==max_val)).all()

    @staticmethod
    def _create_new_tif(arr_in, fn, angle=0, x_offset=1, y_offset=1):
        if (type(arr_in) == np.ndarray):
            arr_in = [arr_in]
        dtype = arr_in[0].dtype
        shape = arr_in[0].shape
        for x in range(1, len(arr_in)):
            if (arr_in[x].dtype != dtype):
                raise ValueError('arrays must have the same dtype')
            if (arr_in[x].shape != shape):
                raise ValueError('arrays must have the same shape')

        rotate = math.pi/2
        transform = affine.Affine(math.cos(angle)*x_offset, -math.cos(angle+rotate)*y_offset, 1, math.sin(angle)*x_offset, -math.sin(angle+rotate)*y_offset, 1)

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
            
        with rasterio.open(fn, 'w', **profile) as dst:
            for x in range(len(arr_in)): 
                dst.write(arr_in[x], x+1)