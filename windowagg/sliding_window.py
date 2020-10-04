import numpy as np
from numpy.polynomial import polynomial as poly
from windowagg.utilities import _Utilities
from windowagg.agg_ops import Agg_ops
from windowagg.dem_arrays import Dem_arrays
import rasterio
import inspect
import os
import math
import affine
import re

import matplotlib.pyplot as plt

class SlidingWindow:

    # TODO create more tests
    # test geoTransform update in _create_tif()
    # test averages (e.g. xz, yz) in DEM_UTILS

    # TODO do I have all required functionality?
    # image creation with geoTransform update
    # array conversion TODO needs updating
    # ndvi
    # binary
    # aggregation
    # regression
    # Pearson
    # fractal
    # fractal 3D TODO ensure this method is written properly (check _boxed_array() for sure)
    # DEM window mean
    # DEM array intialization
    # DEM double_w TODO this has 3 methods, do we need all 3?
    # DEM slope
    # DEM aspect
    # DEM standard curve
    # DEM profile curve
    # DEM planform curve

    # TODO what is the best way to handles aggregations?
    # 1. generate all images at each step
    # 2. specify what images to generate at each step
    # 3. store the results of all aggregations and let user choose images to generate at which step

    # TODO how should RBG and DEM be differentiated?

    # TODO Is the current model for the application desirable?
    # currently: intit with image -> execute operation -> image automatically created
    # cannot currently stack bands, TODO do we want to?

    # TODO research how to create python package
    # TODO add more documentation
    # TODO should all these methods use floating point?

    # TODO should transform be updated in create tiff?

    def __init__(self, file_path, cell_width=1, cell_height=1):
        self._file_name = os.path.split(file_path)[-1]
        self._img = rasterio.open(file_path)
        transform = self._img.profile['transform']
        pixel_width = math.sqrt(transform[0]**2 + transform[3]**2)
        pixel_height = math.sqrt(transform[1]**2 + transform[4]**2)
        self._width_meters = pixel_width * cell_width
        self._height_meters = pixel_height * cell_height

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, traceback):
        if (self._img):
            self._img.close()
    def close(self):
        if (self._img):
            self._img.close()
    def __del__(self):
        if (self._img):
            self._img.close()

    # dictionary of all arrays required for DEM utils
    z, xz, yz, xxz, yyz, xyz = (np.zeros(0) for _ in range(6))
    _dem_arr_dict = {'z':z, 'xz':xz, 'yz':yz, 'xxz':xxz, 'yyz':yyz, 'xyz':xyz}
    @property
    def dem_arr_dict(self):
        return self._dem_arr_dict

    # number of pixels aggregated
    _agg_window_len = 1
    @property
    def _agg_window_len(self):
        return self._agg_window_len

    # create tif with array of numpy arrays representing image bands
    # adjust geoTransform according to how many pixels were aggregated
    def _create_tif(self, arr_in, agg_window_len=1, is_export=False, file_name=None):
        if (type(arr_in) != list):
            arr_in = [arr_in]

        dtype = arr_in[0].dtype
        shape = arr_in[0].shape
        for i in range(1, len(arr_in)):
            if (arr_in[i].dtype != dtype):
                raise ValueError('arrays must have the same dtype')
            if (arr_in[i].shape != shape):
                raise ValueError('arrays must have the same shape')

        profile = self._img.profile
        transform = profile['transform']

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

        if (file_name == None):
            caller_name = inspect.stack()[1].function
            file_name = os.path.splitext(self._file_name)[0] + '_' + caller_name + '.tif'
            
        with rasterio.open(file_name, 'w', **profile, BIGTIFF=big_tiff) as dst:
            if (is_export):
                dst.update_tags(ns='DEM_UTILITIES', aggregation_window_length=str(self._agg_window_len))
            for x in range(len(arr_in)): 
                dst.write(arr_in[x], x + 1)

        return file_name

    # create NDVI image
    def ndvi(self, red_band, ir_band):
        bands = np.array(range(self._img.count)) + 1
        if (red_band not in bands or ir_band not in bands):
            raise ValueError('bands must be in range of %r.' % bands)
        
        red = self._img.read(red_band)
        ir = self._img.read(ir_band)
        ndvi = self._ndvi(red, ir)
        # TODO change later
        ndvi = _Utilities._arr_dtype_conversion(ndvi, np.uint8)
        return self._create_tif(ndvi)

    # i.e. Normalized Difference Vegetation Index
    # for viewing live green vegetation
    # requires red and infrared bands
    # returns floating point array
    def _ndvi(self, red_arr, ir_arr):
        red_arr = red_arr.astype(float)
        ir_arr = ir_arr.astype(float)
        return ( (ir_arr - red_arr) / (ir_arr + red_arr) )

    # create binary image
    def binary(self, band, threshold):
        bands = np.array(range(self._img.count)) + 1
        if (band not in bands):
            raise ValueError('band must be in range of %r.' % bands)

        arr = self._img.read(band)
        arr = self._binary(arr, threshold)
        return self._create_tif(arr)

    # create black and white image
    # values greater than or equal to threshold percentage will be white
    # threshold: percent in decimal of maximum
    # returns array of same data type
    # TODO can I assume minimum is always 0, how would I handle it otherwise?
    def _binary(self, arr, threshold):
        if (threshold < 0 or threshold > 1):
            raise ValueError('threshold must be between 0 and 1')
        dtype = arr.dtype
        maximum = _Utilities._get_max_min(dtype)[0]
        return np.where(arr < (threshold * maximum), 0, maximum).astype(dtype)

    # create image with each band aggregated num_aggre times
    def aggregation(self, operation, num_aggre):        
        arr = []
        for x in range(self._img.count):
            arr.append(self._img.read(x + 1))
            arr[x] = self._partial_aggregation(arr[x], 0, num_aggre, operation)

            # TODO remove later
            arr[x] = _Utilities._arr_dtype_conversion(arr[x], np.uint8)
        
        return self._create_tif(arr, agg_window_len=2**num_aggre)

    # create image with pixel values cooresponding to their aggregated regression slope
    def regression(self, band1, band2, num_aggre):
        bands = np.array(range(self._img.count)) + 1
        if (band1 not in bands or band2 not in bands):
            raise ValueError('bands must be in range of %r.' % bands)

        arr_a = self._img.read(band1)
        arr_b = self._img.read(band2)
        arr_m = self._regression(arr_a, arr_b, num_aggre)

        # TODO remove later
        arr_m = _Utilities._arr_dtype_conversion(arr_m, np.uint8)

        return self._create_tif(arr_m, agg_window_len=2**num_aggre)

    # Do num_aggre aggregations and return the regression slope between two bands
    # returns floating point array
    def _regression(self, arr_a, arr_b, num_aggre):
        arr_a = arr_a.astype(float)
        arr_b = arr_b.astype(float)
        arr_aa = arr_a**2
        arr_ab = (arr_a * arr_b)

        arr_a = self._partial_aggregation(arr_a, 0, num_aggre, Agg_ops.add_all)
        arr_b = self._partial_aggregation(arr_b, 0, num_aggre, Agg_ops.add_all)
        arr_aa = self._partial_aggregation(arr_aa, 0, num_aggre, Agg_ops.add_all)
        arr_ab = self._partial_aggregation(arr_ab, 0, num_aggre, Agg_ops.add_all)

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

    # TODO potentially add R squared method?

    # create image with pixel values cooresponding to their aggregated pearson correlation
    def pearson(self, band1, band2, num_aggre):
        bands = np.array(range(self._img.count)) + 1
        if (band1 not in bands or band2 not in bands):
            raise ValueError('bands must be in range of %r.' % bands)

        arr_a = self._img.read(band1)
        arr_b = self._img.read(band2)
        arr_r = self._pearson(arr_a, arr_b, num_aggre)

        # TODO remove later
        arr_r = _Utilities._arr_dtype_conversion(arr_r, np.uint8)

        return self._create_tif(arr_r, agg_window_len=2**num_aggre)

    # Do num_aggre aggregations and return the regression slope between two bands
    # returns floating point array
    def _pearson(self, arr_a, arr_b, num_aggre):
        arr_a = arr_a.astype(float)
        arr_b = arr_b.astype(float)
        arr_aa = arr_a**2
        arr_bb = arr_b**2
        arr_ab = (arr_a * arr_b)

        arr_a = self._partial_aggregation(arr_a, 0, num_aggre, Agg_ops.add_all)
        arr_b = self._partial_aggregation(arr_b, 0, num_aggre, Agg_ops.add_all)
        arr_aa = self._partial_aggregation(arr_aa, 0, num_aggre, Agg_ops.add_all)
        arr_bb = self._partial_aggregation(arr_bb, 0, num_aggre, Agg_ops.add_all)
        arr_ab = self._partial_aggregation(arr_ab, 0, num_aggre, Agg_ops.add_all)

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
    def _regression_brute(self, arr_a, arr_b, num_aggre):
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
                arr_coef = poly.polyfit(arr_1, arr_2, 1)
                arr_m[j][i] = arr_coef[1]

        return arr_m

    # create image with pixel values cooresponding to their aggregated fractal dimension
    def fractal(self, band, threshold, power_start, power_target):
        bands = np.array(range(self._img.count)) + 1
        if (band not in bands):
            raise ValueError('band must be in range of %r.' % bands)

        arr = self._img.read(band)
        arr = self._fractal(self._binary(arr, threshold), power_start, power_target)

        # TODO remove later
        arr = _Utilities._arr_dtype_conversion(arr, np.uint16)

        return self._create_tif(arr, agg_window_len=2**power_target)

    # Compute fractal dimension on 2**power_target wide pixel areas
    def _fractal(self, arr_in, power_start, power_target):
        if (not _Utilities._is_binary(arr_in)):
            raise ValueError('array must be binary')
        if (power_start < 0 or power_start >= power_target):
            raise ValueError('power_start must be nonzero and less than power_target')

        arr = arr_in.astype(float)
        x_max = arr.shape[1] - (2**power_target - 1)
        y_max = arr.shape[0] - (2**power_target - 1)
        denom_regress = np.empty(power_target - power_start)
        num_regress = np.empty([(power_target - power_start), (x_max * y_max)])
        
        if power_start > 0:
            arr = self._partial_aggregation(arr, 0, power_start, Agg_ops.maximum)

        for i in range(power_start, power_target):
            arr_sum = self._partial_aggregation(arr, i, power_target, Agg_ops.add_all)
            arr_sum = np.maximum(arr_sum, 1)

            arr_sum = np.log2(arr_sum)
            denom_regress[i - power_start] = power_target - i
            num_regress[(i - power_start), ] = arr_sum.flatten()
            if i < (power_target - 1):
                arr = self._partial_aggregation(arr, i, (i + 1), Agg_ops.maximum)

        arr_slope = poly.polyfit(denom_regress, num_regress, 1)[1]
        arr_out = np.reshape(arr_slope, (y_max, x_max))
        return arr_out

    # This is for the 3D fractal dimension that is between 2 and 3, but it isn't tested yet
    def _boxed_array(self, arr_in, power_target):
        arr_min = np.amin(arr_in)
        arr_max = np.amax(arr_in)
        arr_out = np.zeros(arr_in.size)
        if (arr_max > arr_min):
            n_boxes = 2**power_target - 1
            buffer = (arr_in - arr_min) / (arr_max - arr_min)
            arr_out = np.floor(n_boxes * buffer)
        return arr_out

    def fractal_3d(self, band, num_aggre):
        bands = np.array(range(self._img.count)) + 1
        if (band not in bands):
            raise ValueError('band must be in range of %r.' % bands)

        arr = self._img.read(band)
        arr = self._fractal_3d(arr, num_aggre)

        # TODO remove later
        arr = _Utilities._arr_dtype_conversion(arr, np.uint8)

        return self._create_tif(arr, agg_window_len=2**num_aggre)

    # TODO does this need to be binary too? probably not?
    # TODO should this have a power_start?
    def _fractal_3d(self, arr_in, num_aggre):
        if (num_aggre <= 1):
            raise ValueError('number of aggregations must be greater than one')
        y_max = arr_in.shape[0] - (2**num_aggre - 1)
        x_max = arr_in.shape[1] - (2**num_aggre - 1)
        arr_box = self._boxed_array(arr_in, num_aggre).astype(float)
        arr_min = np.array(arr_box)
        arr_max = np.array(arr_box)
        denom_regress = np.empty(num_aggre - 1)
        num_regress = np.empty([(num_aggre - 1), (x_max * y_max)])
        
        # TODO is this supposed to start at 1?
        for i in range(1, num_aggre):
            arr_min = self._partial_aggregation(arr_min, (i - 1), i, Agg_ops.minimum)
            arr_max = self._partial_aggregation(arr_max, (i - 1), i, Agg_ops.maximum)
            arr_sum = self._partial_aggregation((arr_max - arr_min + 1), i, num_aggre, Agg_ops.add_all)
            arr_num = np.log2(arr_sum)
            denom_regress[i - 1] = num_aggre - i
            num_regress[(i - 1), ] = arr_num.flatten()

            # TODO why do we divide by two?
            arr_min /= 2
            arr_max /= 2

        arr_slope = poly.polyfit(denom_regress, num_regress, 1)[1]
        arr_out = np.reshape(arr_slope, (y_max, x_max))
        return arr_out

    # TODO should I assume dem band is the only band?
    def dem_initialize_arrays(self):
        z = self._img.read(1).astype(float)
        xz, yz, xxz, yyz, xyz = (np.zeros(z.shape).astype(z.dtype) for _ in range(5))
        self._dem_arr_dict.update({'z':z, 'xz':xz, 'yz':yz, 'xxz':xxz, 'yyz':yyz, 'xyz':xyz})
        self._agg_window_len = 1
    
    def dem_export_arrays(self):
        agg_window_len = self._agg_window_len
        export = []
        for key in self._dem_arr_dict:
            export.append(self._dem_arr_dict[key])
        file_name = os.path.splitext(self._file_name)[0] + '_export_w' + str(agg_window_len) +'.tif'
        return self._create_tif(export, agg_window_len=agg_window_len, is_export=True, file_name=file_name)

    def dem_import_arrays(self):
        if (self._img.count != len(self._dem_arr_dict)):
            raise ValueError('Cannot import file, %d bands are required for DEM utilities' % len(self._dem_arr_dict))
        i = 1
        for key in self._dem_arr_dict:
            self._dem_arr_dict[key] = self._img.read(i)
            i += 1
        self._agg_window_len = int(self._img.tags(ns='DEM_UTILITIES')['aggregation_window_length'])
        self._file_name = re.sub('_export.*','',self._file_name) + '.tif'

    # generate image of aggregated mean values of designated array
    def dem_mean(self, arr_name='z'):
        if (self._dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before exporting mean')
        if (arr_name not in self._dem_arr_dict):
            raise ValueError('%s must be a member of %r' % (arr_name, self._dem_arr_dict))
        
        arr = self._dem_arr_dict[arr_name]
        arr = _Utilities._arr_dtype_conversion(arr, np.uint16)
        agg_window_len = self._agg_window_len
        file_name = os.path.splitext(self._file_name)[0] + '_' + arr_name + '_mean_w' + str(agg_window_len) + '.tif'
        return self._create_tif(arr, agg_window_len=agg_window_len, file_name=file_name)

    # generate image of aggregated slope values
    def dem_slope(self):
        slope = self._slope()
        slope = _Utilities._arr_dtype_conversion(slope, np.uint16, low_bound=0, high_bound=np.iinfo(np.uint16).max)
        agg_window_len = self._agg_window_len
        file_name = os.path.splitext(self._file_name)[0] + '_slope_w' + str(agg_window_len) +'.tif'
        return self._create_tif(slope, agg_window_len=agg_window_len, file_name=file_name)

    # return array of aggregated slope values
    def _slope(self):
        if (self._dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before calculating slope')

        w = self._width_meters
        h = self._height_meters
        agg_window_len = self._agg_window_len
        xz, yz = tuple (self._dem_arr_dict[i] for i in ('xz', 'yz'))
        xx = (agg_window_len**2 - 1) / 12
        b0 = xz / xx
        b1 = yz / xx

        # directional derivative of the following equation
        # in the direction of the positive gradient, derived in mathematica
        # a00(x*w)**2 + 2a10(x*w)(y*h) + a11(y*h)**2 + b0(x*w) + b1(y*h) + cc
        slope = np.sqrt((b1*h)**2 + (b0*w)**2)

        return slope

        # generate image of aggregated slope values
    def dem_slope_angle(self):
        slope_angle = self._slope_angle()
        slope_angle = _Utilities._arr_dtype_conversion(slope_angle, dtype=np.uint16, low_bound=0, high_bound=math.pi/2)
        agg_window_len = self._agg_window_len
        file_name = os.path.splitext(self._file_name)[0] + '_slope_angle_w' + str(agg_window_len) +'.tif'
        return self._create_tif(slope_angle, agg_window_len=agg_window_len, file_name=file_name)

    # return array of aggregated slope values
    def _slope_angle(self):
        if (self._dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before calculating slope')

        w = self._width_meters
        h = self._height_meters
        agg_window_len = self._agg_window_len
        xz, yz = tuple (self._dem_arr_dict[i] for i in ('xz', 'yz'))

        slope_x = (xz * 12) / (agg_window_len**2 - 1)
        slope_y = (yz * 12) / (agg_window_len**2 - 1)
        mag = np.sqrt(np.power(slope_x, 2) + np.power(slope_y, 2))
        x_unit = slope_x / mag
        y_unit = slope_y / mag
        len_opp = (x_unit * slope_x) + (y_unit * slope_y)
        len_adj = np.sqrt( (x_unit * w)**2 + (y_unit * h)**2 )
        slope_angle = np.arctan(len_opp / len_adj)

        return slope_angle

    # generate image of aggregated angle of steepest descent, calculated as clockwise angle from north 
    def dem_aspect(self):
        aspect = self._aspect()
        aspect = _Utilities._arr_dtype_conversion(aspect, dtype=np.uint16, low_bound=0, high_bound=(2 * math.pi))
        agg_window_len = self._agg_window_len
        file_name = os.path.splitext(self._file_name)[0] + '_aspect_w' + str(agg_window_len) +'.tif'
        return self._create_tif(aspect, agg_window_len=agg_window_len, file_name=file_name)

    # return array of aggregated angle of steepest descent, calculated as clockwise angle from north
    def _aspect(self):
        if (self._dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before calculating aspect')

        xz = self._dem_arr_dict['xz']
        yz = self._dem_arr_dict['yz']
        aspect = (-np.arctan(xz / yz) - (np.sign(yz) * math.pi / 2) + (math.pi / 2)) % (2 * math.pi)
        return aspect

    # generate image of aggregated profile curvature, second derivative parallel to steepest descent
    def dem_profile(self):
        profile = self._profile()
        profile = _Utilities._arr_dtype_conversion(profile, np.uint16)
        agg_window_len = self._agg_window_len
        file_name = os.path.splitext(self._file_name)[0] + '_profile_w' + str(agg_window_len) +'.tif'
        return self._create_tif(profile, agg_window_len=agg_window_len, file_name=file_name)

    # return array of aggregated profile curvature, second derivative parallel to steepest descent
    def _profile(self):
        if (self._dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before calculating profile')

        w = self._width_meters
        h = self._height_meters
        agg_window_len = self._agg_window_len
        z, xz, yz, yyz, xxz, xyz = tuple (self._dem_arr_dict[i] for i in ('z', 'xz', 'yz', 'yyz', 'xxz', 'xyz'))
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

    # generate image of aggregated planform curvature, second derivative perpendicular to steepest descent
    def dem_planform(self):
        planform = self._planform()
        planform = _Utilities._arr_dtype_conversion(planform, np.uint16)
        agg_window_len = self._agg_window_len
        file_name = os.path.splitext(self._file_name)[0] + '_planform_w' + str(agg_window_len) +'.tif'
        return self._create_tif(planform, agg_window_len=agg_window_len, file_name=file_name)

    # return array of aggregated planform curvature, second derivative perpendicular to steepest descent
    def _planform(self):
        if (self._dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before calculating planform')

        w = self._width_meters
        h = self._height_meters
        agg_window_len = self._agg_window_len
        z, xz, yz, yyz, xxz, xyz = tuple (self._dem_arr_dict[i] for i in ('z', 'xz', 'yz', 'yyz', 'xxz', 'xyz'))
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

    # generate image of aggregated standard curvature
    def dem_standard(self):
        standard = self._standard()
        standard = _Utilities._arr_dtype_conversion(standard, np.uint16)
        
        agg_window_len = self._agg_window_len
        file_name = os.path.splitext(self._file_name)[0] + '_standard_w' + str(agg_window_len) +'.tif'
        return self._create_tif(standard, agg_window_len=agg_window_len, file_name=file_name)
    
    # return array of aggregated standard curvature
    def _standard(self):
        if (self._dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before calculating standard curvature')
        
        w = self._width_meters
        h = self._height_meters
        agg_window_len = self._agg_window_len
        z, xz, yz, yyz, xxz, xyz = tuple (self._dem_arr_dict[i] for i in ('z', 'xz', 'yz', 'yyz', 'xxz', 'xyz'))
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