import numpy as np
from numpy.polynomial import polynomial as poly
from windowagg.utilities import _Utilities
import rasterio
import inspect
import os
import math
import affine
import re

import matplotlib.pyplot as plt

class SlidingWindow:

    # TODO create more tests
    # test geoTransform update in __create_tif()
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
    # fractal 3D TODO ensure this method is written properly (check __boxed_array() for sure)
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

    __file_name = None
    __img = None
    __real_width = None
    __real_height = None

    def __init__(self, file_path, cell_width=1, cell_height=1):
        self.__file_name = os.path.split(file_path)[-1]
        self.__img = rasterio.open(file_path)
        transform = self.__img.profile['transform']
        map_width = math.sqrt(transform[0]**2 + transform[3]**2)
        map_height = math.sqrt(transform[1]**2 + transform[4]**2)
        self.__real_width = map_width*cell_width
        self.__real_height = map_height*cell_height

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, traceback):
        if (self.__img):
            self.__img.close()
    def close(self):
        if (self.__img):
            self.__img.close()
    def __del__(self):
        if (self.__img):
            self.__img.close()

    # operations for image aggregation
    __valid_ops = {'++++', '--++', '-+-+', '+--+', 'MAX', 'MIN'}
    @property
    def valid_ops(self):
        return self.__valid_ops

    # dictionary of all arrays required for DEM utils
    z, xz, yz, xxz, yyz, xyz = (np.zeros(0) for _ in range(6))
    __dem_arr_dict = {'z':z, 'xz':xz, 'yz':yz, 'xxz':xxz, 'yyz':yyz, 'xyz':xyz}
    @property
    def dem_arr_dict(self):
        return self.__dem_arr_dict

    # number of dem pixels aggregated
    __dem_pixels_aggre = 1
    @property
    def dem_pixels_aggre(self):
        return self.__dem_pixels_aggre

    # create tif with array of numpy arrays representing image bands
    # adjust geoTransform according to how many pixels were aggregated
    def __create_tif(self, arr_in, pixels_aggre=1, is_export=False, fn=None):
        if (type(arr_in) == np.ndarray):
            arr_in = [arr_in]
        dtype = arr_in[0].dtype
        shape = arr_in[0].shape
        for x in range(1, len(arr_in)):
            if (arr_in[x].dtype != dtype):
                raise ValueError('arrays must have the same dtype')
            if (arr_in[x].shape != shape):
                  raise ValueError('arrays must have the same shape')

        profile = self.__img.profile
        transform = profile['transform']
        big_tiff = 'YES'

        # update geo transform with aggregated pixels
        if (not is_export):
            big_tiff = 'NO'

            temp = np.empty(6)
            # TODO test this stuff, ok?
            pixel_width = math.sqrt(transform[0]**2 + transform[3]**2)
            pixel_height = math.sqrt(transform[1]**2 + transform[4]**2)
            temp[2] = transform[2] + (pixels_aggre-1) * pixel_width / 2
            temp[5] = transform[5] - (pixels_aggre-1) * pixel_height / 2
            temp[0] = transform[0] * pixels_aggre
            temp[1] = transform[1] * pixels_aggre
            temp[3] = transform[3] * pixels_aggre
            temp[4] = transform[4] * pixels_aggre
            transform = affine.Affine(temp[0], temp[1], temp[2], temp[3] , temp[4], temp[5])

        # TODO should nodata be 0?
        profile.update(
            nodata=0,
            transform=transform,
            dtype=dtype,
            count=len(arr_in),
            height=len(arr_in[0]),
            width=len(arr_in[0][0])
            )

        if (fn == None):
            caller_name = inspect.stack()[1].function
            fn = os.path.splitext(self.__file_name)[0] + '_' + caller_name + '.tif'
            
        with rasterio.open(fn, 'w', **profile, BIGTIFF=big_tiff) as dst:
            if (is_export):
                dst.update_tags(ns='DEM_UTILITIES', pixels_aggregated=str(self.__dem_pixels_aggre))
            for x in range(len(arr_in)): 
                dst.write(arr_in[x], x+1)

        return fn

    # create NDVI image
    def ndvi(self, red_band, ir_band):
        bands = np.array(range(self.__img.count))+1
        if (red_band not in bands or ir_band not in bands):
            raise ValueError('bands must be in range of %r.' % bands)
        
        red = self.__img.read(red_band)
        ir = self.__img.read(ir_band)
        ndvi = self.__ndvi(red, ir)
        # TODO change later
        ndvi = _Utilities._arr_dtype_conversion(ndvi, np.uint8)
        return self.__create_tif(ndvi)

    # i.e. Normalized Difference Vegetation Index
    # for viewing live green vegetation
    # requires red and infrared bands
    # returns floating point array
    def __ndvi(self, red_arr, ir_arr):
        red_arr = red_arr.astype(float)
        ir_arr = ir_arr.astype(float)
        return ( (ir_arr - red_arr) / (ir_arr + red_arr) )

    # create binary image
    def binary(self, band, threshold):
        bands = np.array(range(self.__img.count))+1
        if (band not in bands):
            raise ValueError('band must be in range of %r.' % bands)

        arr = self.__img.read(band)
        arr = self.__binary(arr, threshold)
        return self.__create_tif(arr)

    # create black and white image
    # values greater than or equal to threshold percentage will be white
    # threshold: percent in decimal of maximum
    # returns array of same data type
    # TODO can I assume minimum is always 0, how would I handle it otherwise?
    def __binary(self, arr, threshold):
        if (threshold < 0 or threshold > 1):
            raise ValueError('threshold must be between 0 and 1')
        dtype = arr.dtype
        maximum = _Utilities._get_max_min(dtype)[0]
        return np.where(arr < (threshold*maximum), 0, maximum).astype(dtype)

    # non-vectorized aggregation method
    # very slow
    # returns floating point array
    def _aggregation_brute(self, arr_in, operation, num_aggre):
        if (operation.upper() not in self.__valid_ops):
            raise ValueError('operation must be one of %r.' % self.__valid_ops)

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
                    if (operation == '++++'):
                        arr[j, i] = arr_out[j, i] + arr_out[j, i+delta] + arr_out[j+delta, i] + arr_out[j+delta, i+delta]
                    if (operation == '--++'):
                        arr[j, i] = -arr_out[j, i] - arr_out[j, i+delta] + arr_out[j+delta, i] + arr_out[j+delta, i+delta]
                    if (operation == '-+-+'):
                        arr[j, i] = -arr_out[j, i] + arr_out[j, i+delta] - arr_out[j+delta, i] + arr_out[j+delta, i+delta]
                    if (operation == '+--+'):
                        arr[j, i] = arr_out[j, i] - arr_out[j, i+delta] - arr_out[j+delta, i] + arr_out[j+delta, i+delta]
                    elif (operation.upper() == 'MAX'):
                        arr[j, i] = max(max(max(arr_out[j, i], arr_out[j, i+delta]), arr_out[j+delta, i]), arr_out[j+delta, i+delta])
                    elif (operation.upper() == 'MIN'):
                        arr[j, i] = min(min(min(arr_out[j, i], arr_out[j, i+delta]), arr_out[j+delta, i]), arr_out[j+delta, i+delta])
            arr_out = arr
        return arr_out

    # create image with each band aggregated num_aggre times
    def aggregation(self, operation, num_aggre):        
        arr = []
        for x in range(self.__img.count):
            arr.append(self.__img.read(x+1))
            arr[x] = self._partial_aggregation(arr[x], 0, num_aggre, operation)

            # TODO remove later
            arr[x] = _Utilities._arr_dtype_conversion(arr[x], np.uint8)
        
        return self.__create_tif(arr, pixels_aggre=2**num_aggre)

    # do power_target-power_start aggregations on window
    # starting with delta=2**power_start aggregation offset
    # returns floating point array
    def _partial_aggregation(self, arr_in, power_start, power_target, operation):
        if (operation.upper() not in self.__valid_ops):
            raise ValueError('operation must be one of %r.' % self.__valid_ops)
        if (power_start < 0 or power_start >= power_target):
            raise ValueError('power_start must be nonzero and less than power_target')

        arr_in = arr_in.astype(float)
        y_max = arr_in.shape[0]
        x_max = arr_in.shape[1]
        arr_out = arr_in.flatten()
        
        # iterate through sliding window sizes
        for i in range(power_start, power_target):
            delta = 2**i
            size = arr_out.size
            # create offset slices of the array to aggregate elements
            # aggregates the corners of squares of length delta+1
            top_left = arr_out[0: size - (delta*x_max + delta)]
            top_right = arr_out[delta: size - (x_max*delta)]
            bottom_left = arr_out[delta*x_max: size - (delta)]
            bottom_right = arr_out[delta*x_max + delta: size]

            if operation.upper() == '++++':
                arr_out = top_left + top_right + bottom_left + bottom_right
            if operation.upper() == '--++':
                arr_out = -top_left - top_right + bottom_left + bottom_right
            if operation.upper() == '-+-+':
                arr_out = -top_left + top_right - bottom_left + bottom_right
            if operation.upper() == '+--+':
                arr_out = top_left - top_right - bottom_left + bottom_right
            elif operation.upper() == 'MAX':
                arr_out = np.maximum(np.maximum(np.maximum(top_left, top_right), bottom_left), bottom_right)
            elif operation.upper() == 'MIN':
                arr_out = np.minimum(np.minimum(np.minimum(top_left, top_right), bottom_left), bottom_right)

        # remove last removal_num rows and columns, they are not aggregate pixels
        removal_num = (2**power_target) - (2**power_start)
        y_max -= removal_num
        # pad to make array square
        arr_out = np.pad(arr_out, (0, removal_num), 'constant')
        arr_out = np.reshape(arr_out, (y_max, x_max))
        arr_out = np.delete(arr_out, np.s_[-removal_num:], 1)
        
        return arr_out

    # create image with pixel values cooresponding to their aggregated regression slope
    def regression(self, band1, band2, num_aggre):
        bands = np.array(range(self.__img.count))+1
        if (band1 not in bands or band2 not in bands):
            raise ValueError('bands must be in range of %r.' % bands)

        arr_a = self.__img.read(band1)
        arr_b = self.__img.read(band2)
        arr_m = self._regression(arr_a, arr_b, num_aggre)

        # TODO remove later
        arr_m = _Utilities._arr_dtype_conversion(arr_m, np.uint8)

        return self.__create_tif(arr_m, pixels_aggre=2**num_aggre)

    # Do num_aggre aggregations and return the regression slope between two bands
    # returns floating point array
    def _regression(self, arr_a, arr_b, num_aggre):
        arr_a = arr_a.astype(float)
        arr_b = arr_b.astype(float)
        arr_aa = arr_a**2
        arr_ab = arr_a*arr_b

        arr_a = self._partial_aggregation(arr_a, 0, num_aggre, '++++')
        arr_b = self._partial_aggregation(arr_b, 0, num_aggre, '++++')
        arr_aa = self._partial_aggregation(arr_aa, 0, num_aggre, '++++')
        arr_ab = self._partial_aggregation(arr_ab, 0, num_aggre, '++++')

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
        bands = np.array(range(self.__img.count))+1
        if (band1 not in bands or band2 not in bands):
            raise ValueError('bands must be in range of %r.' % bands)

        arr_a = self.__img.read(band1)
        arr_b = self.__img.read(band2)
        arr_r = self._pearson(arr_a, arr_b, num_aggre)

        # TODO remove later
        arr_r = _Utilities._arr_dtype_conversion(arr_r, np.uint8)

        return self.__create_tif(arr_r, pixels_aggre=2**num_aggre)

    # Do num_aggre aggregations and return the regression slope between two bands
    # returns floating point array
    def _pearson(self, arr_a, arr_b, num_aggre):
        arr_a = arr_a.astype(float)
        arr_b = arr_b.astype(float)
        arr_aa = arr_a**2
        arr_bb = arr_b**2
        arr_ab = arr_a*arr_b

        arr_a = self._partial_aggregation(arr_a, 0, num_aggre, '++++')
        arr_b = self._partial_aggregation(arr_b, 0, num_aggre, '++++')
        arr_aa = self._partial_aggregation(arr_aa, 0, num_aggre, '++++')
        arr_bb = self._partial_aggregation(arr_bb, 0, num_aggre, '++++')
        arr_ab = self._partial_aggregation(arr_ab, 0, num_aggre, '++++')

        # total input pixels aggregated per output pixel
        count = (2**num_aggre)**2

        # pearson correlation
        numerator = count*arr_ab - arr_a*arr_b
        denominator = np.sqrt(count * arr_aa - arr_a**2) * np.sqrt(count * arr_bb - arr_b**2)
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
        y_max =  arr_a.shape[0] - (w_out-1)
        x_max = arr_a.shape[1] - (w_out-1)
        arr_m = np.empty([x_max, y_max])
        
        for j in range (y_max):
            for i in range (x_max):
                arr_1 = arr_a[j:j+w_out, i:i+w_out].flatten()
                arr_2 = arr_b[j:j+w_out, i:i+w_out].flatten()
                arr_coef = poly.polyfit(arr_1, arr_2, 1)
                arr_m[j][i] = arr_coef[1]

        return arr_m

    # create image with pixel values cooresponding to their aggregated fractal dimension
    def fractal(self, band, threshold, power_start, power_target):
        bands = np.array(range(self.__img.count))+1
        if (band not in bands):
            raise ValueError('band must be in range of %r.' % bands)

        arr = self.__img.read(band)
        arr = self._fractal(self.__binary(arr, threshold), power_start, power_target)

        # TODO remove later
        arr = _Utilities._arr_dtype_conversion(arr, np.uint16)

        return self.__create_tif(arr, pixels_aggre=2**power_target)

    # Compute fractal dimension on 2**power_target wide pixel areas
    def _fractal(self, arr_in, power_start, power_target):
        if (not _Utilities._is_binary(arr_in)):
            raise ValueError('array must be binary')
        if (power_start < 0 or power_start >= power_target):
            raise ValueError('power_start must be nonzero and less than power_target')

        arr = arr_in.astype(float)
        x_max = arr.shape[1]-(2**power_target-1)
        y_max = arr.shape[0]-(2**power_target-1)
        denom_regress = np.empty(power_target-power_start)
        num_regress = np.empty([power_target-power_start, x_max*y_max])
        
        if power_start > 0:
            arr = self._partial_aggregation(arr, 0, power_start, 'max')

        for i in range(power_start, power_target):
            arr_sum = self._partial_aggregation(arr, i, power_target, '++++')
            arr_sum = np.maximum(arr_sum, 1)

            arr_sum = np.log2(arr_sum)
            denom_regress[i-power_start] = power_target-i
            num_regress[i-power_start,] = arr_sum.flatten()
            if i < power_target-1:
                arr = self._partial_aggregation(arr, i, i+1, 'max')

        arr_slope = poly.polyfit(denom_regress, num_regress, 1)[1]
        arr_out = np.reshape(arr_slope, (y_max, x_max))
        return arr_out

    # This is for the 3D fractal dimension that is between 2 and 3, but it isn't tested yet
    def __boxed_array(self, arr_in, power_target):
        arr_min = np.amin(arr_in)
        arr_max = np.amax(arr_in)
        arr_out = np.zeros(arr_in.size)
        if (arr_max > arr_min):
            n_boxes = 2**power_target-1
            buffer = (arr_in-arr_min)/(arr_max-arr_min)
            arr_out = np.floor(n_boxes * buffer)
        return arr_out

    def fractal_3d(self, band, num_aggre):
        bands = np.array(range(self.__img.count))+1
        if (band not in bands):
            raise ValueError('band must be in range of %r.' % bands)

        arr = self.__img.read(band)
        arr = self._fractal_3d(arr, num_aggre)

        # TODO remove later
        arr = _Utilities._arr_dtype_conversion(arr, np.uint8)

        return self.__create_tif(arr, pixels_aggre=2**num_aggre)

    # TODO does this need to be binary too? probably not?
    # TODO should this have a power_start?
    def _fractal_3d(self, arr_in, num_aggre):
        if (num_aggre <= 1):
            raise ValueError('number of aggregations must be greater than one')
        y_max = arr_in.shape[0] - (2**num_aggre-1)
        x_max = arr_in.shape[1] - (2**num_aggre-1)
        arr_box = self.__boxed_array(arr_in, num_aggre).astype(float)
        arr_min = np.array(arr_box)
        arr_max = np.array(arr_box)
        denom_regress = np.empty(num_aggre-1)
        num_regress = np.empty([num_aggre-1, x_max*y_max])
        
        # TODO is this supposed to start at 1?
        for i in range(1, num_aggre):
            arr_min = self._partial_aggregation(arr_min, i-1, i, 'min')
            arr_max = self._partial_aggregation(arr_max, i-1, i, 'max')
            arr_sum = self._partial_aggregation(arr_max-arr_min+1, i, num_aggre, '++++')
            arr_num = np.log2(arr_sum)
            denom_regress[i-1] = num_aggre - i
            num_regress[i-1,] = arr_num.flatten()

            # TODO why do we divide by two?
            arr_min /= 2
            arr_max /= 2

        arr_slope = poly.polyfit(denom_regress, num_regress, 1)[1]
        arr_out = np.reshape(arr_slope, (y_max, x_max))
        return arr_out

    # TODO should I assume dem band is the only band?
    def dem_initialize_arrays(self):
        z = self.__img.read(1).astype(float)
        xz, yz, xxz, yyz, xyz = (np.zeros(z.shape).astype(z.dtype) for _ in range(5))
        self.__dem_arr_dict.update({'z':z, 'xz':xz, 'yz':yz, 'xxz':xxz, 'yyz':yyz, 'xyz':xyz})
        self.__dem_pixels_aggre = 1
    
    def dem_export_arrays(self):
        pixels_aggre = self.__dem_pixels_aggre
        export = []
        for key in self.__dem_arr_dict:
            export.append(self.__dem_arr_dict[key])
        fn = os.path.splitext(self.__file_name)[0] + '_export_w' + str(pixels_aggre) +'.tif'
        return self.__create_tif(export, pixels_aggre=pixels_aggre, is_export=True, fn=fn)

    def dem_import_arrays(self):
        if (self.__img.count != len(self.__dem_arr_dict)):
            raise ValueError('Cannot import file, %d bands are required for DEM utilities' % len(self.__dem_arr_dict))
        i = 1
        for key in self.__dem_arr_dict:
            self.__dem_arr_dict[key] = self.__img.read(i)
            i += 1
        self.__dem_pixels_aggre = int(self.__img.tags(ns='DEM_UTILITIES')['pixels_aggregated'])
        self.__file_name = re.sub('_export.*','',self.__file_name) + '.tif'

    def dem_aggregation_step(self, num_steps):
        if (self.__dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before performing aggregation steps')

        z, xz, yz, xxz, yyz, xyz = tuple (self.__dem_arr_dict[x] for x in ('z', 'xz', 'yz', 'xxz', 'yyz', 'xyz'))
        pixels_aggre = self.__dem_pixels_aggre
        delta_power = int(math.log2(pixels_aggre))

        for _ in range(num_steps):
            pixels_aggre *= 2

            z_sum_all = self._partial_aggregation(z, delta_power, delta_power+1, '++++')
            z_sum_bottom = self._partial_aggregation(z, delta_power, delta_power+1, '--++')
            z_sum_right = self._partial_aggregation(z, delta_power, delta_power+1, '-+-+')
            z_sum_main_diag = self._partial_aggregation(z, delta_power, delta_power+1, '+--+')

            xz_sum_all = self._partial_aggregation(xz, delta_power, delta_power+1, '++++')
            xz_sum_bottom = self._partial_aggregation(xz, delta_power, delta_power+1, '--++')
            xz_sum_right = self._partial_aggregation(xz, delta_power, delta_power+1, '-+-+')

            yz_sum_all = self._partial_aggregation(yz, delta_power, delta_power+1, '++++')
            yz_sum_bottom = self._partial_aggregation(yz, delta_power, delta_power+1, '--++')
            yz_sum_right = self._partial_aggregation(yz, delta_power, delta_power+1, '-+-+')

            xxz_sum_all = self._partial_aggregation(xxz, delta_power, delta_power+1, '++++')

            yyz_sum_all = self._partial_aggregation(yyz, delta_power, delta_power+1, '++++')

            xyz_sum_all = self._partial_aggregation(xyz, delta_power, delta_power+1, '++++')

            z = 0.25*z_sum_all
            xz = 0.25*(xz_sum_all + 0.25*pixels_aggre*z_sum_right)
            yz = 0.25*(yz_sum_all + 0.25*pixels_aggre*z_sum_bottom)
            xxz = 0.25*(xxz_sum_all + .5*pixels_aggre*xz_sum_right + 0.0625*(pixels_aggre**2)*z_sum_all)
            yyz = 0.25*(yyz_sum_all + .5*pixels_aggre*yz_sum_bottom + 0.0625*(pixels_aggre**2)*z_sum_all)
            xyz = 0.25*(xyz_sum_all + 0.25*pixels_aggre*(xz_sum_bottom + yz_sum_right) + 0.0625*(pixels_aggre**2)*z_sum_main_diag)

            delta_power += 1
        
        self.__dem_arr_dict.update({'z': z, 'xz': xz, 'yz': yz, 'xxz': xxz, 'yyz': yyz, 'xyz': xyz})
        self.__dem_pixels_aggre = pixels_aggre

    def _dem_aggregation_step_brute(self, num_steps):
        if (self.__dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before performing aggregation steps')

        z, xz, yz, xxz, yyz, xyz = tuple (self.__dem_arr_dict[x] for x in ('z', 'xz', 'yz', 'xxz', 'yyz', 'xyz'))
        delta = self.__dem_pixels_aggre
        
        for _ in range(num_steps):
            x_max = z.shape[1] - delta
            y_max = z.shape[0] - delta
            pixels_aggre = delta*2
            for y in range (y_max):
                for x in range (x_max):
                    z_sum_all = z[y, x] + z[y, x+delta] + z[y+delta, x] + z[y+delta, x+delta]
                    z_sum_bottom = -z[y, x] - z[y, x+delta] + z[y+delta, x] + z[y+delta, x+delta]
                    z_sum_right = -z[y, x] + z[y, x+delta] - z[y+delta, x] + z[y+delta, x+delta]
                    z_sum_main_diag = z[y,x] - z[y, x+delta] - z[y+delta, x] + z[y+delta, x+delta]

                    xz_sum_all = xz[y, x] + xz[y, x+delta] + xz[y+delta, x] + xz[y+delta, x+delta]
                    xz_sum_bottom = -xz[y, x] - xz[y, x+delta] + xz[y+delta, x] + xz[y+delta, x+delta]
                    xz_sum_right = -xz[y, x] + xz[y, x+delta] - xz[y+delta, x] + xz[y+delta, x+delta]

                    yz_sum_all = yz[y, x] + yz[y, x+delta] + yz[y+delta, x] + yz[y+delta, x+delta]
                    yz_sum_bottom = -yz[y, x] - yz[y, x+delta] + yz[y+delta, x] + yz[y+delta, x+delta]
                    yz_sum_right = -yz[y, x] + yz[y, x+delta] - yz[y+delta, x] + yz[y+delta, x+delta]

                    xxz_sum_all = xxz[y, x] + xxz[y, x+delta] + xxz[y+delta, x] + xxz[y+delta, x+delta]

                    yyz_sum_all = yyz[y, x] + yyz[y, x+delta] + yyz[y+delta, x] + yyz[y+delta, x+delta]

                    xyz_sum_all = xyz[y, x] + xyz[y, x+delta] + xyz[y+delta, x] + xyz[y+delta, x+delta]

                    xz[y, x] = 0.25*(xz_sum_all + 0.25*pixels_aggre*z_sum_right)
                    yz[y, x] = 0.25*(yz_sum_all + 0.25*pixels_aggre*z_sum_bottom)
                    xxz[y, x] = 0.25*(xxz_sum_all + .5*pixels_aggre*xz_sum_right + 0.0625*(pixels_aggre**2)*z_sum_all)
                    yyz[y, x] = 0.25*(yyz_sum_all + .5*pixels_aggre*yz_sum_bottom + 0.0625*(pixels_aggre**2)*z_sum_all)
                    xyz[y, x] = 0.25*(xyz_sum_all + 0.25*pixels_aggre*(xz_sum_bottom + yz_sum_right) + 0.0625*(pixels_aggre**2)*z_sum_main_diag)
                    z[y, x] = 0.25*z_sum_all
            delta *= 2
        
        self.__dem_arr_dict.update({'z': z, 'xz': xz, 'yz': yz, 'xxz': xxz, 'yyz': yyz, 'xyz': xyz})
        self.__dem_pixels_aggre = pixels_aggre

    # generate image of aggregated mean values of designated array
    def dem_mean(self, arr_name='z'):
        if (self.__dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before exporting mean')
        if (arr_name not in self.__dem_arr_dict):
            raise ValueError('%s must be a member of %r' % (arr_name, self.__dem_arr_dict))
        
        arr = self.__dem_arr_dict[arr_name]
        arr = _Utilities._arr_dtype_conversion(arr, np.uint16)
        pixels_aggre = self.__dem_pixels_aggre
        fn = os.path.splitext(self.__file_name)[0] + '_' + arr_name + '_mean_w' + str(pixels_aggre) + '.tif'
        return self.__create_tif(arr, pixels_aggre=pixels_aggre, fn=fn)

    # generate image of aggregated slope values
    def dem_slope(self):
        slope = self.__slope()
        slope = _Utilities._arr_dtype_conversion(slope, np.uint16)
        pixels_aggre = self.__dem_pixels_aggre
        fn = os.path.splitext(self.__file_name)[0] + '_slope_w' + str(pixels_aggre) +'.tif'
        return self.__create_tif(slope, pixels_aggre=pixels_aggre, fn=fn)

    # return array of aggregated slope values
    def __slope(self):
        if (self.__dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before calculating slope')

        w = self.__real_width
        h = self.__real_height
        pixels_aggre = self.__dem_pixels_aggre
        xz, yz = tuple (self.__dem_arr_dict[i] for i in ('xz', 'yz'))
        xx = (pixels_aggre**2-1)/12
        b0 = xz/xx
        b1 = yz/xx

        # directional derivative of the following equation
        # in the direction of the positive gradient, derived in mathematica
        # a00(x*w)**2 + 2a10(x*w)(y*h) + a11(y*h)**2 + b0(x*w) + b1(y*h) + cc
        slope = np.sqrt((b1*h)**2 + (b0*w)**2)

        return slope

        # generate image of aggregated slope values
    def dem_slope_angle(self):
        slope_angle = self.__slope_angle()
        slope_angle = _Utilities._arr_dtype_conversion(slope_angle, dtype=np.uint16, low_bound=0, up_bound=math.pi/2)
        pixels_aggre = self.__dem_pixels_aggre
        fn = os.path.splitext(self.__file_name)[0] + '_slope_angle_w' + str(pixels_aggre) +'.tif'
        return self.__create_tif(slope_angle, pixels_aggre=pixels_aggre, fn=fn)

    # return array of aggregated slope values
    def __slope_angle(self):
        if (self.__dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before calculating slope')

        w = self.__real_width
        h = self.__real_height
        pixels_aggre = self.__dem_pixels_aggre
        xz, yz = tuple (self.__dem_arr_dict[i] for i in ('xz', 'yz'))

        slope_x = xz*12/(pixels_aggre**2 - 1)
        slope_y = yz*12/(pixels_aggre**2 - 1)
        mag = np.sqrt(np.power(slope_x, 2) + np.power(slope_y, 2))
        x_unit = slope_x / mag
        y_unit = slope_y / mag
        len_opp = x_unit*slope_x + y_unit*slope_y
        len_adj = np.sqrt( ((x_unit*w)**2) + ((y_unit*h)**2) )
        slope_angle = np.arctan(len_opp/len_adj)

        return slope_angle

    # generate image of aggregated angle of steepest descent, calculated as clockwise angle from north 
    def dem_aspect(self):
        aspect = self.__aspect()
        aspect = _Utilities._arr_dtype_conversion(aspect, dtype=np.uint16, low_bound=0, up_bound=2*math.pi)
        pixels_aggre = self.__dem_pixels_aggre
        fn = os.path.splitext(self.__file_name)[0] + '_aspect_w' + str(pixels_aggre) +'.tif'
        return self.__create_tif(aspect, pixels_aggre=pixels_aggre, fn=fn)

    # return array of aggregated angle of steepest descent, calculated as clockwise angle from north
    def __aspect(self):
        if (self.__dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before calculating aspect')

        xz = self.__dem_arr_dict['xz']
        yz = self.__dem_arr_dict['yz']
        aspect = (-np.arctan(xz/yz) - np.sign(yz)*math.pi/2 + math.pi/2) % (2*math.pi)
        return aspect

    # generate image of aggregated profile curvature, second derivative parallel to steepest descent
    def dem_profile(self):
        profile = self.__profile()
        profile = _Utilities._arr_dtype_conversion(profile, np.uint16)
        pixels_aggre = self.__dem_pixels_aggre
        fn = os.path.splitext(self.__file_name)[0] + '_profile_w' + str(pixels_aggre) +'.tif'
        return self.__create_tif(profile, pixels_aggre=pixels_aggre, fn=fn)

    # return array of aggregated profile curvature, second derivative parallel to steepest descent
    def __profile(self):
        if (self.__dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before calculating profile')

        w = self.__real_width
        h = self.__real_height
        pixels_aggre = self.__dem_pixels_aggre
        z, xz, yz, yyz, xxz, xyz = tuple (self.__dem_arr_dict[i] for i in ('z', 'xz', 'yz', 'yyz', 'xxz', 'xyz'))
        xxxxminusxx2 = (pixels_aggre**4 - 5*(pixels_aggre**2) + 4)/180
        xx = (pixels_aggre**2-1)/12
        a00 = (xxz - xx*z)/xxxxminusxx2
        a10 = xyz/(2*(xx**2))
        a11 = (yyz - xx*z)/xxxxminusxx2
        b0 = xz/xx
        b1 = yz/xx

        # directional derivative of the slope of the following equation
        # in the direction of the slope, derived in mathematica
        # a00(x*w)**2 + 2a10(x*w)(y*h) + a11(y*h)**2 + b0(x*w) + b1(y*h) + cc
        profile = (2*(a11*(b1**2)*(h**4) + b0*(w**2)*(2*a10*b1*(h**2) + a00*b0*(w**2)))) / ((b1*h)**2 + (b0*w)**2)

        return profile

    # generate image of aggregated planform curvature, second derivative perpendicular to steepest descent
    def dem_planform(self):
        planform = self.__planform()
        planform = _Utilities._arr_dtype_conversion(planform, np.uint16)
        pixels_aggre = self.__dem_pixels_aggre
        fn = os.path.splitext(self.__file_name)[0] + '_planform_w' + str(pixels_aggre) +'.tif'
        return self.__create_tif(planform, pixels_aggre=pixels_aggre, fn=fn)

    # return array of aggregated planform curvature, second derivative perpendicular to steepest descent
    def __planform(self):
        if (self.__dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before calculating planform')

        w = self.__real_width
        h = self.__real_height
        pixels_aggre = self.__dem_pixels_aggre
        z, xz, yz, yyz, xxz, xyz = tuple (self.__dem_arr_dict[i] for i in ('z', 'xz', 'yz', 'yyz', 'xxz', 'xyz'))
        xxxxminusxx2 = (pixels_aggre**4 - 5*(pixels_aggre**2) + 4)/180
        xx = (pixels_aggre**2-1)/12
        a00 = (xxz - xx*z)/xxxxminusxx2
        a10 = xyz/(2*(xx**2))
        a11 = (yyz - xx*z)/xxxxminusxx2
        b0 = xz/xx
        b1 = yz/xx

        
        # directional derivative of the slope of the following equation
        # in the direction perpendicular to slope, derived in mathematica
        # a00(x*w)**2 + 2a10(x*w)(y*h) + a11(y*h)**2 + b0(x*w) + b1(y*h) + cc
        planform = (2*h*w*(a11*b0*b1*(h**2) - a10*(b1**2)*(h**2) + a10*(b0**2)*(w**2) - a00*b0*b1*(w**2))) / ((b1*h)**2 + (b0*w)**2)

        return planform

    # generate image of aggregated standard curvature
    def dem_standard(self):
        standard = self.__standard()
        standard = _Utilities._arr_dtype_conversion(standard, np.uint16)
        
        pixels_aggre = self.__dem_pixels_aggre
        fn = os.path.splitext(self.__file_name)[0] + '_standard_w' + str(pixels_aggre) +'.tif'
        return self.__create_tif(standard, pixels_aggre=pixels_aggre, fn=fn)
    
    # return array of aggregated standard curvature
    def __standard(self):
        if (self.__dem_arr_dict['z'].size == 0):
            raise ValueError('Arrays must be initialized before calculating standard curvature')
        
        w = self.__real_width
        h = self.__real_height
        pixels_aggre = self.__dem_pixels_aggre
        z, xz, yz, yyz, xxz, xyz = tuple (self.__dem_arr_dict[i] for i in ('z', 'xz', 'yz', 'yyz', 'xxz', 'xyz'))
        xxxxminusxx2 = (pixels_aggre**4 - 5*(pixels_aggre**2) + 4)/180
        xx = (pixels_aggre**2-1)/12
        a00 = (xxz - xx*z)/xxxxminusxx2
        a10 = xyz/(2*(xx**2))
        a11 = (yyz - xx*z)/xxxxminusxx2
        b0 = xz/xx
        b1 = yz/xx

        # (profile + planform) / 2
        # derived in mathematica
        standard = (a00*b0*(w**3)*(-b1*h + b0*w) + a11*b1*(h**3)*(b1*h + b0*w) + a10*h*w*((-b1**2)*(h**2) + 2*b0*b1*h*w + (b0**2)*(w**2))) / ((b1*h)**2 + (b0*w)**2)

        return standard
