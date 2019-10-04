import numpy as np
from numpy.polynomial import polynomial as poly
import rasterio
import inspect
import os
import math
import affine

class SlidingWindow:

    def __init__(self, file_path, band_enum):
        self.file_name = os.path.split(file_path)[-1]
        self.img = rasterio.open(file_path)
        self.band_enum = band_enum

    __valid_ops = {'++++', '++--', '-+-+', '-++-', 'MAX', 'MIN'}
    @property
    def valid_ops(self):
        return self.__valid_ops

    # i.e. Normalized Difference Vegetation Index
    # for viewing live green vegetation
    def ndvi(self):
        red = self.img.read(self.band_enum.red.value)
        ir = self.img.read(self.band_enum.ir.value)
        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        ndvi = ((ir.astype(float) - red.astype(float)) / (ir + red)).astype(np.uint8)

        self.__create_tif(1, [ndvi])
        
    # turn image into black and white
    # values greater than or equal to threshold are white
    def binary(self, band, threshold):
        arr = self.img.read(self.band_enum[band].value)
        arr = self.__binary(arr, threshold)
        self.__create_tif(1, [arr])

    # TODO specify threshold
    def __binary(self, arr, threshold):
        return np.where(arr < threshold, 0, 255).astype(np.uint8)

    def __is_binary(self, arr_in):
        dtype = arr_in.dtype
        max_val = 0
        min_val = 0
        (max_val, min_val) = self.__get_max_min(dtype)
        return ((arr_in==min_val) | (arr_in==max_val)).all()

    # get max and min of numpy data type
    # returns tuple (max, min)
    def __get_max_min(self, dtype):
        max_val = 0
        min_val = 0
        if (np.issubdtype(dtype, np.floating)):
            max_val = np.finfo(dtype).max
            min_val = np.finfo(dtype).min
        else:
            max_val = np.iinfo(dtype).max
            min_val = np.iinfo(dtype).min

        return (max_val, min_val)

    # create tif with array of image bands
    def __create_tif(self, num_bands, arr_in, pixels_aggre=1, fn=None, dtype='uint8'):
        profile = self.img.profile

        # update transform with aggregated pixels
        transform = profile['transform']
        temp = np.empty(6)

        pixel_width = math.sqrt(transform[0]**2 + transform[3]**2)
        pixel_height = math.sqrt(transform[1]**2 + transform[4]**2)
        temp[2] = transform[2] + (pixels_aggre-1) * pixel_width / 2
        temp[5] = transform[5] - (pixels_aggre-1) * pixel_height / 2

        temp[0] = transform[0] * pixels_aggre
        temp[1] = transform[1] * pixels_aggre
        temp[3] = transform[3] * pixels_aggre
        temp[4] = transform[4] * pixels_aggre

        new_transform = affine.Affine(temp[0], temp[1], temp[2], temp[3] , temp[4], temp[5])

        profile.update(
            nodata=0,
            transform=new_transform,
            dtype=dtype,
            count=num_bands,
            height=len(arr_in[0]),
            width=len(arr_in[0][0])
            )

        if (fn == None):
            caller_name = inspect.stack()[1].function
            fn = os.path.splitext(self.file_name)[0] + '_' + caller_name + '.tif'
            
        with rasterio.open(fn, 'w', **profile) as dst:
            for x in range(num_bands): 
                dst.write(arr_in[x], x+1)

    # TODO fix later, not the best way to do this
    # arr_in: array to be converted
    # dtype: numpy type to convert to
    def __arr_dtype_conversion(self, arr_in, dtype):
        arr_max = np.amax(arr_in)
        arr_min = np.amin(arr_in)
        dtype_max = self.__get_max_min(dtype)[0]
        arr_out = ((arr_in - arr_min)/(arr_max - arr_min) * dtype_max).astype(dtype)
        return arr_out

    def _aggregation_brute(self, arr_in, operation, num_aggre):
        if (operation.upper() not in self.__valid_ops):
            raise ValueError('operation must be one of %r.' % self.__valid_ops)

        y_max = arr_in.shape[0]
        x_max = arr_in.shape[1]
        arr_out = np.array(arr_in)
        for i in range(num_aggre):
            delta = 2**i
            y_max -= delta
            x_max -= delta
            arr = np.empty([y_max, x_max])

            for j in range (y_max):
                for i in range (x_max):
                    if (operation.upper() == '++++'):
                        arr[j, i] = arr_out[j, i] + arr_out[j, i+delta] + arr_out[j+delta, i] + arr_out[j+delta, i+delta]
                    if (operation.upper() == '++--'):
                        arr[j, i] = arr_out[j, i] + arr_out[j, i+delta] - arr_out[j+delta, i] - arr_out[j+delta, i+delta]
                    if (operation.upper() == '-+-+'):
                        arr[j, i] = -arr_out[j, i] + arr_out[j, i+delta] - arr_out[j+delta, i] + arr_out[j+delta, i+delta]
                    if (operation.upper() == '-++-'):
                        arr[j, i] = -arr_out[j, i] + arr_out[j, i+delta] + arr_out[j+delta, i] - arr_out[j+delta, i+delta]
                    elif (operation.upper() == 'MAX'):
                        arr[j, i] = max(max(max(arr_out[j, i], arr_out[j, i+delta]), arr_out[j+delta, i]), arr_out[j+delta, i+delta])
                    elif (operation.upper() == 'MIN'):
                        arr[j, i] = min(min(min(arr_out[j, i], arr_out[j, i+delta]), arr_out[j+delta, i]), arr_out[j+delta, i+delta])
            arr_out = arr

        return arr_out

    # convert band into array, then call actual aggregation function
    def aggregation(self, operation, num_aggre):
        if (operation.upper() not in self.__valid_ops):
            raise ValueError('operation must be one of %r.' % self.__valid_ops)
        
        arr = []
        num_bands = len(self.band_enum)
        for x in range(num_bands):
            arr.append(self.img.read(x+1).astype(float))
            arr[x] = self._partial_aggregation(arr[x], 0, num_aggre, operation)

            # TODO remove later
            arr[x] = self.__arr_dtype_conversion(arr[x], np.uint8)
        
        self.__create_tif(num_bands, arr, 2**num_aggre)

    # do power_target-power_start aggregations on window
    # starting with delta=2**power_start
    def _partial_aggregation(self, arr_in, power_start, power_target, operation):
        if (operation.upper() not in self.__valid_ops):
            raise ValueError('operation must be one of %r.' % self.__valid_ops)
        if (power_start < 0 or power_start >= power_target):
            raise ValueError('power_start must be nonzero and less than power_target')

        y_max = arr_in.shape[0]
        x_max = arr_in.shape[1]
        arr_out = arr_in.flatten()
        
        # calculate sliding window for each value of delta
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
            # TODO should this be sum top?
            if operation.upper() == '++--':
                arr_out = top_left + top_right - bottom_left - bottom_right
            if operation.upper() == '-+-+':
                arr_out = -top_left + top_right - bottom_left + bottom_right
            if operation.upper() == '-++-':
                arr_out = -top_left + top_right + bottom_left - bottom_right
            elif operation.upper() == 'MAX':
                arr_out = np.maximum(np.maximum(np.maximum(top_left, top_right), bottom_left), bottom_right)
            elif operation.upper() == 'MIN':
                arr_out = np.minimum(np.minimum(np.minimum(top_left, top_right), bottom_left), bottom_right)

        # remove last removal_num rows and columns
        removal_num = (2**power_target) - (2**power_start)
        y_max -= removal_num
        # pad to make array square
        arr_out = np.pad(arr_out, (0, removal_num), 'constant')
        arr_out = np.reshape(arr_out, (y_max, x_max))
        arr_out = np.delete(arr_out, np.s_[-removal_num:], 1)
        
        return arr_out

    def regression(self, band1, band2, num_aggre):
        arr_a = self.img.read(self.band_enum[band1].value).astype(float)
        arr_b = self.img.read(self.band_enum[band2].value).astype(float)
        arr_m = self._regression(arr_a, arr_b, num_aggre)

        # TODO remove later
        arr_m = self.__arr_dtype_conversion(arr_m, np.uint8)

        self.__create_tif(1, [arr_m], 2**num_aggre)

    # Do num_aggre aggregations and return the regression slope between two bands
    def _regression(self, arr_a, arr_b, num_aggre):
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
        denominator = count * arr_aa - arr_a * arr_a
        # avoid division by zero
        denominator = np.maximum(denominator, 1)
        arr_m = numerator/denominator

        return arr_m

    # TODO potentially add R squared method?

    # Do num_aggre aggregations and return the pearson coorelation between two bands
    def pearson(self, band1, band2, num_aggre):
        arr_a = self.img.read(self.band_enum[band1].value).astype(float)
        arr_b = self.img.read(self.band_enum[band2].value).astype(float)
        arr_r = self._pearson(arr_a, arr_b, num_aggre)

        # TODO remove later
        arr_r = self.__arr_dtype_conversion(arr_r, np.uint8)

        self.__create_tif(1, [arr_r], 2**num_aggre)

    # Do num_aggre aggregations and return the regression slope between two bands
    def _pearson(self, arr_a, arr_b, num_aggre):
        arr_aa = arr_a**2
        arr_bb = arr_b**2
        arr_ab = arr_a*arr_b

        arr_a = self._partial_aggregation(arr_a, 0, num_aggre, '++++')
        arr_b = self._partial_aggregation(arr_b, 0, num_aggre, '++++')
        arr_aa = self._partial_aggregation(arr_aa, 0, num_aggre, '++++')
        arr_bb = self._partial_aggregation(arr_bb, 0, num_aggre, '++++')
        arr_ab = self._partial_aggregation(arr_ab, 0, num_aggre, '++++')

        # total pixels aggregated per pixel
        count = (2**num_aggre)**2

        # pearson correlation
        numerator = count*arr_ab - arr_a*arr_b
        denominator = np.sqrt(count * arr_aa - arr_a**2) * np.sqrt(count * arr_bb - arr_b**2)
        # avoid division by zero
        denominator = np.maximum(denominator, 1)
        arr_r = numerator / denominator
        
        return arr_r

    def _regression_brute(self, arr_a, arr_b, num_aggre):
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

    # TODO specify binary image
    def fractal(self, band, power_start, power_target):
        if (power_start < 0 or power_start >= power_target):
            raise ValueError('power_start must be nonzero and less than power_target')

        arr = self.img.read(self.band_enum[band].value).astype(float)
        arr = self._fractal(self.__binary(arr, 127), power_start, power_target)

        # TODO remove later
        arr = self.__arr_dtype_conversion(arr, np.uint8)

        self.__create_tif(1, [arr], 2**power_target)

    def _fractal(self, arr_in, power_start, power_target):
        if (not self.__is_binary(arr_in)):
            raise ValueError('array must be binary')
        if (power_start < 0 or power_start >= power_target):
            raise ValueError('power_start must be nonzero and less than power_target')

        y_max = arr_in.shape[0]-(2**power_target-1)
        x_max = arr_in.shape[1]-(2**power_target-1)
        arr = np.array(arr_in)
        denom_regress = np.empty(power_target-power_start)
        num_regress = np.empty([power_target-power_start, x_max*y_max])
        
        if power_start > 0:
            arr = self._partial_aggregation(arr, 0, power_start, 'max')

        for i in range(power_start, power_target):
            arr_sum = self._partial_aggregation(arr, i, power_target, '++++')
            arr_sum = np.maximum(arr_sum, 1)

            arr_sum = np.log(arr_sum)/np.log(2)
            denom_regress[i-power_start] = power_target-i
            num_regress[i-power_start,] = arr_sum.flatten()
            if i < power_target-1:
                arr = self._partial_aggregation(arr, i, i+1, 'max')

        arr_coef = poly.polyfit(denom_regress, num_regress, 1)
        arr_out = np.reshape(arr_coef[1], (y_max, x_max))
        return arr_out

    # This is for the 3D fractal dimension that is between 2 and 3, but it isn't tested yet
    def __boxed_array(self, arr_in, power_target):
        arr_min = np.min(arr_in)
        arr_max = np.max(arr_in)
        arr_out = np.zeros(arr_in.size)
        if (arr_max > arr_min):
            n_boxes = 2**power_target-1
            buffer = (arr_in-arr_min)/(arr_max-arr_min)
            arr_out = np.floor(n_boxes * buffer)
        return arr_out

    def fractal_3d(self, band, num_aggre):
        if (num_aggre <= 0):
            raise ValueError('number of aggregations must be greater than zero')

        arr = self.img.read(self.band_enum[band].value).astype(float)
        arr = self._fractal_3d(arr, num_aggre)

        # TODO remove later
        arr = self.__arr_dtype_conversion(arr, np.uint8)

        self.__create_tif(1, [arr], 2**num_aggre)

    # TODO does this need to be binary too?
    # TODO should this have a power_start?
    def _fractal_3d(self, arr_in, num_aggre):
        if (num_aggre <= 0):
            raise ValueError('number of aggregations must be greater than zero')
        y_max = arr_in.shape[0] - (2**num_aggre-1)
        x_max = arr_in.shape[1] - (2**num_aggre-1)
        arr_box = self.__boxed_array(arr_in, num_aggre)
        arr_min = np.array(arr_box)
        arr_max = np.array(arr_box)
        # TODO is this the correct linear regression? one x value per aggregation step?
        denom_regress = np.empty(num_aggre-1)
        num_regress = np.empty([num_aggre-1, x_max*y_max])

        # TODO is this supposed to start at 1?
        for i in range(1, num_aggre):
            arr_min = self._partial_aggregation(arr_min, i-1, i, 'min')
            arr_max = self._partial_aggregation(arr_max, i-1, i, 'max')
            arr_sum = self._partial_aggregation(arr_max-arr_min+1, i, num_aggre, '++++')
            arr_num = np.log(arr_sum)/np.log(2)
            denom_regress[i-1] = num_aggre - i
            num_regress[i-1,] = arr_num.flatten()

            arr_min /= 2
            arr_max /= 2

        arr_coef = poly.polyfit(denom_regress, num_regress, 1)
        arr_out = np.reshape(arr_coef[1], (y_max, x_max))
        return arr_out

    def _fractal_brute(self, arr_in, power_start, power_target):
        if (power_start < 0 or power_start >= power_target):
            raise ValueError('power_start must be nonzero and less than power_target')

        y_max = arr_in.shape[0] - (2**power_target - 1)
        x_max = arr_in.shape[1] - (2**power_target - 1)
        arr = self.__binary(arr_in, np.mean(arr_in))
        denom_regress = np.empty(power_target-power_start)
        num_regress = np.zeros([power_target-power_start, x_max*y_max])
        
        if power_start > 0:
            arr = self._partial_aggregation(arr, 0, power_start, 'max')
        
        for i in range(power_start, power_target):
            arr_sum = self._partial_aggregation(arr, i, power_target, '++++')
            arr_sum = np.maximum(arr_sum, 1)
            num_array = np.log(arr_sum)/np.log(2)
            denom = power_target-i
            denom_regress[i-power_start] = denom
            num_regress[i-power_start,] = num_array.flatten()
            if i < power_target-1:
                arr = self._partial_aggregation(arr, i, i+1, 'max')
        
        arr_coef = poly.polyfit(denom_regress, num_regress, 1)
        arr_out = np.reshape(arr_coef[1], (y_max, x_max))
        return arr_out

    # TODO should i assume that this is the first band?
    def dem_utils(self, num_aggre):
        arr = self.img.read(1).astype(float)
        arr_dic = self._initialize_arrays(arr)

        for i in range(num_aggre):
            self._double_w(i, arr_dic)
            self.__window_mean(i, arr_dic['z'])

    def __window_mean(self, delta_power, arr_in):
        delta = 2**delta_power
        arr = self.__arr_dtype_conversion(arr_in, np.uint16)
        fn = os.path.splitext(self.file_name)[0] + '_mean_w' + str(delta*2) + '.tif'
        self.__create_tif(1, [arr], delta*2, fn, 'uint16')

    def _initialize_arrays(self, z):
        # TODO will there be a problem with x and y when they have odd sizes?
        # TODO these might actually need to be zero?
        x_max = z.shape[1]
        y_max = z.shape[0]
        x_range = np.arange(x_max)
        x_range = x_range - np.mean(x_range)
        y_range = np.arange(y_max)[::-1]
        y_range = y_range - np.mean(y_range)

        x = np.tile(x_range, (y_max, 1))
        y = np.transpose(np.tile(y_range, (x_max, 1)))
        yz = y*z
        xz = x*z
        yyz = y*y*z
        xxz = x*x*z
        xyz = x*y*z

        arr_dic = {'z':z, 'xz':xz, 'yz':yz, 'xxz':xxz, 'yyz':yyz, 'xyz':xyz, 'orig_width': z.shape[1], 'orig_height': z.shape[0]}
        return arr_dic

    def _double_w(self, delta_power, arr_dic):
        delta = 2**delta_power

        z_sum_all = self._partial_aggregation(arr_dic['z'], delta_power, delta_power+1, '++++')
        z_sum_top = self._partial_aggregation(arr_dic['z'], delta_power, delta_power+1, '++--')
        z_sum_right = self._partial_aggregation(arr_dic['z'], delta_power, delta_power+1, '-+-+')
        z_sum_anti_diag = self._partial_aggregation(arr_dic['z'], delta_power, delta_power+1, '-++-')

        xz_sum_all = self._partial_aggregation(arr_dic['xz'], delta_power, delta_power+1, '++++')
        xz_sum_top = self._partial_aggregation(arr_dic['xz'], delta_power, delta_power+1, '++--')
        xz_sum_right = self._partial_aggregation(arr_dic['xz'], delta_power, delta_power+1, '-+-+')

        yz_sum_all = self._partial_aggregation(arr_dic['yz'], delta_power, delta_power+1, '++++')
        yz_sum_top = self._partial_aggregation(arr_dic['yz'], delta_power, delta_power+1, '++--')
        yz_sum_right = self._partial_aggregation(arr_dic['yz'], delta_power, delta_power+1, '-+-+')

        xxz_sum_all = self._partial_aggregation(arr_dic['xxz'], delta_power, delta_power+1, '++++')

        yyz_sum_all = self._partial_aggregation(arr_dic['yyz'], delta_power, delta_power+1, '++++')

        xyz_sum_all = self._partial_aggregation(arr_dic['xyz'], delta_power, delta_power+1, '++++')

        xxz = 0.25*(xxz_sum_all + delta*xz_sum_right + 0.25*(delta**2)*z_sum_all)
        yyz = 0.25*(yyz_sum_all + yz_sum_top*delta + 0.25*(delta**2)*z_sum_all)
        xyz = 0.25*(xyz_sum_all + 0.5*delta*(xz_sum_top + yz_sum_right) + 0.25*(delta**2)*z_sum_anti_diag)
        xz = 0.25*(xz_sum_all + 0.5*delta*z_sum_right)
        yz = 0.25*(yz_sum_all + 0.5*delta*z_sum_top)
        z = 0.25*z_sum_all
        
        for i in (['z', z], ['xz', xz], ['yz', yz], ['xxz', xxz], ['yyz', yyz], ['xyz', xyz]):
            arr_dic[i[0]] = i[1]

    def _double_w_brute(self, delta_power, arr_dic):
        delta = 2**delta_power
        z, xz, yz, xxz, yyz, xyz = (arr_dic[x] for x in ('z', 'xz', 'yz', 'xxz', 'yyz', 'xyz'))
        x_max = z.shape[1] - delta
        y_max = z.shape[0] - delta
        z_loc, xz_loc, yz_loc, xxz_loc, yyz_loc, xyz_loc = (np.zeros([y_max, x_max]) for _ in range(6))

        for y in range (y_max):
            for x in range (x_max):
                xz_loc[y, x] = (
                    (xz[y, x] + xz[y, x+delta] + xz[y+delta, x] + xz[y+delta, x+delta]) + 
                    (-z[y, x] + z[y, x+delta] - z[y+delta, x] + z[y+delta, x+delta])*0.5*delta
                )*0.25

                yz_loc[y, x] = (
                    (yz[y, x] + yz[y, x+delta] + yz[y+delta, x] + yz[y+delta, x+delta]) + 
                    (z[y, x] + z[y, x+delta] - z[y+delta, x] - z[y+delta, x+delta])*0.5*delta
                )*0.25

                xxz_loc[y, x] = (
                    (xxz[y, x] + xxz[y, x+delta] + xxz[y+delta, x] + xxz[y+delta, x+delta]) + 
                    (-xz[y, x] + xz[y, x+delta] - xz[y+delta, x] + xz[y+delta, x+delta])*delta + 
                    (z[y, x] + z[y, x+delta] + z[y+delta, x] + z[y+delta, x+delta])*0.25*(delta**2)
                )*0.25

                yyz_loc[y, x] = (
                    (yyz[y, x] + yyz[y, x+delta] + yyz[y+delta, x] + yyz[y+delta, x+delta]) + 
                    (yz[y, x] + yz[y, x+delta] - yz[y+delta, x] - yz[y+delta, x+delta])*delta + 
                    (z[y, x] + z[y, x+delta] + z[y+delta, x] + z[y+delta, x+delta])*0.25*(delta**2)
                )*0.25

                xyz_loc[y, x] = (
                    (xyz[y, x] + xyz[y, x+delta] + xyz[y+delta, x] + xyz[y+delta, x+delta]) + (
                        (xz[y, x] + xz[y, x+delta] - xz[y+delta, x] - xz[y+delta, x+delta]) +
                        (-yz[y, x] + yz[y, x+delta] - yz[y+delta, x] + yz[y+delta, x+delta])
                    )*0.5*delta + 
                    (-z[y,x] + z[y, x+delta] + z[y+delta, x] - z[y+delta, x+delta])*0.25*(delta**2)
                )*0.25

                z_loc[y, x] = (z[y, x] + z[y, x+delta] + z[y+delta, x] + z[y+delta, x+delta])*0.25
        
        for i in (['z', z_loc], ['xz', xz_loc], ['yz', yz_loc], ['xxz', xxz_loc], ['yyz', yyz_loc], ['xyz', xyz_loc]):
            arr_dic[i[0]] = i[1]

    def _double_w_old(self, delta_power, arr_dic):
        # arrays represent corners of aggregation square: [top_left, top_right, bottom_left, bottom_right]
        sum_right = np.array([-1, 1, -1, 1])[:,None]
        sum_top = np.array([1, 1, -1, -1])[:,None]
        sum_anti_diag = np.array([-1, 1, 1, -1])[:,None]

        delta = 2**delta_power
        z_in = arr_dic['z']
        x_max_old = z_in.shape[1]
        x_max = z_in.shape[1] - delta
        y_max = z_in.shape[0] - delta
        # sum the 4 corners of a square of width delta
        # separation of indices to sum
        corner_indices = np.array([[0], [delta], [delta*x_max_old], [delta*x_max_old+delta]])
        # create array of indices to sum: [[top_left], [top_right], [bottom_left], [bottom_right]]
        top_left_indices = (np.arange(y_max)[:, np.newaxis]*x_max_old + np.arange(x_max)).flatten()
        full_selector = corner_indices + top_left_indices

        z_corners = np.take(z_in, full_selector)
        z_sum_all = z_corners.mean(axis=0)
        z_sum_top = (z_corners*sum_top).mean(axis=0)
        z_sum_right = (z_corners*sum_right).mean(axis=0)
        z_sum_anti_diag = (z_corners*sum_anti_diag).mean(axis=0)

        xz_corners = np.take(arr_dic['xz'], full_selector)
        xz_sum_all = xz_corners.mean(axis=0)
        xz_sum_top = (xz_corners*sum_top).mean(axis=0)
        xz_sum_right = (xz_corners*sum_right).mean(axis=0)

        yz_corners = np.take(arr_dic['yz'], full_selector)
        yz_sum_all = yz_corners.mean(axis=0)
        yz_sum_top = (yz_corners*sum_top).mean(axis=0)
        yz_sum_right = (yz_corners*sum_right).mean(axis=0)

        xxz_corners = np.take(arr_dic['xxz'], full_selector)
        xxz_sum_all = xxz_corners.mean(axis=0)

        yyz_corners = np.take(arr_dic['yyz'], full_selector)
        yyz_sum_all = yyz_corners.mean(axis=0)

        xyz_corners = np.take(arr_dic['xyz'], full_selector)
        xyz_sum_all = xyz_corners.mean(axis=0)

        z = z_sum_all
        xz = xz_sum_all  + 0.5*delta*z_sum_right
        yz = yz_sum_all + 0.5*delta*z_sum_top
        xxz = xxz_sum_all + xz_sum_right*delta + 0.25*(delta**2)*z
        yyz = yyz_sum_all + yz_sum_top*delta + 0.25*(delta**2)*z
        xyz = xyz_sum_all + .5*delta*(xz_sum_top + yz_sum_right) + 0.25*(delta**2)*z_sum_anti_diag
    
        for i in (['z', z], ['xz', xz], ['yz', yz], ['xxz', xxz], ['yyz', yyz], ['xyz', xyz]):
            arr_dic[i[0]] = i[1].reshape((y_max, x_max))

    # TODO NOT FUNCTIONAL
    def slope(self, arr_dic, delta_power, prefactor):
        delta = 2**delta_power
        z = arr_dic['z']
        yz = arr_dic['yz']
        xz = arr_dic['xz']
        y_max = arr_dic['orig_height'] - (delta-1)
        x_max = arr_dic['orig_width'] - (delta-1)
        #slope_image = Image.new('F',(width-w+1,height-w+1))
        slope_array = np.zeros([y_max, x_max])
        xx = ((delta**2) - 1) / 12.0

        slope = np.arctan(np.sqrt((xz**2) + (yz**2)) / xx)

        for j in range (y_max):
            for i in range (x_max):
                # TODO why do we need transform and prefactor
                value = math.atan(prefactor / xx * math.sqrt(xz[j,i]**2 + yz[j,i]**2)/abs(self.img.profile['transform'][0]))
                #slope_image.putpixel((i,j),value)
                slope_array[j,i]=value

        #im = Image.fromarray(slope_array)
        #slope_image.save(fn_loc)
        slope_min = np.min(slope_array)
        slope_max = np.max(slope_array)
        slope_array = ((slope_array - slope_min) / (slope_max - slope_min) * np.iinfo(np.uint16).max).astype(np.uint16)
        fn = os.path.splitext(self.file_name)[0] + '_slope_w' + str(delta*2) +'.tif'
        self.__create_tif(1, [slope_array], delta*2, fn, 'uint16')
        #print_tfw(fn,fn_loc,w,GeoT)
        print_display_tfw(fn,fn_loc,w,GeoT,3)

    # TODO NOT FUNCTIONAL
    # Sign of pixel directions considered here and in directional curvaturen terms
    def aspect(w,z,xz,yz,fn,GeoT):
        #aspect_image = Image.new('F',(width-w+1,height-w+1))
        aspect_array = np.zeros((height-w+1,width-w+1))
        factor = maxuint16 / (2*math.pi)
        for j in range (0, height-w+1):
            for i in range (0, width-w+1):
                xz_loc = xz[j,i]
                yz_loc = yz[j,i]
                if yz_loc == 0:
                    if xz_loc < 0:
                        value = 1.5*math.pi
                    else:
                        value = 0.5*math.pi
                else:
                    value = -math.atan(xz_loc/yz_loc)
                    if yz_loc < 0:
                        value += math.pi
                    elif xz_loc > 0:
                        value += 2*math.pi
                #aspect_image.putpixel((i,j),value)
                aspect_array[j,i]=value*factor
        #im = Image.fromarray(aspect_array)
        #im.save(fn_loc)
        print("aspect_array")
        print(aspect_array)
        fn_loc = os.path.splitext(fn)[0] +'_aspect_w'+str(w)+'.tif'
        n = aspect_array.astype(np.uint16)
        tiff.imsave(fn_loc,n)
        #print_tfw(fn,fn_loc,w,GeoT)
        print_display_tfw(fn,fn_loc,w,GeoT,1)

    # TODO NOT FUNCTIONAL
    def norm_curv(w,z,xz,yz,xxz,yyz,fn,GeoT):
        curv_array = np.empty((height-w+1,width-w+1))
        
        xx = (w*w - 1) / 12.
        inv_x4mxx2 = 180. / (w*w*w*w - 5*w*w + 4)
        curv_array = (0.5*(xxz + yyz) - np.multiply(xx,z)) * inv_x4mxx2
        #curv_array = (xxz - np.multiply(xx,z)) * inv_x4mxx2
        curv_min = np.min(curv_array)
        curv_max = np.max(curv_array)
        curv_array = (curv_array - curv_min) / (curv_max - curv_min) * maxuint16
        
        fn_loc = os.path.splitext(fn)[0] +'_curv_w'+str(w)+'.tif'
        n = curv_array.astype(np.uint16)
        tiff.imsave(fn_loc,n)
        #print_tfw(fn,fn_loc,w,GeoT)
        print_display_tfw(fn,fn_loc,w,GeoT,2)

    # TODO NOT FUNCTIONAL
    def all_curv(w,z,xz,yz,xxz,yyz,xyz,fn,GeoT):
        curv_array = np.empty((height-w+1,width-w+1))
        
        xx = (w*w - 1) / 12.
        inv_xx_sq = 1./(xx*xx)
        inv_x4mxx2 = 180. / (w*w*w*w - 5*w*w + 4)
        curv_array = (0.5*(xxz + yyz) - np.multiply(xx,z)) * inv_x4mxx2
        #curv_array = (xxz - np.multiply(xx,z)) * inv_x4mxx2

        pmp_array = np.divide((0.5 * (xxz - yyz) * inv_x4mxx2 * (xz * xz - yz * yz) + xyz * inv_xx_sq * xz * yz), (xz * xz + yz * yz)) 
        #pmp_array = np.divide((0.5 * (xxz - yyz) * (xz * xz - yz * yz) + xyz * xz * yz), (inv_x4mxx2 *(xz * xz + yz * yz))) 
        
        curv_min = np.amin(curv_array)
        curv_max = np.amax(curv_array)    
        fn_loc = os.path.splitext(fn)[0] +'_curv_w'+str(w)+'.tif'
        n = ((curv_array - curv_min) / (curv_max - curv_min) * maxuint16).astype(np.uint16)
        tiff.imsave(fn_loc,n)
        #print_tfw(fn,fn_loc,w,GeoT)
        print_display_tfw(fn,fn_loc,w,GeoT,2)
        
        prof_array = curv_array + pmp_array
        prof_min = np.amin(prof_array)
        prof_max = np.amax(prof_array)    
        fn_loc = os.path.splitext(fn)[0] +'_prof_w'+str(w)+'.tif'
        n = ((prof_array - prof_min) / (prof_max - prof_min) * maxuint16).astype(np.uint16)
        tiff.imsave(fn_loc,n)
        #print_tfw(fn,fn_loc,w,GeoT)
        print_display_tfw(fn,fn_loc,w,GeoT,2)

        plan_array = curv_array - pmp_array
        plan_min = np.amin(plan_array)
        plan_max = np.amax(plan_array)    
        fn_loc = os.path.splitext(fn)[0] +'_plan_w'+str(w)+'.tif'
        n = ((plan_array - plan_min) / (plan_max - plan_min) * maxuint16).astype(np.uint16)
        tiff.imsave(fn_loc,n)
        #print_tfw(fn,fn_loc,w,GeoT)
        print_display_tfw(fn,fn_loc,w,GeoT,2)

    # TODO NOT FUNCTIONAL
    # TODO same as planform?
    def profile(self, delta_power, arr_dic):
        delta = 2**delta_power
        z = arr_dic['z']
        yyz = arr_dic['yyz']
        xxz = arr_dic['xxz']
        y_max = arr_dic['orig_height'] - (delta-1)
        x_max = arr_dic['orig_width'] - (delta-1)
        curv_array = np.empty((y_max, x_max))
        curv_array = np.empty((y_max, x_max))
        
        xx = ((delta**2) - 1) / 12.
        inv_x4mxx2 = 180.0 / ((delta**4) - 5*(delta**2) + 4)
        curv_array = (0.5*(xxz + yyz) - np.multiply(xx,z)) * inv_x4mxx2
        #curv_array = (xxz - np.multiply(xx,z)) * inv_x4mxx2
        curv_min = np.mina(curv_array)
        curv_max = np.maxa(curv_array)
        curv_array = (curv_array - curv_min) / (curv_max - curv_min) * np.iinfo(np.uint16).max
        
        fn = os.path.splitext(self.file_name)[0] + '_profile_w' + str(delta*2) +'.tif'
        n = curv_array.astype(np.uint16)
        self.__create_tif(1, [n], delta*2, fn, 'uint16')
        #print_tfw(fn,fn_loc,w,GeoT)
        print_display_tfw(fn,fn_loc,w,GeoT,3)

    # TODO NOT FUNCTIONAL
    # TODO same as profile?
    def planform(self, delta_power, arr_dic):
        delta = 2**delta_power
        z = arr_dic['z']
        yyz = arr_dic['yyz']
        xxz = arr_dic['xxz']
        y_max = arr_dic['orig_height'] - (delta-1)
        x_max = arr_dic['orig_width'] - (delta-1)
        curv_array = np.empty((y_max, x_max))

        
        xx = ((delta**2) - 1) / 12.
        inv_x4mxx2 = 180.0 / ((delta**4) - 5*(delta**2) + 4)
        curv_array = (0.5*(xxz + yyz) - np.multiply(xx,z)) * inv_x4mxx2
        #curv_array = (xxz - np.multiply(xx,z)) * inv_x4mxx2
        curv_min = np.min(curv_array)
        curv_max = np.max(curv_array)
        curv_array = (curv_array - curv_min) / (curv_max - curv_min) * np.iinfo(np.uint16).max
        
        fn = os.path.splitext(self.file_name)[0] + '_plan_w' + str(delta*2) +'.tif'
        n = curv_array.astype(np.uint16)
        self.__create_tif(1, [n], delta*2, fn, 'uint16')
        #print_tfw(fn,fn_loc,w,GeoT)
        print_display_tfw(fn,fn_loc,w,GeoT,4)
