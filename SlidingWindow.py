import numpy as np
import rasterio
import inspect

class SlidingWindow:

    def __init__(self, file_path, band_enum):
        self.file_path = file_path
        self.img = rasterio.open(file_path)
        self.band_enum = band_enum

    __valid_ops = {'SUM', 'MAX'}
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
    def binary(self, band, threshold):
        img = self.img.read(self.band_enum[band].value)
        binary = np.where(img < threshold, 0, 255).astype(np.uint8)
        self.__create_tif(1, [binary])

    # create tif with array of image bands
    def __create_tif(self, num_bands, arr):
        profile = self.img.profile
        profile.update(
            count=num_bands,
            height=len(arr[0]),
            width=len(arr[0][0])
            )
        caller_name = inspect.stack()[1].function
        with rasterio.open(caller_name + '_' + self.file_path, 'w', **profile) as dst:
            for x in range(num_bands): 
                dst.write(arr[x], x+1)

    # TODO fix later, not the best way to do this
    def __arr_float_to_uint8(self, arr):
        max_val = np.amax(arr)
        min_val = np.amin(arr)
        arr = ((arr - min_val)/(max_val - min_val)) * 255
        arr = arr.astype(np.uint8)
        return arr

    def _aggregation_brute(self, arr, operation, num_aggre):
        if (operation.upper() not in self.__valid_ops):
            raise ValueError('operation must be one of %r.' % self.__valid_ops)

        y_max = arr.shape[0]
        x_max = arr.shape[1]

        for i in range(num_aggre):
            delta = 2**i
            y_max -= delta
            x_max -= delta

            if (operation.upper() == 'SUM'):
                arr = self.__window_sum(arr, delta)
            elif (operation.upper() == 'MAX'):
                arr = self.__window_max(arr, delta)

        return arr

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
            arr[x] = self.__arr_float_to_uint8(arr[x])
        
        self.__create_tif(num_bands, arr)

    # do power_target-power_start aggregations on window
    # starting with delta=2**power_start
    def _partial_aggregation(self, arr, power_start, power_target, operation):
        if (operation.upper() not in self.__valid_ops):
            raise ValueError('operation must be one of %r.' % self.__valid_ops)
        if (power_start < 0 or power_start >= power_target):
            raise ValueError('power_start must be nonzero and less than power_target')

        y_max = arr.shape[0]
        x_max = arr.shape[1]
        arr = arr.flatten()
        
        # calculate sliding window for each value of delta
        for x in range(power_start, power_target):
            delta = 2**x
            size = arr.size
            # create offset slices of the array to aggregate elements
            # aggregates the corners of squares of length delta+1
            top_left = arr[0: size - (delta*x_max + delta)]
            top_right = arr[delta: size - (x_max*delta)]
            bottom_left = arr[delta*x_max: size - (delta)]
            bottom_right = arr[delta*x_max + delta: size]

            if operation.upper() == 'SUM':
                arr = top_left + top_right + bottom_left + bottom_right
            elif operation.upper() == 'MAX':
                arr = np.maximum(np.maximum(np.maximum(top_left, top_right), bottom_left), bottom_right)

        # remove last removal_num rows and columns
        removal_num = 2**power_target - 2**power_start
        y_max -= removal_num
        # pad to make array square
        arr = np.pad(arr, (0, removal_num), 'constant')
        arr = np.reshape(arr, (y_max, x_max))
        arr = np.delete(arr, np.s_[-removal_num:], 1)
        
        return arr

    # Does one window-aggregation step of a 2d pixel array arr
    def __window_sum(self, arr, delta):
        y_max = arr.shape[0]-delta
        x_max = arr.shape[1]-delta
        arr_out = np.empty([y_max, x_max])

        for j in range (y_max):
            for i in range (x_max):
                arr_out[j, i] = arr[j, i] + arr[j, i+delta] + arr[j+delta, i] + arr[j+delta, i+delta]
        return arr_out

    # Does one window-aggregation step of a 2d pixel array arr
    def __window_max (self, arr, delta):
        y_max = arr.shape[0]-delta
        x_max = arr.shape[1]-delta
        arr_out = np.empty([y_max, x_max])

        for j in range (y_max):
            for i in range (x_max):
                arr_out[j, i] = max(max(max(arr[j, i], arr[j, i+delta]), arr[j+delta, i]), arr[j+delta, i+delta])
        return arr_out

    def regression(self, band1, band2, num_aggre):
        arr_a = self.img.read(self.band_enum[band1].value).astype(float)
        arr_b = self.img.read(self.band_enum[band2].value).astype(float)
        arr_m = self._regression(arr_a, arr_b, num_aggre)

        # TODO remove later
        arr_m = self.__arr_float_to_uint8(arr_m)

        self.__create_tif(1, [arr_m])

    # Do num_aggre aggregations and return the regression slope between two bands
    def _regression(self, arr_a, arr_b, num_aggre):
        arr_aa = arr_a**2
        arr_ab = arr_a*arr_b

        arr_a = self._partial_aggregation(arr_a, 0, num_aggre, 'sum')
        arr_b = self._partial_aggregation(arr_b, 0, num_aggre, 'sum')
        arr_aa = self._partial_aggregation(arr_aa, 0, num_aggre, 'sum')
        arr_ab = self._partial_aggregation(arr_ab, 0, num_aggre, 'sum')

        # total pixels aggregated per pixel
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
        arr_r = self.__arr_float_to_uint8(arr_r)

        self.__create_tif(1, [arr_r])

    # Do num_aggre aggregations and return the regression slope between two bands
    def _pearson(self, arr_a, arr_b, num_aggre):
        arr_aa = arr_a**2
        arr_bb = arr_b**2
        arr_ab = arr_a*arr_b

        arr_a = self._partial_aggregation(arr_a, 0, num_aggre, 'sum')
        arr_b = self._partial_aggregation(arr_b, 0, num_aggre, 'sum')
        arr_aa = self._partial_aggregation(arr_aa, 0, num_aggre, 'sum')
        arr_bb = self._partial_aggregation(arr_bb, 0, num_aggre, 'sum')
        arr_ab = self._partial_aggregation(arr_ab, 0, num_aggre, 'sum')

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
                arr_out = np.polyfit(arr_1, arr_2, 1)[0]
                arr_m[j][i] = arr_out

        return arr_m

    def fractal(self, band, power_start, power_target):
        if (power_start < 0 or power_start >= power_target):
            raise ValueError('power_start must be nonzero and less than power_target')
        arr = self.img.read(self.band_enum[band].value).astype(float)
        arr, residuals = self.__fractal(arr, power_start, power_target)

        # TODO remove later
        arr = self.__arr_float_to_uint8(arr)

        self.__create_tif(1, [arr])

    def __fractal(self, arr, power_start, power_target):
        if (power_start < 0 or power_start >= power_target):
            raise ValueError('power_start must be nonzero and less than power_target')
        arr_init = np.array(arr)
        y_max = arr.shape[0]-(2**power_target-1)
        x_max = arr.shape[1]-(2**power_target-1)
        denom_regress = np.empty(power_target-power_start)
        num_regress = np.empty((power_target-power_start, x_max*y_max))
        
        arr_init = self._partial_aggregation(arr_init, 0, power_start, 'max')

        for i in range(power_start, power_target):
            arr_sum = self._partial_aggregation(arr_init, i, power_target, 'sum')
            arr_sum = np.maximum(arr_sum, 1)

            arr_sum = np.log(arr_sum)/np.log(2)
            denom = power_target-i
            denom_regress[i-power_start] = denom
            num_regress[i-power_start,] = arr_sum.flatten()
            if i < power_target-1:
                arr_init = self._partial_aggregation(arr_init, i, i+1, 'max')

        arr_out, residuals, rank, singular_values, rcond = np.polyfit(denom_regress, num_regress, 1, full=True)
        arr_out = np.reshape(arr_out[0], (y_max, x_max))
        return arr_out, residuals