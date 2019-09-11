import numpy as np
import rasterio
import inspect
import time

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

    # do num_aggre aggregations on window
    def _aggregation_brute(self, band, operation, num_aggre):
        if (operation.upper() not in self.__valid_ops):
            raise ValueError('operation must be one of %r.' % self.__valid_ops)

        arr = self.img.read(self.band_enum[band].value).astype(float)
        y_max = arr.shape[0]
        x_max = arr.shape[1]

        for i in range(num_aggre):
            delta = 2**i
            y_max -= delta
            x_max -= delta

            if (operation.upper() == 'SUM'):
                arr = self.__windowSum(arr, delta)
            elif (operation.upper() == 'MAX'):
                arr = self.__windowMax(arr, delta)

        # TODO remove later
        arr = self.__arr_float_to_uint8(arr)

        self.__create_tif(1, [arr])

    # convert band into array, then call actual aggregation function
    def aggregation(self, operation, num_aggre):
        if (operation.upper() not in self.__valid_ops):
            raise ValueError('operation must be one of %r.' % self.__valid_ops)
        
        arr = []
        num_bands = len(self.band_enum)
        for x in range(num_bands):
            arr.append(self.img.read(x+1).astype(float))
            arr[x] = self.__aggregation(arr[x], operation, num_aggre)

            # TODO remove later
            arr[x] = self.__arr_float_to_uint8(arr[x])
        
        self.__create_tif(num_bands, arr)

    # do num_aggre aggregations on window
    def __aggregation(self, arr, operation, num_aggre):        
        y_max = arr.shape[0]
        x_max = arr.shape[1]
        arr = arr.flatten()
        
        # calculate sliding window for each value of delta
        for x in range(num_aggre):
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
        removal_num = 2**num_aggre - 1
        y_max -= removal_num
        # pad to make array square
        arr = np.pad(arr, (0, removal_num), 'constant')
        arr = np.reshape(arr, (y_max, x_max))
        arr = np.delete(arr, np.s_[-removal_num:], 1)

        return arr

    # Does one window-aggregation step of a 2d pixel array arr
    def __windowSum(self, arr, delta):
        y_max = arr.shape[0]
        x_max = arr.shape[1]
        y_max_out = y_max-delta
        x_max_out = x_max-delta
        arr_out = np.empty([y_max_out, x_max_out])

        for j in range (y_max_out):
            for i in range (x_max_out):
                arr_out[j, i] = arr[j, i] + arr[j, i+delta] + arr[j+delta, i] + arr[j+delta, i+delta]
        return arr_out

    # Does one window-aggregation step of a 2d pixel array arr
    def __windowMax (self, arr, delta):
        y_max = arr.shape[0]
        x_max = arr.shape[1]
        y_max_out = y_max-delta
        x_max_out = x_max-delta
        arr_out = np.empty([y_max_out, x_max_out])

        for j in range (y_max_out):
            for i in range (x_max_out):
                arr_out[j, i] = max(max(max(arr[j, i], arr[j, i+delta]), arr[j+delta, i]), arr[j+delta, i+delta])
        return arr_out

    # Do num_aggre aggregations and return the regression slope between two bands
    def regression(self, band1, band2, num_aggre):
        arr_a = self.img.read(self.band_enum[band1].value).astype(float)
        arr_b = self.img.read(self.band_enum[band2].value).astype(float)
        arr_aa = arr_a*arr_a
        arr_ab = arr_a*arr_b

        arr_a = self.__aggregation(arr_a, 'sum', num_aggre)
        arr_b = self.__aggregation(arr_b, 'sum', num_aggre)
        arr_aa = self.__aggregation(arr_aa, 'sum', num_aggre)
        arr_ab = self.__aggregation(arr_ab, 'sum', num_aggre)

        # Allow division by zero
        # TODO is this necessary? denominator is only zero when x is constant
        np.seterr(divide='ignore', invalid='ignore')

        # total pixels aggregated per pixel
        count = (2**num_aggre)**2
        numerator = count * arr_ab - arr_a * arr_b
        denominator = count * arr_aa - arr_a * arr_a
        # regression coefficient of linear least squares fitting
        # m = cov(a,b) / var(x)
        arr_m = numerator/denominator

        # TODO remove later
        arr_m = self.__arr_float_to_uint8(arr_m)

        self.__create_tif(1, [arr_m])