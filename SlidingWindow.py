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

        self.__create_tif(ndvi)
        
    # turn image into black and white
    def binary(self, band, threshold):
        img = self.img.read(self.band_enum[band].value)
        binary = np.where(img < threshold, 0, 255).astype(np.uint8)
        self.__create_tif(binary)

    def __create_tif(self, img_arr):
        profile = self.img.profile
        profile.update(count=1)
        caller_name = inspect.stack()[1].function
        with rasterio.open(caller_name + '_' + self.file_path, 'w', **profile) as dst:
            dst.write(img_arr, 1)

    # do num_aggre aggregations on window
    def _window_agg_brute(self, band, operation, num_aggre):
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
        # for testing purposed to convert float to uint8
        max_val = np.amax(arr)
        min_val = np.amin(arr)
        arr = ((arr - min_val)/(max_val - min_val)) * 255
        arr = arr.astype(np.uint8)

        self.__create_tif(arr)

    # convert band into array, then call actual aggregation function
    def window_agg(self, band, operation, num_aggre):
        if (operation.upper() not in self.__valid_ops):
            raise ValueError('operation must be one of %r.' % self.__valid_ops)
        arr = self.img.read(self.band_enum[band].value).astype(float)
        arr = self.__window_agg(arr, operation, num_aggre)
        self.__create_tif(arr)

    # do num_aggre aggregations on window
    def __window_agg(self, arr, operation, num_aggre):        
        y_max = arr.shape[0]
        x_max = arr.shape[1]
        arr = arr.flatten()
        
        # calculate sliding window for each value of delta
        for x in range(num_aggre):
            delta = 2**x
            size = arr.shape[0]

            # create offset slices of the array to aggregate elements
            # aggregates the corners of squares of length delta+1
            top_left = arr[0: size - (delta*x_max + delta)]
            top_right = arr[delta: size - (x_max*delta)]
            bottom_left = arr[delta*x_max: size - (delta)]
            bottom_right = arr[delta*x_max + delta: size]

            # operate on arrays
            if operation.upper() == 'SUM':
                arr = top_left + top_right + bottom_left + bottom_right
            elif operation.upper() == 'MAX':
                arr = np.maximum(np.maximum(np.maximum(top_left, top_right), bottom_left), bottom_right)

        # number of rows and columns removed from the ends of the array
        pad_num = 2**num_aggre - 1
        # last pad_num rows already removed
        y_max -= pad_num
        # pad to make array square
        arr = np.pad(arr, (0, pad_num), 'constant')
        arr = np.reshape(arr, (y_max, x_max))
        # truncate last pad_num columns
        arr = np.delete(arr, np.s_[-pad_num:], 1)

        # TODO remove later
        # for testing purposed to convert float to uint8
        max_val = np.amax(arr)
        min_val = np.amin(arr)
        arr = ((arr - min_val)/(max_val - min_val)) * 255
        arr = arr.astype(np.uint8)

        return arr

    # Regression paper
    # Create an array of window-based regression slopes for precalculated sums of x (a) and y (b), sums of squares of x (aa) and sums of xy (ab)
    # Uses the original image dimensions x_max and y_max and window size delta.  Note that the actual array sizes are smaller than x_max*y_max
    def __regression(self, a, b, aa, ab, x_max, y_max, delta):
        # TODO implement this

    # Does one window-aggregation step of a 2d pixel array arr
    def __windowSum (self, arr, delta):
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

    # Regression paper
    # Do num_aggre aggregations and return the regression slope
    def windowRegression(self, band1, band2, num_aggre):
        arr1 = self.img.read(self.band_enum[band1].value).astype(float)
        arr2 = self.img.read(self.band_enum[band2].value).astype(float)
        y_max = arr1.shape[0]
        x_max = arr1.shape[1]

        arr11 = arr1*arr1
        arr22 = arr2*arr2
        arr12 = arr1*arr2

        arr1 = self.__window_agg(arr1, 'sum', num_aggre)
        arr2 = self.__window_agg(arr2, 'sum', num_aggre)
        arr11 = self.__window_agg(arr11, 'sum', num_aggre)
        arr22 = self.__window_agg(arr22, 'sum', num_aggre)
        arr12 = self.__window_agg(arr12, 'sum', num_aggre)

        m = self.__regression(arr1, arr2 , arr11 , arr12, x_max, y_max, num_aggre*2)
        return m