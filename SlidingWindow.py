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

    def __create_tif(self, img_arr):
        profile = self.img.profile
        profile.update(count=1)
        caller_name = inspect.stack()[1].function
        with rasterio.open(caller_name + '_' + self.file_path, 'w', **profile) as dst:
            dst.write(img_arr, 1)



    def sliding_window_bad(self, band, operation, max_delta_power):
        if (operation.upper() not in self.__valid_ops):
            raise ValueError('operation must be one of %r.' % self.__valid_ops)

        start = time.time()  # TODO remove time
        pixels = self.img.read(self.band_enum[band].value).astype(float)
        y_max = pixels.shape[0]
        x_max = pixels.shape[1]

        for i in range(max_delta_power+1):
            delta = 2**i
            y_max -= delta
            x_max -= delta

            new_pixels = np.zeros((y_max, x_max), dtype=pixels.dtype)
            for y in range(y_max):
                for x in range(x_max):
                    if (operation.upper() == 'SUM'):
                        new_pixels[y, x] = pixels[y, x] + pixels[y, x+delta] + pixels[y+delta, x] + pixels[y+delta, x+delta]
                    elif (operation.upper() == 'MAX'):
                        new_pixels[y, x] = np.maximum(np.maximum(np.maximum(pixels[y, x], pixels[y, x+delta]), pixels[y+delta, x]), pixels[y+delta, x+delta])
            pixels = new_pixels

        end = time.time() # TODO remove time
        print('BAD TIME: ', end-start)

        # TODO remove later
        # for testing purposed to convert float to uint8
        max_val = np.amax(pixels)
        min_val = np.amin(pixels)
        pixels = ((pixels - min_val)/(max_val - min_val)) * 255
        pixels = pixels.astype(np.uint8)

        self.__create_tif(pixels)

    def sliding_window_vec(self, band, operation, max_delta_power):
        if (operation.upper() not in self.__valid_ops):
            raise ValueError('operation must be one of %r.' % self.__valid_ops)
        
        start = time.time()  # TODO remove time
        pixels = self.img.read(self.band_enum[band].value).astype(float)
        y_max = pixels.shape[0]
        x_max = pixels.shape[1]
        pixels = pixels.flatten()
        
        # calculate sliding window for each value of delta
        for x in range(max_delta_power+1):
            delta = 2**x
            size = pixels.shape[0]

            # create offset slices of the array to aggregate elements
            # aggregates the corners of squares of length delta+1
            top_left = pixels[0: size - (delta*x_max + delta)]
            top_right = pixels[delta: size - (x_max*delta)]
            bottom_left = pixels[delta*x_max: size - (delta)]
            bottom_right = pixels[delta*x_max + delta: size]

            # operate on arrays
            if operation.upper() == 'SUM':
                pixels = top_left + top_right + bottom_left + bottom_right
            elif operation.upper() == 'MAX':
                pixels = np.maximum(np.maximum(np.maximum(top_left, top_right), bottom_left), bottom_right)

        # number of rows and columns removed from the ends of the array
        pad_num = 2**(max_delta_power+1) - 1
        # last pad_num rows already removed
        y_max -= pad_num
        # pad to make array square
        pixels = np.pad(pixels, (0, pad_num), 'constant')
        pixels = np.reshape(pixels, (y_max, x_max))
        # truncate last pad_num columns
        pixels = np.delete(pixels, np.s_[-pad_num:], 1)

        end = time.time() # TODO remove time
        print('GOOD TIME: ', end-start)

        # TODO remove later
        # for testing purposed to convert float to uint8
        max_val = np.amax(pixels)
        min_val = np.amin(pixels)
        pixels = ((pixels - min_val)/(max_val - min_val)) * 255
        pixels = pixels.astype(np.uint8)

        self.__create_tif(pixels)

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
    # pixels are black if below threshold, white if greater than or equal to threshold
    def binary_image(self, img, threshold):
        binary_image = np.array(img)
        binary_image[binary_image < threshold] = 0
        binary_image[binary_image >= threshold] = 255
        return binary_image