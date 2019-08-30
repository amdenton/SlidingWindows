import numpy as np
import rasterio
import time

class SlidingWindow:

    valid_ops = {'SUM', 'MAX'}

    def __init__(self, img_source, band_enum):
        self.img = rasterio.open(img_source)
        self.band_enum = band_enum

    def sliding_window_bad(self, band, operation, max_delta_power):
        if (operation.upper() not in self.valid_ops):
            raise ValueError('operation must be one of %r.' % self.valid_ops)

        start = time.time()  # TODO remove time
        pixels = self.img.read(self.band_enum[band].value).astype(float) # TODO change dtype later?
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

        return pixels

    def sliding_window_vec(self, band, operation, max_delta_power):
        if (operation.upper() not in self.valid_ops):
            raise ValueError('operation must be one of %r.' % self.valid_ops)
        
        start = time.time()  # TODO remove time
        pixels = self.img.read(self.band_enum[band].value).astype(float) # TODO change dtype later?
        y_max = pixels.shape[0]
        x_max = pixels.shape[1]
        pixels = pixels.flatten() # TODO change dtype later?
        
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

        return pixels

    # i.e. Normalized Difference Vegetation Index
    # for viewing live green vegetation
    def create_ndvi(self):
        red_band = self.img.read(self.band_enum.red.value)
        ir_band = self.img.read(self.band_enum.ir.value)
        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        ndvi_band = (ir_band.astype(float) - red_band.astype(float)) / (ir_band + red_band)
        return ndvi_band

        
    # turn image into black and white
    # pixels are black if below threshold, white if greater than or equal to threshold
    def create_binary_image(self, img, threshold):
        binary_image = np.array(img)
        binary_image[binary_image < threshold] = 0
        binary_image[binary_image >= threshold] = 255
        return binary_image