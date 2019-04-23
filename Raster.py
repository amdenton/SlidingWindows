import rasterio
from PIL import Image
import sys
import numpy as np
import time
from BandEnum import BandEnum

class Raster:

    def __init__(self, img_source):
        self.img_source = img_source
        self.img = rasterio.open(img_source)
        self.ndvi = None
        self.binary = None

    def create_ndvi(self):
        red_band = self.img.read(BandEnum.red.value)
        ir_band = self.img.read(BandEnum.ir.value)
        # Allow division by zero
        np.seterr(divide='ignore', invalid='ignore')
        ndvi_band = (ir_band.astype(float) - red_band.astype(float)) / (ir_band + red_band)
        self.ndvi = ndvi_band

    def create_binary_image(self, img, threshold):
        binary_image = np.array(img)
        binary_image[binary_image < threshold] = 0
        binary_image[binary_image >= threshold] = 1
        self.binary = binary_image
    