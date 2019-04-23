from osgeo import gdal
from PIL import Image
import sys
import numpy as np
from BandEnum import BandEnum
# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()

class RasterOld:

    def __init__(self, img_source):
        self.img_source = img_source
        try:
            self.img = gdal.Open(img_source)
        except RuntimeError as e:
            print('Unable to open' + img_source)
            print(e)
            sys.exit(1)

        self.x_max = self.img.RasterXSize
        self.y_max = self.img.RasterYSize
        self.band_count = self.img.RasterCount
        self.info = gdal.Info(self.img)
        self.metadata = self.img.GetMetadata()
        self.ndvi = None

    def get_band(self, band_color_num):
        try:
            band = self.img.GetRasterBand(band_color_num)
        except RuntimeError as e:
            print('Band ( %i ) not found' % band_color_num)
            print(e)
            sys.exit(1)
        return band

    def band_stats(self):
        for band in range( self.band_count ):
            band += 1
            print("[ GETTING BAND ]: " + str(band))
            src_band = self.img.GetRasterBand(band)
            if src_band is None:
                continue
            stats = src_band.GetStatistics( True, True )
            if stats is None:
                continue

            print("[ STATS ] =  Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % ( \
                stats[0], stats[1], stats[2], stats[3] ))

    def get_1d_band_array(self, bandColorNum):
        return self.img.GetRasterBand(bandColorNum).ReadAsArray().flatten()


    def create_ndvi(self):
        red_band = self.get_1d_band_array(BandEnum.red.value)
        ir_band = self.get_1d_band_array(BandEnum.ir.value)
        sum_arr = ir_band + red_band
        # replace 0s with 1s, cant divide by zero
        sum_arr[sum_arr == 0] = 1
        difference_arr = (ir_band - red_band)
        ndvi_band = difference_arr / sum_arr
        self.ndvi = ndvi_band