from osgeo import gdal
from PIL import Image
import sys
import numpy as np
from BandEnum import BandEnum
# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()

class Raster:

    def __init__(self, imgSource):
        self.imgSource = imgSource
        try:
            self.img = gdal.Open(imgSource)
        except RuntimeError as e:
            print('Unable to open' + imgSource + '.tif')
            print(e)
            sys.exit(1)

        self.xMax = self.img.RasterXSize
        self.yMax = self.img.RasterYSize
        self.bandCount = self.img.RasterCount
        self.info = gdal.Info(self.img)
        self.metadata = self.img.GetMetadata()
        self.NDVI = None

    def getBand(self, bandColorNum):
        try:
            srcband = self.img.GetRasterBand(bandColorNum)
        except RuntimeError as e:
            print('Band ( %i ) not found' % bandColorNum)
            print(e)
            sys.exit(1)
        return srcband

    def bandStats(self):
        for band in range( self.bandCount ):
            band += 1
            print("[ GETTING BAND ]: " + str(band))
            srcband = self.img.GetRasterBand(band)
            if srcband is None:
                continue

            stats = srcband.GetStatistics( True, True )
            if stats is None:
                continue

            print("[ STATS ] =  Minimum=%.3f, Maximum=%.3f, Mean=%.3f, StdDev=%.3f" % ( \
                stats[0], stats[1], stats[2], stats[3] ))

    def get1DBandArray(self, bandColorNum):
        return self.img.GetRasterBand(bandColorNum).ReadAsArray().flatten()


    def createNDVI(self):
        redBand = self.get1DBandArray(BandEnum.Red.value)
        irBand = self.get1DBandArray(BandEnum.IR.value)
        sumArr = irBand + redBand
        # replace 0s with 1s, cant divide by zero
        sumArr[sumArr == 0] = 1
        differenceArr = (irBand - redBand)
        ndviBand = differenceArr / sumArr
        self.NDVI = ndviBand