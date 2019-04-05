from osgeo import gdal
from PIL import Image
import sys
# this allows GDAL to throw Python Exceptions
gdal.UseExceptions()

class Raster:

    def __init__(self, imgSource):
        self.imgSource = imgSource
        try:
            self.img = gdal.Open(imgSource)
        except RuntimeError as e:
            print('Unable to open INPUT.tif')
            print(e)
            sys.exit(1)

        self.bandCount = self.img.RasterCount
        self.bandNames = []
        self.createBandColorInterpretations()

    def createBandColorInterpretations(self):
        for band in range(self.bandCount):
            band += 1
            self.bandNames.append(self.img.GetRasterBand(band).GetColorInterpretation())

    def GetColorInterpretationName(self, bandColorNum):
        return gdal.GetColorInterpretationName(bandColorNum)

    def getBand(self, bandColorNum):
        try:
            srcband = self.img.GetRasterBand(bandColorNum)
        except RuntimeError as e:
            print('Band ( %i ) not found' % bandColorNum)
            print(e)
            sys.exit(1)
        return srcband

    def get1DBandArray(self, bandColorNum):
        return self.img.GetRasterBand(bandColorNum).ReadAsArray().flatten() 

    def getInfo(self):
        return gdal.Info(self.img)

    def getMetadata(self):
        return self.img.GetMetadata()