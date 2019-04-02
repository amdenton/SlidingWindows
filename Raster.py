from osgeo import gdal
from PIL import Image
import sys

class Raster:

    def __init__(self, imgSource):
        self.imgSource = imgSource
        self.img = gdal.Open(self.imgSource)
        self.bandCount = self.img.RasterCount
        self.bandDict = {}
        if self.img is None:
            print('Unable to open' + self.imgSource)
            sys.exit(1)
        self.createBands()

    def createBands(self):
       for band in range(self.img.RasterCount):
           band += 1
           self.bandDict[gdal.GetColorInterpretationName(self.img.GetRasterBand(band).GetColorInterpretation())] = self.img.GetRasterBand(band)

    def getImgSource(self):
        return self.imgSource
        
    def getBandCount(self):
        return self.bandCount

    def getInfo(self):
        return gdal.Info(self.img)
    
    def getBands(self):
        return self.bandDict