from osgeo import gdal

class Band:

    def __init__(self, band):
        self.array2D = band.ReadAsArray()
        self.yMax = len(self.array2D)
        self.xMax = len(self.array2D[0])
        self.array = []
        self.readImage(0, 0, self.array)

    def get2DArray(self):
        return self.array2D

    def getArray(self):
        return self.array

    # Creates 1D pixel array from 2D array
    # xStart, yStart: pixel indices in image where reading begins
    # self.xMax, self.yMax: array dimensions
    def readImage (self, xStart, yStart, array):
        for j in range (0, self.yMax):
            for i in range (0, self.xMax):
                value = self.array2D[yStart+j, xStart+i]
                array.append(value)