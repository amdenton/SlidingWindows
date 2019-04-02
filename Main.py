from Band import Band 
from Raster import Raster
from osgeo import gdal

rast = Raster("m_4509601_ne_14_1_20120705.tif")
#rast = Raster("r322_nir_reraster.tif")
#print(rast.getInfo())
#print(rast.getBands()["Red"].ReadAsArray())
band = Band(rast.getBands()["Red"])
print(band.get2DArray())
print(band.getArray())