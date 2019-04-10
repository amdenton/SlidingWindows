import numpy as np
from Raster import Raster
from BandEnum import BandEnum

rast = Raster("m_4509601_ne_14_1_20120705.tif")
#rast = Raster("r322_nir_reraster.tif")
#print(rast.getMetadata())
#print(rast.get1DBandArray(BandEnum.Red))
#print(rast.info)
rast.createNDVI()
print(rast.NDVI)