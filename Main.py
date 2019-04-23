import numpy as np
from Raster import Raster
from SlidingWindow import SlidingWindow
import matplotlib.pyplot as plt

from BandEnum import BandEnum

rast = Raster("m_4509601_ne_14_1_20120705.tif")
slide_window = SlidingWindow(rast.img.meta['height'], rast.img.meta['width'], 1)

print(rast.img.meta)

img = rast.img.read(BandEnum.red.value)
rast.create_binary_image(img, 124)
rast.create_ndvi()

plt.imshow(rast.ndvi)
plt.show()
