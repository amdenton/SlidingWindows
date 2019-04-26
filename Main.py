import numpy as np
from Raster import Raster
from SlidingWindow import SlidingWindow
import matplotlib.pyplot as plt

from BandEnum import BandEnum

rast = Raster("m_4509601_ne_14_1_20120705.tif")
slide_window = SlidingWindow(rast.img.meta['height'], rast.img.meta['width'], 1)

# print(rast.img.meta)

# arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# arr = np.array([[1,1,1,0],[0,1,0,1],[0,0,1,1],[1,0,1,1]])
img = rast.img.read(BandEnum.red.value )
img = rast.create_binary_image(img, 127)
img2 = slide_window.analyze_number_two(img, 'sum')

# rast.create_ndvi()

plt.imshow(img2)
plt.show()

# indexing practice
# arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# print(arr)
# print('')
# print(arr[...,-3:])
# print(np.delete(arr, arr[...,-3:], 1))
