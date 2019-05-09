import numpy as np
from Raster import Raster
from SlidingWindow import SlidingWindow
import matplotlib.pyplot as plt

from BandEnum import BandEnum

rast = Raster("m_4509601_ne_14_1_20120705.tif")
slide_window = SlidingWindow(rast.img.meta['height'], rast.img.meta['width'], 1)

# print(rast.img.meta)

# arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
arr = np.random.randint(2, size=(7700,5650))
img = rast.img.read(BandEnum.red.value)
img = rast.create_binary_image(img, 127)
print(img.shape[0])
print(img.shape[1])
img2 = slide_window.analyze(arr, 'sum')
img3 = slide_window.analyze_test(arr, 'sum')
print('EQUAL? ', np.array_equal(img2, img3))

test = np.zeros(img2.shape)
y_max = img2.shape[0]
x_max = img2.shape[1]

print(img2)
print(img3)

# rast.create_ndvi()

# plt.imshow(img2)
# plt.show()

# indexing practice
# arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# print(arr)
# print('')
# print(arr[...,-3:])
# print(np.delete(arr, arr[...,-3:], 1))
