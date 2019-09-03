import numpy as np
from SlidingWindow import SlidingWindow
from BandEnum import rgbIr
import rasterio
import matplotlib.pyplot as plt

slide_window = SlidingWindow('m_4509601_ne_14_1_20120705.tif', rgbIr)

""" print('VEC SLIDING WINDOW')
slide_window.sliding_window_vec('red', 'sum', 5)
print('BAD SLIDING WINDOW')
slide_window.sliding_window_bad('red', 'sum', 5) """

img_og = rasterio.open('m_4509601_ne_14_1_20120705.tif')
img_good = rasterio.open('sliding_window_vec_m_4509601_ne_14_1_20120705.tif')
img_bad = rasterio.open('sliding_window_bad_m_4509601_ne_14_1_20120705.tif')

plt.imshow(img_og.read(1))
plt.show()
plt.imshow(img_good.read(1))
plt.show()

# print('EQUAL? ', np.array_equal(img_good.read(1), img_bad.read(1)))