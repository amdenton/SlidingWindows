import numpy as np
from SlidingWindow import SlidingWindow
from BandEnum import rgbIr
import rasterio
import matplotlib.pyplot as plt

slide_window = SlidingWindow('m_4509601_ne_14_1_20120705.tif', rgbIr)

# print('BINARY IAMGE')
# slide_window.binary('red', 127)
print('VEC SLIDING WINDOW')
slide_window.window_agg('red', 'sum', 6)
print('BAD SLIDING WINDOW')
slide_window._window_agg_brute('red', 'sum', 6)

img_og = rasterio.open('m_4509601_ne_14_1_20120705.tif')
img_good = rasterio.open('window_agg_m_4509601_ne_14_1_20120705.tif')
img_bad = rasterio.open('_window_agg_brute_m_4509601_ne_14_1_20120705.tif')
# img_binary = rasterio.open('binary_m_4509601_ne_14_1_20120705.tif')

# plt.imshow(img_og.read(1))
# plt.show()

print('EQUAL? ', np.array_equal(img_good.read(1), img_bad.read(1)))