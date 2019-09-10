import numpy as np
from SlidingWindow import SlidingWindow
from BandEnum import rgbIr
import rasterio
import matplotlib.pyplot as plt

slide_window = SlidingWindow('m_4509601_ne_14_1_20120705.tif', rgbIr)

# slide_window.binary('red', 127)
# slide_window.regression('red', 'green', 6)
slide_window.aggregation('red', 'sum', 6)
slide_window._aggregation_brute('red', 'sum', 6)

img_og = rasterio.open('m_4509601_ne_14_1_20120705.tif')
# img_regression = rasterio.open('regression_m_4509601_ne_14_1_20120705.tif')
img_good = rasterio.open('aggregation_m_4509601_ne_14_1_20120705.tif')
img_bad = rasterio.open('_aggregation_brute_m_4509601_ne_14_1_20120705.tif')
# img_binary = rasterio.open('binary_m_4509601_ne_14_1_20120705.tif')

# plt.imshow(img_og.read(1))
# plt.show()
# plt.imshow(img_regression.read(1))
# plt.show()

print('EQUAL? ', np.array_equal(img_good.read(1), img_bad.read(1)))