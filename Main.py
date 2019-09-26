import numpy as np
from SlidingWindow import SlidingWindow
from BandEnum import rgbIr
import rasterio

slide_window = SlidingWindow('test.tif', rgbIr)
slide_window_2 = SlidingWindow('dem/gunsite_dem-2-1.tif', rgbIr)

slide_window_2.dem_utils(1)
# slide_window.fractal_3d('red', 6)
# slide_window.fractal('red', 3, 6)
# slide_window.binary('red', 127)
# slide_window.regression('red', 'green', 6)
# slide_window.pearson('red', 'green', 6)
# slide_window.aggregation('++++', 6)

# img_og = rasterio.open('m_4509601_ne_14_1_20120705.tif')
# img_regression = rasterio.open('regression_m_4509601_ne_14_1_20120705.tif')
# img_good = rasterio.open('aggregation_m_4509601_ne_14_1_20120705.tif')
# img_binary = rasterio.open('binary_m_4509601_ne_14_1_20120705.tif')