import numpy as np
from SlidingWindow import SlidingWindow
from BandEnum import rgbIr
import rasterio
import time

slide_window = SlidingWindow('test.tif', rgbIr)

# slide_window.binary('red', 127)
# slide_window.regression('red', 'green', 6)
slide_window.pearson('red', 'green', 6)
# slide_window.aggregation('sum', 6)

# img_og = rasterio.open('m_4509601_ne_14_1_20120705.tif')
# img_regression = rasterio.open('regression_m_4509601_ne_14_1_20120705.tif')
# img_good = rasterio.open('aggregation_m_4509601_ne_14_1_20120705.tif')
# img_binary = rasterio.open('binary_m_4509601_ne_14_1_20120705.tif')