import numpy as np
from SlidingWindow import SlidingWindow
import rasterio
import matplotlib.pyplot as plt
import math

slide_window = SlidingWindow('test.tif')
slide_window_2 = SlidingWindow('dem/gunsite_dem-2-1.tif')

# slide_window_2.dem_utils(6)
# slide_window.fractal_3d(1, 6)
# slide_window.fractal(1, .5, 0, 6)
# slide_window.binary(1, .5)
# slide_window.regression(1, 2, 6)
# slide_window.pearson(1, 2, 6)
# slide_window.aggregation('++++', 6)
# slide_window.ndvi(1, 4)