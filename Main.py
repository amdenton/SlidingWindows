import numpy as np
from windowagg.sliding_window import SlidingWindow
from windowagg.utilities import _Utilities as util
from image_generator import ImageGenerator
import rasterio
import matplotlib.pyplot as plt
import math
import os

#img_gen = ImageGenerator()
#img_gen.all()

#slide_window = SlidingWindow('test.tif')
#slide_window_2 = SlidingWindow('test_img/gauss_0skew_1-1offset.tif')
#slide_window_2 = SlidingWindow('gunsite_dem-2-1_export_w64.tif', cell_width=4928, cell_height=3264)

#img = rasterio.open('test_img/se_gradient_0skew_1-1offset.tif').read(1)
#plt.imshow(img)
#plt.show()

#slide_window_2.dem_import_arrays()
#slide_window_2.dem_initialize_arrays()
#for _ in range(5):
#    slide_window_2.dem_aggregation_step(1)
#    slide_window_2.dem_export_arrays()
#slide_window_2.dem_mean()
#slide_window_2.dem_slope()
#slide_window_2.dem_slope_angle()
#slide_window_2.dem_aspect()
#slide_window_2.dem_profile()
#slide_window_2.dem_planform()
#slide_window_2.dem_standard()

# slide_window.fractal_3d(1, 6)
# slide_window.fractal(1, .5, 0, 6)
# slide_window.binary(1, .5)
# slide_window.regression(1, 2, 6)
# slide_window.pearson(1, 2, 6)
# slide_window.aggregation('++++', 6)
# slide_window.ndvi(1, 4)