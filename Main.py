import numpy as np
from windowagg.sliding_window import SlidingWindow
from image_generator import ImageGenerator
import rasterio
import matplotlib.pyplot as plt
import math

img_gen = ImageGenerator()
slide_window = SlidingWindow('test.tif')
slide_window_2 = SlidingWindow('gunsite_dem-2-1_export_w64.tif')
cell_width = 4928
cell_height = 3264

img_gen.gauss(image_size=300, prefactor=100, sigma=30, mu=150, noise=0)
img_gen.gauss_x(image_size=300, prefactor=100, sigma=100, mu=20 , noise=0)
img_gen.cone(image_size=300, mu=150)
img_gen.se_gradient()
img_gen.nw_gradient()
img_gen.s_gradient()
img_gen.n_gradient()
img_gen.random(num_bands=4)

#img = rasterio.open('gunsite_dem-2-1.tif').read(1)
#plt.imshow(img)
#plt.show()
#img = rasterio.open('gunsite_dem-2-1_z_mean_w64.tif').read(1)
#plt.imshow(img)
#plt.show()
#img = rasterio.open('gunsite_dem-2-1_standard_w64.tif').read(1)
#plt.imshow(img)
#plt.show()
#img = rasterio.open('gunsite_dem-2-1_slope_w64.tif').read(1)
#plt.imshow(img)
#plt.show()
#img = rasterio.open('gunsite_dem-2-1_profile_w64.tif').read(1)
#plt.imshow(img)
#plt.show()
#img = rasterio.open('gunsite_dem-2-1_planform_w64.tif').read(1)
#plt.imshow(img)
#plt.show()
#img = rasterio.open('gunsite_dem-2-1_aspect_w64.tif').read(1)
#plt.imshow(img)
#plt.show()

#slide_window_2.dem_import_arrays()
#slide_window_2.dem_initialize_arrays()
#for _ in range(6):
#    slide_window_2.dem_aggregation_step(1)
#    slide_window_2.dem_export_arrays()
#slide_window_2.dem_mean()
#slide_window_2.dem_slope(cell_width, cell_height)
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