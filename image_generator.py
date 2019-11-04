import rasterio
import math
import affine
import numpy as np
from windowagg.utilities import _Utilities
from windowagg.sliding_window import SlidingWindow
import numpy.random as rand

class ImageGenerator:

    def gauss(self, image_size, prefactor, sigma, mu, noise):
        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = prefactor*(math.exp(-((x-mu)**2 + (y-mu)**2) / sigma**2) + noise*rand.normal())
                arr[y][x] = value

        arr = _Utilities._arr_dtype_conversion(arr, np.uint8)
        _Utilities._create_new_tif(arr, 'test_img/gauss_image_no_noise.tif')

    def gauss_x(self, image_size, prefactor, sigma, mu, noise):
        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = prefactor*(math.exp(-((x-mu)**2) / sigma**2) + noise*rand.normal())
                arr[y][x] = value

        arr = _Utilities._arr_dtype_conversion(arr, np.uint8)
        _Utilities._create_new_tif(arr, 'test_img/gauss_image_x_small.tif')

    def cone(self, image_size, mu):
        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = (1 - math.sqrt((x-mu)**2 + (y-mu)**2) / image_size)
                arr[y][x] = value

        arr = _Utilities._arr_dtype_conversion(arr, np.uint8)
        _Utilities._create_new_tif(arr, 'test_img/cone_image.tif')

    def se_gradient(self, angle=0):
        arr = np.empty([128, 129]).astype(np.uint8)

        i = j = 0
        for y in range(arr.shape[0]):
            i = j
            for x in range(arr.shape[1]):
                arr[y][x] = i
                i += 1
            j += 1
        
        _Utilities._create_new_tif(arr, 'test_img/se_gradient.tif', angle)

    def nw_gradient(self, angle=0):
        arr = np.empty([128, 129]).astype(np.uint8)

        i = j = 255
        for y in range(arr.shape[0]):
            i = j
            for x in range(arr.shape[1]):
                arr[y][x] = i
                i -= 1
            j -= 1
        
        _Utilities._create_new_tif(arr, 'test_img/nw_gradient.tif', angle)

    def s_gradient(self):
        arr = np.empty([256, 256]).astype(np.uint8)

        i = 0
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                arr[y][x] = i
            i += 1
        
        _Utilities._create_new_tif(arr, 'test_img/s_gradient.tif')

    def n_gradient(self):
        arr = np.empty([256, 256]).astype(np.uint8)

        i = 255
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                arr[y][x] = i
            i -= 1
        
        _Utilities._create_new_tif(arr, 'test_img/n_gradient.tif')

    def random(self, num_bands):
        arr = []
        for _ in range(num_bands):
            arr.append(np.random.random_integers(0,255, [256, 256]).astype(np.uint8))
        
        _Utilities._create_new_tif(arr, 'test_img/rand.tif')

    