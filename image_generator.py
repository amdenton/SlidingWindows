import rasterio
import math
import affine
import numpy as np
from windowagg.utilities import _Utilities
from windowagg.sliding_window import SlidingWindow
import numpy.random as rand
import os

class ImageGenerator:

    test_dir = 'test_img/'

    def __init__(self):
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def all(self, image_size, prefactor, sigma, mu, noise, angle=0, num_bands=4):
        self.gauss(image_size=image_size, prefactor=prefactor, sigma=sigma, mu=mu, noise=noise, angle=angle)
        self.gauss_x(image_size=image_size, prefactor=prefactor, sigma=sigma, mu=mu , noise=noise, angle=angle)
        self.cone(image_size=image_size, mu=mu, angle=angle)
        self.se_gradient(angle=angle)
        self.nw_gradient(angle=angle)
        self.s_gradient(angle=angle)
        self.n_gradient(angle=angle)
        self.random(num_bands=num_bands, angle=angle)

    def gauss(self, image_size, prefactor, sigma, mu, noise, angle=0):
        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = prefactor*(math.exp(-((x-mu)**2 + (y-mu)**2) / sigma**2) + noise*rand.normal())
                arr[y][x] = value

        arr = _Utilities._arr_dtype_conversion(arr, np.uint8)
        _Utilities._create_new_tif(arr, self.test_dir + 'gauss_image_no_noise.tif', angle)

    def gauss_x(self, image_size, prefactor, sigma, mu, noise, angle=0):
        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = prefactor*(math.exp(-((x-mu)**2) / sigma**2) + noise*rand.normal())
                arr[y][x] = value

        arr = _Utilities._arr_dtype_conversion(arr, np.uint8)
        _Utilities._create_new_tif(arr, self.test_dir + 'gauss_image_x_small.tif', angle)

    def cone(self, image_size, mu, angle=0):
        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = (1 - math.sqrt((x-mu)**2 + (y-mu)**2) / image_size)
                arr[y][x] = value

        arr = _Utilities._arr_dtype_conversion(arr, np.uint8)
        _Utilities._create_new_tif(arr, self.test_dir + 'cone_image.tif', angle)

    def se_gradient(self, angle=0):
        arr = np.empty([128, 129]).astype(np.uint8)

        i = j = 0
        for y in range(arr.shape[0]):
            i = j
            for x in range(arr.shape[1]):
                arr[y][x] = i
                i += 1
            j += 1
        
        _Utilities._create_new_tif(arr, self.test_dir + 'se_gradient.tif', angle)

    def nw_gradient(self, angle=0):
        arr = np.empty([128, 129]).astype(np.uint8)

        i = j = 255
        for y in range(arr.shape[0]):
            i = j
            for x in range(arr.shape[1]):
                arr[y][x] = i
                i -= 1
            j -= 1
        
        _Utilities._create_new_tif(arr, self.test_dir + 'nw_gradient.tif', angle)

    def s_gradient(self, angle=0):
        arr = np.empty([256, 256]).astype(np.uint8)

        i = 0
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                arr[y][x] = i
            i += 1
        
        _Utilities._create_new_tif(arr, self.test_dir + 's_gradient.tif', angle)

    def n_gradient(self, angle=0):
        arr = np.empty([256, 256]).astype(np.uint8)

        i = 255
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                arr[y][x] = i
            i -= 1
        
        _Utilities._create_new_tif(arr, self.test_dir + 'n_gradient.tif', angle)

    def random(self, num_bands=4, angle=0):
        arr = []
        for _ in range(num_bands):
            arr.append(np.random.random_integers(0,255, [256, 256]).astype(np.uint8))
        
        _Utilities._create_new_tif(arr, self.test_dir + 'rand.tif', angle)

    