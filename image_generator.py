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

    def all(self, image_size, sigma, noise=0, angle=0, num_bands=4):
        star(self, image_size=image_size, angle=0):
        self.gauss(image_size=image_size, sigma=sigma, noise=noise, angle=angle)
        self.gauss_horizontal(image_size=image_size, sigma=sigma, noise=noise, angle=angle)
        self.gauss_vertical(image_size=image_size, sigma=sigma, noise=noise, angle=angle)
        self.se_gradient(angle=angle)
        self.nw_gradient(angle=angle)
        self.s_gradient(angle=angle)
        self.n_gradient(angle=angle)
        self.random(num_bands=num_bands, angle=angle)

    def star(self, image_size=21, angle=0):
        arr = np.zeros([image_size,image_size])

        i=j=1
        decx = decy = False
        for y in range(image_size):
            i=j
            for x in range(image_size):
                arr[y][x] = i
                if (decx):
                    i -= 1
                elif (i == j+math.ceil(image_size/2)-1):
                    decx = True
                    if (image_size%2 == 1):
                        i -= 1
                else:
                    i += 1
            decx = False
            if (decy):
                j -= 1
            elif (j == math.ceil(image_size/2)):
                decy = True
                if (image_size%2 == 1):
                        j -= 1
            else:
                j += 1
            
        arr = _Utilities._arr_dtype_conversion(arr, np.uint8)
        _Utilities._create_new_tif(arr, self.test_dir + 'star_' + str(angle) + 'skew.tif', angle)

    def gauss(self, image_size, sigma, noise=0, angle=0):
        arr = np.empty([image_size, image_size])
        mu = image_size/2

        for y in range (image_size):
            for x in range (image_size):
                value = (1/math.sqrt(2*math.pi*sigma**2))*math.exp(-(((x-mu)**2+(y-mu)**2) / (2*sigma**2)) + noise*rand.normal())
                arr[y][x] = value

        arr = _Utilities._arr_dtype_conversion(arr, np.uint8)
        _Utilities._create_new_tif(arr, self.test_dir + 'gauss_' + str(angle) + 'skew.tif', angle)

    def gauss_horizontal(self, image_size, sigma, noise=0, angle=0):
        arr = np.empty([image_size, image_size])
        mu = image_size/2

        for y in range (image_size):
            for x in range (image_size):
                value = (1/math.sqrt(2*math.pi*sigma**2))*math.exp(-((y-mu)**2 / (2*sigma**2)) + noise*rand.normal())
                arr[y][x] = value

        arr = _Utilities._arr_dtype_conversion(arr, np.uint8)
        _Utilities._create_new_tif(arr, self.test_dir + 'gauss_horizontal_' + str(angle) + 'skew.tif', angle)

    def gauss_vertical(self, image_size, sigma, noise=0, angle=0):
        arr = np.empty([image_size, image_size])
        mu = image_size/2

        for y in range (image_size):
            for x in range (image_size):
                value = (1/math.sqrt(2*math.pi*sigma**2))*math.exp(-((x-mu)**2 / (2*sigma**2)) + noise*rand.normal())
                arr[y][x] = value

        arr = _Utilities._arr_dtype_conversion(arr, np.uint8)
        _Utilities._create_new_tif(arr, self.test_dir + 'gauss_vertical_' + str(angle) + 'skew.tif', angle)

    def se_gradient(self, angle=0):
        arr = np.empty([128, 129]).astype(np.uint8)

        i = j = 0
        for y in range(arr.shape[0]):
            i = j
            for x in range(arr.shape[1]):
                arr[y][x] = i
                i += 1
            j += 1
        
        _Utilities._create_new_tif(arr, self.test_dir + 'se_gradient_' + str(angle) + 'skew.tif', angle)

    def nw_gradient(self, angle=0):
        arr = np.empty([128, 129]).astype(np.uint8)

        i = j = 255
        for y in range(arr.shape[0]):
            i = j
            for x in range(arr.shape[1]):
                arr[y][x] = i
                i -= 1
            j -= 1
        
        _Utilities._create_new_tif(arr, self.test_dir + 'nw_gradient_' + str(angle) + 'skew.tif', angle)

    def s_gradient(self, angle=0):
        arr = np.empty([256, 256]).astype(np.uint8)

        i = 0
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                arr[y][x] = i
            i += 1
        
        _Utilities._create_new_tif(arr, self.test_dir + 's_gradient_' + str(angle) + 'skew.tif', angle)

    def n_gradient(self, angle=0):
        arr = np.empty([256, 256]).astype(np.uint8)

        i = 255
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                arr[y][x] = i
            i -= 1
        
        _Utilities._create_new_tif(arr, self.test_dir + 'n_gradient_' + str(angle) + 'skew.tif', angle)

    def random(self, num_bands=4, angle=0):
        arr = []
        for _ in range(num_bands):
            arr.append(np.random.random_integers(0,255, [256, 256]).astype(np.uint8))
        
        _Utilities._create_new_tif(arr, self.test_dir + 'rand_' + str(angle) + 'skew.tif', angle)

    