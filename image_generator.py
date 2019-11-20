import rasterio
import math
import affine
import numpy as np
from windowagg.utilities import _Utilities
from windowagg.sliding_window import SlidingWindow
import numpy.random as rand
import os

class ImageGenerator:

    __test_dir = 'test_img/'
    __dtype = np.uint16

    def __init__(self):
        if not os.path.exists(self.__test_dir):
            os.makedirs(self.__test_dir)

    def all(self, image_size=300, sigma=None, mu=None, noise=0, angle=0, num_bands=4, x_offset=1, y_offset=1):
        self.star(image_size=image_size, angle=angle)
        self.gauss(image_size=image_size, sigma=sigma, mu=mu, noise=noise, angle=angle, x_offset=x_offset, y_offset=x_offset)
        self.gauss_horizontal(image_size=image_size, sigma=sigma, mu=mu, noise=noise, angle=angle, x_offset=x_offset, y_offset=x_offset)
        self.gauss_vertical(image_size=image_size, sigma=sigma, mu=mu, noise=noise, angle=angle, x_offset=x_offset, y_offset=x_offset)
        self.se_gradient(image_size=image_size, angle=angle, x_offset=x_offset, y_offset=x_offset)
        self.nw_gradient(image_size=image_size, angle=angle, x_offset=x_offset, y_offset=x_offset)
        self.s_gradient(image_size=image_size, angle=angle, x_offset=x_offset, y_offset=x_offset)
        self.n_gradient(image_size=image_size, angle=angle, x_offset=x_offset, y_offset=x_offset)
        self.random(image_size=image_size, num_bands=num_bands, angle=angle, x_offset=x_offset, y_offset=x_offset)

    def star(self, image_size=300, angle=0, x_offset=1, y_offset=1):
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
            
        fn = self.__test_dir + 'star_' + str(angle) + 'skew_' + str(x_offset) + '-' + str(y_offset) + 'offset.tif'
        _Utilities._create_new_tif(arr.astype(self.__dtype), fn, angle, x_offset, y_offset)
        return fn

    def gauss(self, image_size=300, sigma=None, mu=None, noise=0, angle=0, x_offset=1, y_offset=1):
        # standard deviation
        if (sigma==None):
            sigma = image_size/4
        # median offset
        if (mu==None):
            mu = image_size/2

        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = (1/math.sqrt(2*math.pi*sigma**2))*math.exp(-(((x-mu)**2+(y-mu)**2) / (2*sigma**2)) + noise*rand.normal())
                arr[y][x] = value

        arr = _Utilities._arr_dtype_conversion(arr, self.__dtype)
        fn = self.__test_dir + 'gauss_' + str(angle) + 'skew_' + str(x_offset) + '-' + str(y_offset) + 'offset.tif'
        _Utilities._create_new_tif(arr, fn, angle, x_offset, y_offset)
        return fn

    def gauss_horizontal(self, image_size=300, sigma=None, mu=None, noise=0, angle=0, x_offset=1, y_offset=1):
        # standard deviation
        if (sigma==None):
            sigma = image_size/4
        # median offset
        if (mu==None):
            mu = image_size/2

        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = (1/math.sqrt(2*math.pi*sigma**2))*math.exp(-((y-mu)**2 / (2*sigma**2)) + noise*rand.normal())
                arr[y][x] = value

        arr = _Utilities._arr_dtype_conversion(arr, self.__dtype)
        fn = self.__test_dir + 'gauss_horizontal_' + str(angle) + 'skew_' + str(x_offset) + '-' + str(y_offset) + 'offset.tif'
        _Utilities._create_new_tif(arr, fn, angle, x_offset, y_offset)
        return fn

    def gauss_vertical(self, image_size=300, sigma=None, mu=None, noise=0, angle=0, x_offset=1, y_offset=1):
        # standard deviation
        if (sigma==None):
            sigma = image_size/4
        # median offset
        if (mu==None):
            mu = image_size/2

        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = (1/math.sqrt(2*math.pi*sigma**2))*math.exp(-((x-mu)**2 / (2*sigma**2)) + noise*rand.normal())
                arr[y][x] = value

        arr = _Utilities._arr_dtype_conversion(arr, self.__dtype)
        fn = self.__test_dir + 'gauss_vertical_' + str(angle) + 'skew_' + str(x_offset) + '-' + str(y_offset) + 'offset.tif'
        _Utilities._create_new_tif(arr, fn, angle, x_offset, y_offset)
        return fn

    def se_gradient(self, image_size=300, angle=0, x_offset=1, y_offset=1):
        arr = np.empty([image_size, image_size])

        i = j = 0
        for y in range(arr.shape[0]):
            i = j
            for x in range(arr.shape[1]):
                arr[y][x] = i
                i += 1
            j += 1
        
        fn = self.__test_dir + 'se_gradient_' + str(angle) + 'skew_' + str(x_offset) + '-' + str(y_offset) + 'offset.tif'
        _Utilities._create_new_tif(arr.astype(self.__dtype), fn, angle, x_offset, y_offset)
        return fn

    def nw_gradient(self, image_size=300, angle=0, x_offset=1, y_offset=1):
        arr = np.empty([image_size, image_size])

        i = j = np.iinfo(self.__dtype).max
        for y in range(arr.shape[0]):
            i = j
            for x in range(arr.shape[1]):
                arr[y][x] = i
                i -= 1
            j -= 1
        
        fn = self.__test_dir + 'nw_gradient_' + str(angle) + 'skew_' + str(x_offset) + '-' + str(y_offset) + 'offset.tif'
        _Utilities._create_new_tif(arr.astype(self.__dtype), fn, angle, x_offset, y_offset)
        return fn

    def s_gradient(self, image_size=300, angle=0, x_offset=1, y_offset=1):
        arr = np.empty([image_size, image_size])

        i = 0
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                arr[y][x] = i
            i += 1
        
        fn = self.__test_dir + 's_gradient_' + str(angle) + 'skew_' + str(x_offset) + '-' + str(y_offset) + 'offset.tif'
        _Utilities._create_new_tif(arr.astype(self.__dtype), fn, angle, x_offset, y_offset)
        return fn

    def n_gradient(self, image_size=300, angle=0, x_offset=1, y_offset=1):
        arr = np.empty([image_size, image_size])

        i = np.iinfo(self.__dtype).max
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                arr[y][x] = i
            i -= 1
        
        fn = self.__test_dir + 'n_gradient_' + str(angle) + 'skew_' + str(x_offset) + '-' + str(y_offset) + 'offset.tif'
        _Utilities._create_new_tif(arr.astype(self.__dtype), fn, angle, x_offset, y_offset)
        return fn

    def random(self, image_size=300, num_bands=4, angle=0, x_offset=1, y_offset=1):
        arr = []
        for _ in range(num_bands):
            arr.append(np.random.random_integers(0, np.iinfo(self.__dtype).max, [image_size, image_size]).astype(self.__dtype))
        
        fn = self.__test_dir + 'rand_' + str(angle) + 'skew_' + str(x_offset) + '-' + str(y_offset) + 'offset.tif'
        _Utilities._create_new_tif(arr, fn, angle, x_offset, y_offset)
        return fn

    