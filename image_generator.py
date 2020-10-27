from windowagg.sliding_window import SlidingWindow
import windowagg.helper as helper

import os
import math

import rasterio
import affine
import numpy as np
import numpy.random as rand

class ImageGenerator:

    test_dir = 'test_img/'
    dtype = np.uint16

    def __init__(self):
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def all(self, image_size=300, sigma=None, mu=None, noise=0, num_bands=4):
        self.star(image_size=image_size)
        self.gauss(image_size=image_size, sigma=sigma, mu=mu, noise=noise)
        self.gauss_horizontal(image_size=image_size, sigma=sigma, mu=mu, noise=noise)
        self.gauss_vertical(image_size=image_size, sigma=sigma, mu=mu, noise=noise)
        self.se_gradient(image_size=image_size)
        self.nw_gradient(image_size=image_size)
        self.s_gradient(image_size=image_size)
        self.n_gradient(image_size=image_size)
        self.random(image_size=image_size, num_bands=num_bands)

    def star(self, image_size=300):
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
            
        fn = self.test_dir + 'star.tif'
        helper.create_tif(arr.astype(self.dtype), fn)
        return fn

    def gauss(self, image_size=300, sigma=None, mu=None, noise=0):
        # standard deviation
        if (sigma==None):
            sigma = image_size/4
        # median offset
        if (mu==None):
            mu = image_size/2

        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = (1 / 2 * math.pi * sigma**2) * math.exp(-(((x - mu)**2 + (y - mu)**2) / (2 * sigma**2)) + noise*rand.normal())
                arr[y][x] = value

        arr = helper.arr_dtype_conversion(arr, self.dtype)
        fn = self.test_dir + 'gauss.tif'
        helper.create_tif(arr, fn)
        return fn

    def gauss_horizontal(self, image_size=300, sigma=None, mu=None, noise=0):
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

        arr = helper.arr_dtype_conversion(arr, self.dtype)
        fn = self.test_dir + 'gauss_horizontal.tif'
        helper.create_tif(arr, fn)
        return fn

    def gauss_vertical(self, image_size=300, sigma=None, mu=None, noise=0):
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

        arr = helper.arr_dtype_conversion(arr, self.dtype)
        fn = self.test_dir + 'gauss_vertical.tif'
        helper.create_tif(arr, fn)
        return fn

    def se_gradient(self, image_size=300):
        arr = np.empty([image_size, image_size])

        i = j = 0
        for y in range(arr.shape[0]):
            i = j
            for x in range(arr.shape[1]):
                arr[y][x] = i
                i += 1
            j += 1
        
        fn = self.test_dir + 'se_gradient.tif'
        helper.create_tif(arr.astype(self.dtype), fn)
        return fn

    def nw_gradient(self, image_size=300):
        arr = np.empty([image_size, image_size])

        i = j = np.iinfo(self.dtype).max
        for y in range(arr.shape[0]):
            i = j
            for x in range(arr.shape[1]):
                arr[y][x] = i
                i -= 1
            j -= 1
        
        fn = self.test_dir + 'nw_gradient.tif'
        helper.create_tif(arr.astype(self.dtype), fn)
        return fn

    def s_gradient(self, image_size=300):
        arr = np.empty([image_size, image_size])

        i = 0
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                arr[y][x] = i
            i += 1
        
        fn = self.test_dir + 's_gradient.tif'
        helper.create_tif(arr.astype(self.dtype), fn)
        return fn

    def n_gradient(self, image_size=300):
        arr = np.empty([image_size, image_size])

        i = np.iinfo(self.dtype).max
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                arr[y][x] = i
            i -= 1
        
        fn = self.test_dir + 'n_gradient.tif'
        helper.create_tif(arr.astype(self.dtype), fn)
        return fn

    def random(self, image_size=300, num_bands=4):
        arr = []
        for _ in range(num_bands):
            arr.append(np.random.randint(0, np.iinfo(self.dtype).max, [image_size, image_size]).astype(self.dtype))
        
        fn = self.test_dir + 'rand.tif'
        helper.create_tif(arr, fn)
        return fn
