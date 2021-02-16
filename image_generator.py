from windowagg.sliding_window import SlidingWindow
import windowagg.helper as helper

import os
import math

import rasterio
import affine
import numpy as np
import numpy.random as rand

_image_size = 512

class ImageGenerator:

    def __init__(self, path=None, dtype=None):
        if (path is None):
            path = 'img_gen/'
        if (dtype is None):
            dtype = np.uint16
        self.path = path
        self.dtype = dtype

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def all(self, image_size=_image_size, mu=None, sigma=None, noise=0, num_bands=4):
        self.star(image_size=image_size)
        self.gauss(image_size=image_size, sigma=sigma, mu=mu, noise=noise)
        self.gauss_horizontal(image_size=image_size, sigma=sigma, mu=mu, noise=noise)
        self.gauss_vertical(image_size=image_size, sigma=sigma, mu=mu, noise=noise)
        self.se_gradient(image_size=image_size)
        self.nw_gradient(image_size=image_size)
        self.s_gradient(image_size=image_size)
        self.n_gradient(image_size=image_size)
        self.random(image_size=image_size, num_bands=num_bands)

    def star(self, image_size=_image_size):
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
            
        fn = self.path + 'star.tif'
        helper.create_tif(helper.arr_dtype_conversion(arr, self.dtype), fn)
        return fn

    def gauss(self, image_size=_image_size, mu=None, sigma=None, noise=0):
        # Median offset
        if (mu is None):
            mu = image_size  / 2
        # Standard deviation
        if (sigma is None):
            sigma = image_size / 8

        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = (
                    (
                        1
                        /
                        (sigma * np.sqrt(2 * math.pi))
                    )
                    *
                    math.exp(
                        -(
                            ((x - mu)**2 + (y - mu)**2)
                            /
                            (2 * sigma**2)
                        )
                        +
                        noise*rand.normal()
                    )
                )
                arr[y][x] = value

        fn = self.path + 'gauss.tif'
        helper.create_tif(helper.arr_dtype_conversion(arr, self.dtype), fn)
        return fn

    def gauss_aspect_point(self, x, y, image_size=_image_size, mu=None, sigma=None):
        # Median offset
        if (mu is None):
            mu = image_size  / 2
        # Standard deviation
        if (sigma is None):
            sigma = image_size / 8

        dx = -(
            (
                np.exp(-(
                    ((mu - x)**2 + (mu - y)**2)
                    /
                    (2 * sigma**2)
                ))
                *
                (x - mu)
            )
            /
            (np.sqrt(2 * math.pi) * sigma**3)
        )

        dy = -(
            (
                np.exp(-(
                    ((mu - x)**2 + (mu - y)**2)
                    /
                    (2 * sigma**2)
                ))
                *
                (y - mu)
            )
            /
            (np.sqrt(2 * math.pi) * sigma**3)
        )

        aspect = (np.arctan2(-dy, -dx) + (math.pi / 2)) % (2 * math.pi)
        # TODO change this to something more appropriate for nodata
        if ((dx == 0) and (dy == 0)):
            aspect = 0

        return aspect

    def gauss_slope_point(self, x, y, image_size=_image_size, mu=None, sigma=None):
        # Median offset
        if (mu is None):
            mu = image_size  / 2
        # Standard deviation
        if (sigma is None):
            sigma = image_size / 8

        # Formula for the first derivative in the direction of steepest descent
        return -(
            math.sqrt(
                (
                    math.exp(-(
                        ((mu - x)**2 + (mu - y)**2)
                        /
                        sigma**2
                    ))
                    *
                    ((2 * mu**2) + x**2 + y**2 - (2 * mu * (x + y)))
                )
                /
                sigma**6
            )            
            /
            (math.sqrt( 2 * math.pi))
        )

    def gauss_standard_point(self, x, y, image_size=_image_size, mu=None, sigma=None):
        # Median offset
        if (mu is None):
            mu = image_size  / 2
        # Standard deviation
        if (sigma is None):
            sigma = image_size / 8
        
        # Standard curvature formula
        return (
            (
                math.exp(-(
                    ((2 * mu**2) + x**2 + y**2 - (2 * mu * (x + y)))
                    /
                    (2 * sigma**2)
                ))
                *
                ((2 * mu**2) - sigma**2 + x**2 + y**2 - (2 * mu * (x + y)))
            )
            /
            (2 * math.sqrt(2 * math.pi) * sigma**5)
        ) 

    def gauss_horizontal(self, image_size=_image_size, mu=None, sigma=None, noise=0):
        # Median offset
        if (mu is None):
            mu = image_size  / 2
        # Standard deviation
        if (sigma is None):
            sigma = image_size / 8

        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = (1 / (sigma * np.sqrt(2 * math.pi))) * math.exp(-((y - mu)**2 / (2 * sigma**2)) + (noise * rand.normal()))
                arr[y][x] = value

        fn = self.path + 'gauss_horizontal.tif'
        helper.create_tif(helper.arr_dtype_conversion(arr, self.dtype), fn)
        return fn

    def gauss_vertical(self, image_size=_image_size, mu=None, sigma=None, noise=0):
        # Median offset
        if (mu is None):
            mu = image_size  / 2
        # Standard deviation
        if (sigma is None):
            sigma = image_size / 8

        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = (1 / (sigma * np.sqrt(2 * math.pi))) * math.exp(-((x - mu)**2 / (2 * sigma**2)) + (noise * rand.normal()))
                arr[y][x] = value

        fn = self.path + 'gauss_vertical.tif'
        helper.create_tif(helper.arr_dtype_conversion(arr, self.dtype), fn)
        return fn

    def se_gradient(self, image_size=_image_size):
        arr = np.empty([image_size, image_size])

        i = j = 0
        for y in range(arr.shape[0]):
            i = j
            for x in range(arr.shape[1]):
                arr[y][x] = i
                i += 1
            j += 1
        
        fn = self.path + 'se_gradient.tif'
        helper.create_tif(helper.arr_dtype_conversion(arr, self.dtype), fn)
        return fn

    def nw_gradient(self, image_size=_image_size):
        arr = np.empty([image_size, image_size])

        i = j = np.iinfo(self.dtype).max
        for y in range(arr.shape[0]):
            i = j
            for x in range(arr.shape[1]):
                arr[y][x] = i
                i -= 1
            j -= 1
        
        fn = self.path + 'nw_gradient.tif'
        helper.create_tif(helper.arr_dtype_conversion(arr, self.dtype), fn)
        return fn

    def s_gradient(self, image_size=_image_size):
        arr = np.empty([image_size, image_size])

        i = 0
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                arr[y][x] = i
            i += 1
        
        fn = self.path + 's_gradient.tif'
        helper.create_tif(helper.arr_dtype_conversion(arr, self.dtype), fn)
        return fn

    def n_gradient(self, image_size=_image_size):
        arr = np.empty([image_size, image_size])

        i = np.iinfo(self.dtype).max
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                arr[y][x] = i
            i -= 1
        
        fn = self.path + 'n_gradient.tif'
        helper.create_tif(helper.arr_dtype_conversion(arr, self.dtype), fn)
        return fn

    def random(self, image_size=_image_size, num_bands=4):
        arr = []
        for _ in range(num_bands):
            rand_arr = np.random.random_sample(size=[image_size, image_size])
            rand_arr = helper.arr_dtype_conversion(rand_arr, self.dtype)
            arr.append(rand_arr)
        
        fn = self.path + 'rand.tif'
        helper.create_tif(arr, fn)
        return fn
