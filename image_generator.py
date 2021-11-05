from windowagg.sliding_window import SlidingWindow
import windowagg.helper as helper

from osgeo import gdal
from osgeo import osr
import os
import math

import rasterio
import affine
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import windowagg.rbg as rbg

_image_size = 512

class ImageGenerator:

    def __init__(self, noise_true = True, path=None, dtype=None):
        self.noise_true = noise_true
        if (path is None):
#            path = 'img_gen/'
            path = 'img_dir/'
        if (dtype is None):
            dtype = np.uint16
        self.path = path
        self.dtype = dtype
        
        self.noise = 0#0.01#0.02#0.05 #0
        self.multi_random_gauss_geotiff_fn = self.path+'multi_random_gauss_geotiff.tif'
        self.multi_random_gauss_fn = self.path + 'multi_random_gauss.tif'

        if self.noise_true:
            self.noise = 0.01#0.02#0.05 #0
            self.multi_random_gauss_geotiff_fn = self.path+'multi_random_gauss_n001_geotiff.tif'
            self.multi_random_gauss_fn = self.path + 'multi_random_gauss_n001.tif'

        self.sigma_min = 0.1
        self.sigma_spread = 0.2
        self.random_set_size = 10
        self.random_set = np.empty((3,self.random_set_size))
        self.height_per_peak = 1

        rand.seed(1)
        for i in range(self.random_set_size):
            self.random_set[0,i] = rand.random()
            self.random_set[1,i] = rand.random()
            self.random_set[2,i] = rand.random()
            #self.random_set[3,i] = rand.random()

            print('Random point '+str(i)+' x = '+str(self.random_set[0,i])+' y = '+str(self.random_set[1,i])+' sigma = '+str(self.random_set[2,i]))

        for i in range(self.random_set_size):
            print(str(self.random_set[0,i])+'& '+str(self.random_set[1,i])+'& '+str((self.sigma_min+self.sigma_spread*self.random_set[2,i]))+'//')
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)
    def landscape (self,fn):
        return fn+'.tif'

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
                        noise*rand.normal() # should this be added to the exponent?
                    )
                )
                arr[y][x] = value

        fn = self.path + 'gauss.tif'
        helper.create_tif(arr,fn)
        #helper.create_tif(helper.arr_dtype_conversion(arr, self.dtype), fn)
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

    def parabola(self, image_size=_image_size, noise=0):
        # Median offset
        #if (mu is None):
        mu = image_size  / 2
        # Standard deviation
        #if (sigma is None):
        sigma = image_size / 8

        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = -((y - mu)**2 / sigma**2) + (noise * rand.normal())
                arr[y][x] = value

        fn = self.path + 'parabola.tif'
        helper.create_tif(arr, fn)
        return fn
    
    def parabola_arr(self, kind=0, image_size=_image_size, aggs=0):
        # Median offset
        mu = image_size  / 2
        # Standard deviation
        sigma = image_size / 8

        arr = np.empty([image_size, image_size])
        if kind == 0:
            for y in range (image_size):
                for x in range (image_size):
                    value = -((y - mu)**2 / sigma**2) 
                    arr[y][x] = value
        elif kind == 1:
            for y in range (image_size):
                for x in range (image_size):
                    value = (2*abs(y - mu) / sigma**2) 
                    arr[y][x] = value
        elif kind == 2:
            for y in range (image_size):
                for x in range (image_size):
                    value = -(1 / sigma**2) 
                    arr[y][x] = value

        return arr
    
    def random(self, image_size=_image_size, num_bands=4):
        arr = []
        for _ in range(num_bands):
            rand_arr = np.random.random_sample(size=[image_size, image_size])
            rand_arr = helper.arr_dtype_conversion(rand_arr, self.dtype)
            arr.append(rand_arr)
        
        fn = self.path + 'rand.tif'
        helper.create_tif(arr, fn)
        return fn
    
#    def multi_random_gauss(self, image_size=_image_size, noise=0):
#        arr = np.empty([image_size, image_size])
#        
#        for y in range (image_size):
#            for x in range (image_size):
#                value = 0
#                for peak in range(self.random_set_size):
#                    mu = self.random_set[0,peak] * image_size
#                    nu = self.random_set[1,peak] * image_size
#                    sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * image_size
#                    value += math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2))
#                    + noise*rand.normal())
#                arr[x][y] = value
#                    
#        fn = self.path + 'multi_random_gauss.tif'
#        helper.create_tif(self.height_per_peak*arr,fn)
        #helper.create_tif(helper.arr_dtype_conversion(arr, self.dtype), fn)
#        return fn
             
    def multi_random_gauss_geotiff(self, image_size=_image_size):
        fn = self.multi_random_gauss_geotiff_fn
        arr = np.zeros([image_size, image_size])
        arr0 = np.zeros([image_size, image_size])
        arr1 = np.zeros([image_size, image_size])
        arr2 = np.zeros([image_size, image_size])
        arr3 = np.zeros([image_size, image_size])
        arr4 = np.zeros([image_size, image_size])
        arr5 = np.zeros([image_size, image_size])
        arr6 = np.zeros([image_size, image_size])
        arr7 = np.zeros([image_size, image_size])
        arr8 = np.zeros([image_size, image_size])
        arr9 = np.zeros([image_size, image_size])
#        original_size = image_size + math.floor(2**aggs - 1)
#        border = math.floor((2**aggs - 1) / 2)
        
        for y in range (image_size):
            for x in range (image_size):
                value = 0
                values = np.empty(10)
                for peak in range(self.random_set_size):
                    mu = self.random_set[0,peak] * image_size
                    nu = self.random_set[1,peak] * image_size
                    sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * image_size
                    epsilon = self.noise * rand.random()
                    value += math.exp(-((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)) + epsilon
                    values[peak] = math.exp(-((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)) + epsilon
                arr[x,y] = value
                arr0[x,y] = values[0]
                arr1[x,y] = values[1]
                arr2[x,y] = values[2]
                arr3[x,y] = values[3]
                arr4[x,y] = values[4]
                arr5[x,y] = values[5]
                arr6[x,y] = values[6]
                arr7[x,y] = values[7]
                arr8[x,y] = values[8]
                arr9[x,y] = values[9]
        
        
        plt.figure()
        plt.imshow(arr0)
        plt.figure()
        plt.imshow(arr1)
        plt.figure()
        plt.imshow(arr2)
        plt.figure()
        plt.imshow(arr3)
        plt.figure()
        plt.imshow(arr4)
        plt.figure()
        plt.imshow(arr5)
        plt.figure()
        plt.imshow(arr6)
        plt.figure()
        plt.imshow(arr7)
        plt.figure()
        plt.imshow(arr8)
        plt.figure()
        plt.imshow(arr9)

#        geotransform = (500000,1,0,5000000, 0,-1)
#        dst_ds = gdal.GetDriverByName('GTiff').Create(fn, image_size, image_size, 1, gdal.GDT_Float32)
#        dst_ds.SetGeoTransform(geotransform)    # specify coords
#        srs = osr.SpatialReference()            # establish encoding
#        srs.ImportFromEPSG(32614)                
#        dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
#        dst_ds.GetRasterBand(1).WriteArray(arr)   
#        dst_ds.FlushCache()                     # write to disk
        return fn

    #doesn't work
    def multi_random_gauss_av_geotiff(self, image_size=_image_size,agg=4): 
        fn = self.path+'multi_random_gauss_n01_av_geotiff.tif'
#        original_size = image_size + math.floor(2**aggs - 1)
        w = 2**agg
        factor = 1. / w**2
        border = math.floor((w - 1) / 2)
        print('border = ',border)
        arr = np.zeros([image_size+w-1, image_size+w-1])
        
        for y in range (image_size):
            for x in range (image_size):
                value = 0
                for peak in range(self.random_set_size):
                    mu = self.random_set[0,peak] * image_size
                    nu = self.random_set[1,peak] * image_size
                    sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * image_size
                    epsilon = self.noise * rand.random()
                    value += math.exp(-((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)) + epsilon
                arr[x,y] = value
                
        arr = factor*rbg.mean(arr,agg)
        print(arr)
        new_arr = np.zeros([image_size, image_size])
        
        for y in range(image_size):
            for x in range(image_size):
                new_arr[x,y] = arr[x,y]
        print(new_arr)

        geotransform = (500000,1,0,5000000, 0,-1)
        dst_ds = gdal.GetDriverByName('GTiff').Create(fn, image_size, image_size, 1, gdal.GDT_Float32)
        dst_ds.SetGeoTransform(geotransform)    # specify coords
        srs = osr.SpatialReference()            # establish encoding
        srs.ImportFromEPSG(32614)                
        dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
        dst_ds.GetRasterBand(1).WriteArray(new_arr)   
        dst_ds.FlushCache()                     # write to disk
        return fn

    def multi_random_gauss_bare(self, image_size=_image_size):
        fn = self.multi_random_gauss_fn 
        arr = np.zeros([image_size, image_size])
#        original_size = image_size + math.floor(2**aggs - 1)
#        border = math.floor((2**aggs - 1) / 2)
        
        for y in range (image_size):
            for x in range (image_size):
                value = 0
                for peak in range(self.random_set_size):
                    mu = self.random_set[0,peak] * image_size
                    nu = self.random_set[1,peak] * image_size
                    sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * image_size
                    epsilon = self.noise * rand.random()
                    value += math.exp(-((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)) + epsilon
                arr[x,y] = value
        helper.create_tif(self.height_per_peak*arr,fn)
        return fn

    def multi_random_gauss(self, image_size=_image_size):
        arr = np.zeros([image_size, image_size])
#        original_size = image_size + math.floor(2**aggs - 1)
#        border = math.floor((2**aggs - 1) / 2)
        fn = self.multi_random_gauss_fn         
        for y in range (image_size):
            for x in range (image_size):
                value = 0
                for peak in range(self.random_set_size):
                    mu = self.random_set[0,peak] * image_size
                    nu = self.random_set[1,peak] * image_size
                    sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * image_size
                    epsilon = self.noise * rand.random()
                    value += math.exp(-((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)) + epsilon
                arr[x,y] = value

        helper.create_tif(self.height_per_peak*arr,fn)
# Derivative
        for y in range (image_size):
            for x in range (image_size):
                fx = 0
                fy = 0
                for peak in range(self.random_set_size):
                    mu = self.random_set[0,peak] * image_size
                    nu = self.random_set[1,peak] * image_size
                    sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * image_size
                    fx += (mu-x)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fy += (nu-y)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        
                arr[x,y] = math.atan(math.sqrt(fx**2 + fy**2))*180/math.pi
                
        fn1 = self.path + 'multi_random_gauss_slope_analytical.tif'
        helper.create_tif(self.height_per_peak*arr,fn1)

# Aspect
        for y in range (image_size):
            for x in range (image_size):
                fx = 0
                fy = 0
                for peak in range(self.random_set_size):
                    mu = self.random_set[0,peak] * image_size
                    nu = self.random_set[1,peak] * image_size
                    sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * image_size
                    fx += (mu-x)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fy += (nu-y)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        
                if fx !=0 or fy !=0:
                    arr[x,y] = (math.atan(-fy/fx) + (math.pi / 2))
                    while arr[x,y] > (2 * math.pi):
                        arr[x,y] -= 2 * math.pi
                else:
                    arr[x,y] = 0
        print('Just generating Aspect')
        print(arr)

        fn5 = self.path + 'multi_random_gauss_aspect_analytical.tif'
        helper.create_tif(self.height_per_peak*arr,fn5)        
     
# Standard curvature
        for y in range (image_size):
            for x in range (image_size):
                fxx = 0
                fyy = 0
                fxy = 0
                for peak in range(self.random_set_size):
                    #print('original size: ',(image_size+total_border))
                    mu = self.random_set[0,peak] * image_size
                    nu = self.random_set[1,peak] * image_size
                    sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * image_size
                    fxx += ((x-mu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fyy += ((y-nu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fxy += ((x-mu)*(y-nu) - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    
                arr[x,y] = 0.5*(fxx + fyy)
        fn2 = self.path + 'multi_random_gauss_standard_analytical.tif'
        helper.create_tif(self.height_per_peak*arr,fn2)
         
# Profile curvature
        for y in range (image_size):
            for x in range (image_size):
                fx = 0
                fy = 0
                fxx = 0
                fyy = 0
                fxy = 0
                for peak in range(self.random_set_size):
                    #print('original size: ',(image_size+total_border))
                    mu = self.random_set[0,peak] * image_size
                    nu = self.random_set[1,peak] * image_size
                    sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * image_size
                    fx += (mu-x)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fy += (nu-y)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fxx += ((x-mu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fyy += ((y-nu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fxy += (x-mu)*(y-nu)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))

#                fsquared = fx**2 + fy**2
                denom = (fx**2 + fy**2)*math.sqrt(1+fx**2+fy**2)**3
                arr[x,y] = -(fxx * fx**2 + 2 * fxy * fx*fy + fyy * fy**2) / denom
                #arr[x,y] = fx*fy
        fn3 = self.path + 'multi_random_gauss_profile_analytical.tif'
        helper.create_tif(self.height_per_peak*arr,fn3)

# Planform curvature                    
        for y in range (image_size):
            for x in range (image_size):
                fx = 0
                fy = 0
                fxx = 0
                fyy = 0
                fxy = 0
                for peak in range(self.random_set_size):
                    #print('original size: ',(image_size+total_border))
                    mu = self.random_set[0,peak] * image_size
                    nu = self.random_set[1,peak] * image_size
                    sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * image_size
                    fx += (mu-x)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fy += (nu-y)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fxx += ((x-mu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fyy += ((y-nu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fxy += (x-mu)*(y-nu)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))

                #fsquared = fx**2 + fy**2
                denom = math.sqrt(fx**2+fy**2)**3
                arr[x,y] = -(fxx * fy**2 - 2 * fxy * fx*fy + fyy * fx**2) / denom
                    #arr[x,y] = fx*fy
        fn4 = self.path + 'multi_random_gauss_planform_analytical.tif'
        helper.create_tif(self.height_per_peak*arr,fn4)

# Horizontal curvature                    
        for y in range (image_size):
            for x in range (image_size):
                fx = 0
                fy = 0
                fxx = 0
                fyy = 0
                fxy = 0
                for peak in range(self.random_set_size):
                    #print('original size: ',(image_size+total_border))
                    mu = self.random_set[0,peak] * image_size
                    nu = self.random_set[1,peak] * image_size
                    sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * image_size
                    fx += (mu-x)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fy += (nu-y)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fxx += ((x-mu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fyy += ((y-nu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    fxy += (x-mu)*(y-nu)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))

                #fsquared = fx**2 + fy**2
                denom = (fx**2+fy**2)*math.sqrt(1+fx**2+fy**2)
                arr[x,y] = -(fxx * fy**2 - 2 * fxy * fx*fy + fyy * fx**2) / denom
                    #arr[x,y] = fx*fy
        fn5 = self.path + 'multi_random_gauss_horizontal_analytical.tif'
        helper.create_tif(self.height_per_peak*arr,fn5)

        return fn

    def multi_random_gauss_arr(self, kind=0, image_size=_image_size, aggs=0):
        arr = np.zeros([image_size, image_size])
        original_size = image_size + math.floor(2**aggs - 1)
#        border = math.floor((2**aggs - 1) / 2)
        w = 2**aggs
        border = (2**aggs - 1) / 2
        print('aggs: ',aggs)
        print('original size: ',original_size)
        print('border: ',border)
        
        if kind == 0:
            for y in range (image_size):
                for x in range (image_size):
                    value = 0
                    for peak in range(self.random_set_size):
                        mu = self.random_set[0,peak] * original_size - border
                        nu = self.random_set[1,peak] * original_size - border
                        sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * original_size
#                        value += math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2))
#                            + noise*rand.normal())
                        value += math.exp(-((x - mu)**2 + (y - nu)**2)/(2 * sigma**2))
                    arr[x,y] = value
#            fn = self.path + 'multi_random_gauss.tif'
#            helper.create_tif(self.height_per_peak*arr,fn)
# Derivative
        elif kind == 1:
            for y in range (image_size):
                for x in range (image_size):
                    fx = 0
                    fy = 0
                    for peak in range(self.random_set_size):
                        mu = self.random_set[0,peak] * original_size - border
                        nu = self.random_set[1,peak] * original_size - border
                        sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * original_size
                        fx += (mu-x)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fy += (nu-y)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        
                    #arr[x,y] = math.sqrt(fx**2 + fy**2) 
                    arr[x,y] = math.atan(math.sqrt(fx**2 + fy**2))*180/math.pi
            fn = self.path + 'multi_random_gauss_slope_analytical_w='+str(w)+'.tif'
            helper.create_tif(self.height_per_peak*arr,fn)
# Standard curvature
        elif kind == 2:
            for y in range (image_size):
                for x in range (image_size):
                    fxx = 0
                    fyy = 0
                    fxy = 0
                    for peak in range(self.random_set_size):
                        #print('original size: ',(image_size+total_border))
                        mu = self.random_set[0,peak] * original_size - border
                        nu = self.random_set[1,peak] * original_size - border
                        sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * original_size
                        fxx += ((x-mu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fyy += ((y-nu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fxy += ((x-mu)*(y-nu) - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                    
                    arr[x,y] = 0.5*(fxx + fyy)
            fn = self.path + 'multi_random_gauss_standard_analytical_w='+str(w)+'.tif'
            helper.create_tif(self.height_per_peak*arr,fn)
         
# Profile curvature
        elif kind == 3:
            for y in range (image_size):
                for x in range (image_size):
                    fx = 0
                    fy = 0
                    fxx = 0
                    fyy = 0
                    fxy = 0
                    for peak in range(self.random_set_size):
                        #print('original size: ',(image_size+total_border))
                        mu = self.random_set[0,peak] * original_size - border
                        nu = self.random_set[1,peak] * original_size - border
                        sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * original_size
                        fx += (mu-x)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fy += (nu-y)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fxx += ((x-mu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fyy += ((y-nu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fxy += (x-mu)*(y-nu)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))

                    #fsquared = fx**2 + fy**2
                    denom = (fx**2 + fy**2)*math.sqrt(1+fx**2+fy**2)**3
                    #arr[x,y] = (fxx * 0.25 + 2 * fxy * 0.25 + fyy * 0.25) / 0.5
                    #arr[x,y] = (fxx * fx**2 + fyy * fy**2) / fsquared
                    arr[x,y] = -(fxx * fx**2 + 2 * fxy * fx*fy + fyy * fy**2) / denom
                    #arr[x,y] = fx*fy
            fn = self.path + 'multi_random_gauss_profile_analytical_w='+str(w)+'.tif'
            helper.create_tif(self.height_per_peak*arr,fn)

# Planform curvature                    
        elif kind == 4:
            for y in range (image_size):
                for x in range (image_size):
                    fx = 0
                    fy = 0
                    fxx = 0
                    fyy = 0
                    fxy = 0
                    for peak in range(self.random_set_size):
                        #print('original size: ',(image_size+total_border))
                        mu = self.random_set[0,peak] * original_size - border
                        nu = self.random_set[1,peak] * original_size - border
                        sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * original_size
                        fx += (mu-x)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fy += (nu-y)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fxx += ((x-mu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fyy += ((y-nu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fxy += (x-mu)*(y-nu)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))

                    #fsquared = fx**2 + fy**2
                    denom = math.sqrt(fx**2+fy**2)**3
                    #denom = (fx**2+fy**2)*math.sqrt(1+fx**2+fy**2)
                    #arr[x,y] = (fxx * 0.25 + 2 * fxy * 0.25 + fyy * 0.25) / 0.5
                    #arr[x,y] = (fxx * fx**2 + fyy * fy**2) / fsquared
                    arr[x,y] = -(fxx * fy**2 - 2 * fxy * fx*fy + fyy * fx**2) / denom
                    #arr[x,y] = fx*fy
                    
#            helper.create_tif(arr,'img/test.tif')
#            plt.figure()
#            plt.imshow(arr)
#            plt.savefig('img/test.png')
            fn = self.path + 'multi_random_gauss_planform_analytical_w='+str(w)+'.tif'
            helper.create_tif(self.height_per_peak*arr,fn)
        elif kind == 5:
            for y in range (image_size):
                for x in range (image_size):
                    fx = 0
                    fy = 0
                    for peak in range(self.random_set_size):
                        mu = self.random_set[0,peak] * image_size
                        nu = self.random_set[1,peak] * image_size
                        sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * image_size
                        fx += (mu-x)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fy += (nu-y)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        
                    if fx !=0 or fy !=0:
                        arr[x,y] = (np.arctan2(-fy, -fx) + (math.pi / 2)) 
                    else:
                        arr[x,y] = 0
                    while arr[x,y] > 2*math.pi:
                        arr[x,y] -= 2*math.pi
                        
                        
# Planform curvature                    
        elif kind == 6:
            for y in range (image_size):
                for x in range (image_size):
                    fx = 0
                    fy = 0
                    fxx = 0
                    fyy = 0
                    fxy = 0
                    for peak in range(self.random_set_size):
                        #print('original size: ',(image_size+total_border))
                        mu = self.random_set[0,peak] * original_size - border
                        nu = self.random_set[1,peak] * original_size - border
                        sigma = (self.sigma_min + self.sigma_spread * self.random_set[2,peak]) * original_size
                        fx += (mu-x)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fy += (nu-y)/sigma**2 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fxx += ((x-mu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fyy += ((y-nu)**2 - sigma**2)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))
                        fxy += (x-mu)*(y-nu)/sigma**4 * math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2)))

                    #fsquared = fx**2 + fy**2
                    #denom = math.sqrt(fx**2+fy**2)**3
                    denom = (fx**2+fy**2)*math.sqrt(1+fx**2+fy**2)
                    #arr[x,y] = (fxx * 0.25 + 2 * fxy * 0.25 + fyy * 0.25) / 0.5
                    #arr[x,y] = (fxx * fx**2 + fyy * fy**2) / fsquared
                    arr[x,y] = -(fxx * fy**2 - 2 * fxy * fx*fy + fyy * fx**2) / denom
                    #arr[x,y] = fx*fy
                    
#            helper.create_tif(arr,'img/test.tif')
#            plt.figure()
#            plt.imshow(arr)
#            plt.savefig('img/test.png')
            fn = self.path + 'multi_random_gauss_horizontal_analytical_w='+str(w)+'.tif'
            helper.create_tif(self.height_per_peak*arr,fn)


        return self.height_per_peak * arr

    def noisy_line(self, image_size=_image_size, slope=0.5):
        arr = np.zeros([image_size, image_size])

        noise = 2
        x0 = noise
        y0 = noise
        x1 = image_size-noise
        y1 = slope * image_size
        print('image_size: '+str(image_size))

        no_points = image_size*2
        step = 1/no_points
        white = np.iinfo(self.dtype).max
        print(white)
        for i in range (0, no_points):
            x_loc = x0+i*step*(x1-x0) + rand.normal(0,noise)
            y_loc = y0+i*step*(y1-y0) + rand.normal(0,noise)
            i_loc = min(image_size-1,max(0,int(x_loc)))
            j_loc = min(image_size-1,max(0,int(y_loc)))
            arr[j_loc,i_loc] = 1#white
        
        fn = self.path + 'noisy_line.tif'
        helper.create_tif(arr,fn)
        #helper.create_tif(helper.arr_dtype_conversion(arr, self.dtype), fn)
        return fn
