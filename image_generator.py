"""
Last updated on Tue Dec 14

@authors: Anne Denton, David Schwarz, Rahul Gomes

License information:
https://opensource.org/licenses/GPL-3.0
"""
import windowagg.helper as helper

import os
import math

import numpy as np
import numpy.random as rand

_image_size = 512

class ImageGenerator:

    def __init__(self, path=None, dtype=None):
        if (path is None):
            path = 'img/'
        if (dtype is None):
            dtype = np.uint16
        self.path = path
        self.dtype = dtype
        self.auto_plot = False
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            
    def landscape (self,fn):
        return fn
    
    def dome (self, image_size=_image_size, mu=None, sigma=None):
        # Median offset
        if (mu is None):
            mu = image_size  / 2
        # Standard deviation
        if (sigma is None):
            sigma = image_size

        arr = np.empty([image_size, image_size])

        for y in range (image_size):
            for x in range (image_size):
                value = math.sqrt(sigma**2 - (x-mu)**2 - (y-mu)**2)
                arr[y,x] = value

        fn = self.path + 'dome.tif'
        helper.create_tif(arr,fn)
        return fn
        
    def multi_gauss(self, image_size=_image_size, experiment_id=0, noise=0.01):    
        if experiment_id == 0:
            height_per_peak = 100
            sigma_min = 0.1
            sigma_spread = 0.2
            random_set_size = 10
            fn_trunc = 'multi_random'+'_id'+str(int(experiment_id))
        elif experiment_id == 1:
            height_per_peak = 10
            sigma_min = 0.1
            sigma_spread = 0.2
            random_set_size = 20
            fn_trunc = 'multi_random'+'_id'+str(int(experiment_id))
        else:
            experiment_id = 0
            height_per_peak = 100
            sigma_min = 0.1
            sigma_spread = 0.2
            random_set_size = 10
            fn_trunc = 'multi_random'+'_id'+str(int(experiment_id))
            
        if noise > 0:
            fn_trunc = fn_trunc+'_n'+str(int(round(noise*1000)))
        else:
            fn_trunc = fn_trunc+'_nn'

        random_set = np.empty((3,random_set_size))
        rand.seed(1)
        for i in range(random_set_size):
            random_set[0,i] = rand.random()
            random_set[1,i] = rand.random()
            random_set[2,i] = rand.random()

            print('Random point '+str(i)+' x = '+str(random_set[0,i])+' y = '+str(random_set[1,i])+' sigma factor = '+str(random_set[2,i]))
        
        arr = np.zeros([image_size, image_size])
        for y in range (image_size):
            for x in range (image_size):
                for peak in range(random_set_size):
                    mu = random_set[0,peak] * image_size
                    nu = random_set[1,peak] * image_size
                    sigma = (sigma_min + sigma_spread * random_set[2,peak]) * image_size
                    epsilon = noise * rand.random()
                    arr[x,y] += height_per_peak * (math.exp(-(((x - mu)**2 + (y - nu)**2)/(2 * sigma**2))) + epsilon)

# Original image
        fn = self.path + fn_trunc + '.tif'
        helper.create_tif(arr,fn)
        if (self.auto_plot):
            helper.plot(fn)

# Derivatives are shifted by half a pixel to make them comparable to the numerical results
        fx_array = np.zeros([image_size, image_size])
        fy_array = np.zeros([image_size, image_size])
        fxx_array = np.zeros([image_size, image_size])
        fyy_array = np.zeros([image_size, image_size])
        fxy_array = np.zeros([image_size, image_size])
        for y in range (image_size):
            for x in range (image_size):
                for peak in range(random_set_size):
                    mu = random_set[0,peak] * image_size
                    nu = random_set[1,peak] * image_size
                    sigma = (sigma_min + sigma_spread * random_set[2,peak]) * image_size
                    xs = x - 0.5
                    ys = y - 0.5
                    hill = math.exp(-(((xs - mu)**2 + (ys - nu)**2)/(2 * sigma**2)))
                    fx_array[x,y] += height_per_peak * ((mu - xs)/sigma**2 * hill)
                    fy_array[x,y] += height_per_peak * ((nu - ys)/sigma**2 * hill)
                    fxx_array[x,y] += height_per_peak * (((xs-mu)**2 - sigma**2)/sigma**4 * hill)
                    fyy_array[x,y] += height_per_peak * (((ys-nu)**2 - sigma**2)/sigma**4 * hill)
                    fxy_array[x,y] += height_per_peak * ((xs-mu)*(ys-nu)/sigma**4 * hill)
# Derivative
                            
        arr = np.arctan(np.sqrt(fx_array**2 + fy_array**2))*180/math.pi
                
        fn = self.path + fn_trunc + '_slope.tif'
        helper.create_tif(arr,fn)
        if (self.auto_plot):
            helper.plot(fn)
             
# Profile curvature    
        denom = fx_array**2 + fy_array**2
        arr = -100*(fxx_array * fx_array**2 + 2 * fxy_array * fx_array * fy_array + fyy_array * fy_array**2) / denom
        fn = self.path + fn_trunc + '_profile.tif'
        helper.create_tif(arr,fn)
        if (self.auto_plot):
            helper.plot(fn)
    
# Tangential curvature                    
        denom = fx_array**2 + fy_array**2
        arr = -100*(fxx_array * fy_array**2 - 2 * fxy_array * fx_array * fy_array + fyy_array * fx_array**2) / denom
        fn = self.path + fn_trunc + '_tangential.tif'
        helper.create_tif(arr,fn)
        if (self.auto_plot):
            helper.plot(fn)
                    
# Contour curvature                    
        denom = (fx_array**2 + fy_array**2) * np.sqrt(fx_array**2+fy_array**2)
        arr = -100*(fxx_array * fy_array**2 - 2 * fxy_array * fx_array * fy_array + fyy_array * fx_array**2) / denom
        fn = self.path + fn_trunc + '_contour.tif'
        helper.create_tif(arr,fn)
        if (self.auto_plot):
            helper.plot(fn)

# Proper Profile curvature
        denom = (fx_array**2 + fy_array**2) * np.sqrt(1+(fx_array**2+fy_array**2))**3
        arr = -100*(fxx_array * fx_array**2 + 2 * fxy_array * fx_array * fy_array + fyy_array * fy_array**2) / denom
        fn = self.path + fn_trunc + '_proper_profile.tif'
        helper.create_tif(arr,fn)
        if (self.auto_plot):
            helper.plot(fn)

# Proper Tangential curvature                    
        denom = (fx_array**2 + fy_array**2) * np.sqrt(1+(fx_array**2+fy_array**2))
        arr = -100*(fxx_array * fy_array**2 - 2 * fxy_array * fx_array * fy_array + fyy_array * fx_array**2) / denom
        fn = self.path + fn_trunc + '_proper_tangential.tif'
        helper.create_tif(arr,fn)
        if (self.auto_plot):
            helper.plot(fn)

                
        return self.path+fn_trunc + '.tif'
