#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last updated on Tue Dec 14

@authors: Anne Denton, David Schwarz, Rahul Gomes

License information:
https://opensource.org/licenses/GPL-3.0
"""

from windowagg.sliding_window import SlidingWindow
from image_generator import ImageGenerator
import windowagg.config as config

import time
import os

import numpy as np
import rasterio
import matplotlib.pyplot as plt

class SlidingWindowComparison():
    
    def __init__(self):
#=============================================================================
# Please select one of the valid ids, currently between 0 and 3
# 0: 'artificial_no_noise': Produce the artificial image and its analytical and numerical slope and curvatures without noise
# 1: 'artificial_noise': Same as 0 with noise
# 2: 'landscape': Compute numerical slope and curvature for Souris landscape
# 3: 'speed': Runtime over large basic image
# 4: 'difference': Shows difference between numerical and analytical results for IART
#     Assumes that all plosts exist already, so please choice 1!
# 5:  r_squared
        self.analyses = ['artificial_no_noise','artificial_noise','landscape','speed','difference']
        analysis_id = 1
# Display Tiff images using matplotlib.pyplot as they are created (True/False)
# Note that Choice 3 (speed) has auto_plot turned off unless it is changed further down
# and Choice 4 (difference) only provides pyplot figures as output, and does so regardless of auto_plot setting
        self.auto_plot = True
# Number of aggregations that are requested, maximum window size will be 2^num_aggre
        self.num_aggre = 6
#=============================================================================
        self.analysis = self.analyses[analysis_id]
        self.dtype = config.work_dtype
        if self.analysis == self.analyses[0]:
            self.img_dir = 'img_artificial/'
            self.image_size = 512
            self.img_gen = ImageGenerator(self.img_dir)
            self.img_gen.auto_plot = self.auto_plot
            self.multi_gauss = self.img_gen.multi_gauss(self.image_size,0,0)
        elif self.analysis == self.analyses[1]:
            self.img_dir = 'img_artificial/'
            self.image_size = 512
            self.img_gen = ImageGenerator(self.img_dir)
            self.img_gen.auto_plot = self.auto_plot
            self.multi_gauss = self.img_gen.multi_gauss(self.image_size,0,0.01)
        elif self.analysis == self.analyses[2]:
            self.img_dir = 'img_souris/'
            self.image_size = 500
            self.img_gen = ImageGenerator(self.img_dir)
            self.img_gen.auto_plot = self.auto_plot
            self.landscape = self.img_gen.landscape(self.img_dir+'SourisSmall.tif')
        elif self.analysis == self.analyses[3]:
            self.auto_plot = False
            self.img_dir = 'img_dome/'
            self.image_size = 5000
            self.img_gen = ImageGenerator(self.img_dir)
            self.img_gen.auto_plot = self.auto_plot
            self.dome = self.img_gen.dome(self.image_size)
        elif self.analysis == self.analyses[4]:
            self.img_dir = 'img_artificial/'
            self.image_size = 512
            self.img_gen = ImageGenerator(self.img_dir)
            self.img_gen.auto_plot = self.auto_plot
            self.landscape = self.img_gen.landscape(self.img_dir+'multi_random_id0_n10.tif')
                                    
    def compute_multi(self,img_gen,num_aggre):
        with SlidingWindow(img_gen) as slide_window:
            print('img_gen: ',self.img_gen)
            slide_window.auto_plot = self.auto_plot
            slide_window.initialize_dem()
            slide_window.aggregate_dem(1)
            slide_window.dem_slope()
            for num_loc in range(2,num_aggre+1):
                slide_window.aggregate_dem(1)
                slide_window.dem_slope()
                slide_window.dem_profile()
                slide_window.dem_tangential()
                slide_window.dem_contour()
                slide_window.dem_proper_profile()
                slide_window.dem_proper_tangential()
 
    def speed_numbers(self,img_gen_fn,num_aggre):
        time_array = np.zeros([num_aggre-1,4])
        with SlidingWindow(img_gen_fn) as slide_window:
            slide_window.auto_plot = self.auto_plot
            slide_window.initialize_dem()
            time_start = time.time_ns()
            i = 0
            time_start = time.time_ns()
            slide_window.aggregate_dem()
            time0 = time.time_ns() - time_start
            print('time0: ',time0)
            time_start = time.time_ns()
            for num_aggre_loc in range(2,num_aggre+1):
                slide_window.aggregate_dem()
                time_array[i,0] = time.time_ns() - time_start
                time_start = time.time_ns()
                slide_window.dem_slope()
                time_array[i,1] = time.time_ns() - time_start
                time_start = time.time_ns()
                slide_window.dem_profile()
                time_array[i,2] = time.time_ns() - time_start
                time_start = time.time_ns()
                slide_window.dem_tangential()
                time_array[i,3] = time.time_ns() - time_start
                time_start = time.time_ns()
                i += 1
            print('============================================================')
            print('Aggregation [from w=2 to w=4, w=4 to w=8, w=8 to w=16, w=16 to w=32, and w=32 to w=64]')
            print(time_array[:,0])
            print()
            print('slope [w=4, w=8, w=16, w=32, w=64]')
            print(time_array[:,1])
            print()
            print('profile [w=4, w=8, w=16, w=32, w=64]')
            print(time_array[:,2])
            print()
            print('tangential [w=4, w=8, w=16, w=32, w=64]')
            print(time_array[:,3])
            print('============================================================')            
        return time_array

    def remove_frame_iart(self,arr,w):
        shift = int(w/2)
        out_array = np.empty((arr.shape[0]-w+1,arr.shape[1]-w+1))
        height = out_array.shape[0]
        width = out_array.shape[1]
        for j in range(height):
            for i in range(width):
                out_array[j,i] = arr[j+shift,i+shift]
        return out_array
    
    def plot_differences(self,fn_trunc,fn_trunc_analytical):
        base_name = os.path.basename(fn_trunc).split('.')[0]
        quant_min = 0.1
        quant_max = 0.9
        for analysis in ('_slope','_profile','_tangential'):
            for num_agg_loc in range (2,self.num_aggre+1):
                w = int(2**num_agg_loc)
                file_name = self.img_dir+base_name+analysis+'_w='+str(w)+'.tif'
                with rasterio.open(file_name) as img:
                    array_0 = img.read(1)
                    file_name_analytical = self.img_dir+base_name+analysis+'.tif'
                with rasterio.open(file_name_analytical) as img:
                    array_1 = self.remove_frame_iart(img.read(1),w)
    
                plt.figure()
                array_diff = array_0-array_1
                low = np.quantile(array_diff,quant_min)
                array_diff[array_diff < low] = low
                high = np.quantile(array_diff,quant_max)
                array_diff[array_diff > high] = high
                shift = int(round(array_diff.shape[0]/50))
                plt.text(0,-4*shift,'Difference from analytical for '+file_name)
                plt.text(0,-shift,'min: '+str(round(np.amin(array_diff),3))+' max: '+str(round(np.amax(array_diff),3)))
                plt.imshow(array_diff,'gray_r')
    
self = SlidingWindowComparison()


if self.analysis == self.analyses[0]:
    img_gen_fn = self.multi_gauss
    self.compute_multi(img_gen_fn,self.num_aggre)

elif self.analysis == self.analyses[1]:
    img_gen_fn = self.multi_gauss
    self.compute_multi(img_gen_fn,self.num_aggre)

elif self.analysis == self.analyses[2]:
    img_gen_fn = self.landscape
    self.compute_multi(img_gen_fn,self.num_aggre)

elif self.analysis == self.analyses[3]:
    img_gen_fn = self.dome
    self.speed_numbers(img_gen_fn,self.num_aggre)

elif self.analysis == self.analyses[4]:
    img_gen_fn = self.landscape
    self.plot_differences(img_gen_fn,img_gen_fn)

