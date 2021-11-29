#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:46:47 2021

@author: adenton
"""

from windowagg.sliding_window import SlidingWindow
import windowagg.rbg as rbg
import windowagg.dem as dem
from windowagg.agg_ops import Agg_ops
import windowagg.aggregation as aggregation
from image_generator import ImageGenerator
import windowagg.config as config
import windowagg.helper as helper

import unittest
import math
import os
import shutil
import random

import numpy as np
import rasterio
import rasterio.features

class TestSpatialRegression(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print('Setting things up')
        #self.test_dir = 'test_img/'
        self.test_dir = ''
        self.image_size = 1024
        self.num_aggre = 6
        self.num_test_points = 20
        self.dtype = config.work_dtype
        self.thresh = 50
        self.zthresh = 0.01

        if False:
            with rasterio.open('corn_image_access_green_int16.tif') as dataset:
        
                # Read the dataset's valid data mask as a ndarray.
                mask = dataset.dataset_mask()
        
                # Extract feature shapes and values from the array.
                for geom, val in rasterio.features.shapes(
                        mask, transform=dataset.transform):
            
                    print(dataset.crs)

                print(dataset.meta)
        
                array = dataset.read(1)
                orig_width = np.size(array,0)
                orig_height = np.size(array,1)
                small_array = np.empty([self.image_size,self.image_size])
                for i in range(self.image_size):
                    for j in range(self.image_size):
                        a = array[int(orig_width/4)+i,int(orig_height/4)+j]
                        if a < self.thresh: 
                            small_array[i,j] = 0
                        else:
                            small_array[i,j] = 1
            
                print(small_array)
                print(np.amax(small_array))
                for i, dtype, nodataval in zip(dataset.indexes, dataset.dtypes, dataset.nodatavals):
                    print(i, dtype, nodataval)
                
                fn = 'corn_small.tif'
                helper.create_tif(small_array,fn)
                
        self.img_gen = ImageGenerator(self.test_dir)
        self.corn_small = self.img_gen.corn_small('corn_small.tif')
        self.s_gradient = self.img_gen.s_gradient(self.image_size-2**self.num_aggre+1)
        self.e_gradient = self.img_gen.e_gradient(self.image_size-2**self.num_aggre+1)

    @classmethod
    def tearDownClass(self):
        print('Keeping things for now')
        #if (os.path.exists(self.test_dir)):
        #    shutil.rmtree(self.test_dir)
        
    def test_img_reading(self):
        print('Will test image reading')
        
    def test_data_aggregation(self):
        with rasterio.open('s_gradient.tif') as dataset:
            s_arr = dataset.read(1)
            print('s_arr')
            print(s_arr)
            print(s_arr[0,range(20)])
            print(s_arr[range(20),0])
        with rasterio.open('e_gradient.tif') as dataset:
            e_arr = dataset.read(1)
            print('e_arr')
            print(e_arr)
            print(e_arr[0,range(20)])
            print(e_arr[range(20),0])

        with SlidingWindow(self.corn_small) as slide_window:
            arr = slide_window._img.read(1).astype(self.dtype)
            print(arr)
            slide_window.initialize_dem()
            slide_window.aggregate_dem(self.num_aggre)
            arr_dict = slide_window._dem_data
            # e_arr and s_arr seem to be creating rounding errors or maybe there's an error in the logic of adding them
            print('z: ')
            arr_z = arr_dict.z()
            print(arr_z)
            print('xz: ')
            arr_xz = arr_dict.xz() #+ e_arr*arr_dict.z()
            print(arr_xz)
            print('yz: ')
            arr_yz = arr_dict.yz() #+ s_arr*arr_dict.z()
            print(arr_yz)
            print('xxz: ')
            arr_xxz = arr_dict.xxz() #+ 2*arr_dict.xz()*e_arr*arr_dict.z() + e_arr*e_arr*arr_dict.z()*arr_dict.z()
            print(arr_xxz)
            print('yyz: ')
            arr_yyz = arr_dict.yyz() #+ 2*arr_dict.yz()*s_arr*arr_dict.z() + s_arr*s_arr*arr_dict.z()*arr_dict.z()
            print(arr_yyz)
            print('xyz: ')
            arr_xyz = arr_dict.xyz() #+ e_arr*arr_dict.z()*arr_dict.yz() + s_arr*arr_dict.z()*arr_dict.xz() + e_arr * s_arr * arr_dict.z() * arr_dict.z() 
            print(arr_xyz)
            arr_slope_num = (np.multiply(arr_z,arr_xyz - np.multiply(arr_xz,arr_yz)))
            print("numerator: ")
            print(arr_slope_num)
            arr_slope_denom = (np.multiply(arr_z,arr_xxz - np.multiply(arr_xz,arr_xz)))
            print("denominator: ")
            print(arr_slope_denom)
            arr_slope = np.divide(arr_slope_num,arr_slope_denom)
            print("slope: ")
            print(arr_slope)
            arr_y_offset = arr_dict.yz() - arr_slope * arr_dict.xz()
            arr_y_center = np.zeros(arr_z.shape)
            #quantile_z = np.quantile(arr_z,0.5)
            #print('quantile_z ',quantile_z)
            #arr_var = arr_slope_num * arr_slope_num / (arr_slope_denom * (arr_z*arr_yyz - arr_yz*arr_yz))
            for x in range(arr_z.shape[0]):
                for y in range(arr_z.shape[1]):
                    if arr_z[y,x] > 0 and arr_yz[y,x] > 0:
                        # I haven't understood that factor of 2
                        y_new = y+int(arr_y_offset[y,x]/arr_z[y,x])
                        if y_new >= 0 and y_new < arr_z.shape[0]:
                            if abs(arr_y_offset[y,x]/arr_z[y,x])<5 and (arr_z[y,x]*arr_yyz[y,x]/(arr_yz[y,x]*arr_yz[y,x])<20):
                                arr_y_center[y_new,x] = 1
            print('arr_y_center')
            print(arr_y_center)
                            
            arr_center_clean_bool = aggregation.aggregate(arr_y_center,Agg_ops.add_all,3)>7
            print('arr_center_clean_bool')
            print(arr_center_clean_bool)
            arr_center_clean = arr_center_clean_bool.astype('int8')
            #arr_center_clean_bool = aggregation.aggregate(arr_center_clean,Agg_ops.add_all)>1
            #print('arr_center_clean_bool')
            #print(arr_center_clean_bool)
            #arr_center_clean = arr_center_clean_bool.astype('int8')
            print('arr_center_clean')
            print(arr_center_clean)

            #arr_center_clean = arr_center_clean_bool.astype('int16')
            #arr_large = np.absolute(arr_slope)
            #arr_mean = 
            #arr_bool_large = arr_large > 0.5
            #arr_int_large = arr_bool_large.astype(int)
            #arr_spread = np.multiply(arr_dict.yyz(),arr_dict.z() - np.multiply(arr_dict.yz(),arr_dict.yz()))
            #arr_cover = 10*arr_dict.z()
            #arr_bool_cover = arr_cover > 0.01
            #arr_int_cover = arr_bool_cover.astype(int)
            #arr_large = np.multiply(arr_int_cover,arr_int_large)
            #arr_spread = np.multiply(arr_dict.yyz(),arr_dict.z())
            #arr_spread = np.multiply(arr_dict.yz(),arr_dict.yz())
            
            #fn = 'new_slope.tif'
            fn = 'clean_center.tif'
            helper.create_tif(helper.arr_dtype_conversion(arr_center_clean, self.dtype),fn)
            #helper.create_tif(arr_y_center,fn)
            #helper.create_tif(arr_y_offset,fn)
            #helper.create_tif(arr_cover,fn)

if __name__ == '__main__':
    unittest.main()
