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

import unittest
import math
import os
import shutil
import random

import numpy as np
import rasterio

class TestSpatialRegression(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print('Setting things up')
        self.test_dir = 'test_img/'
        self.image_size = 128
        self.num_aggre = 5
        self.num_test_points = 20
        self.dtype = config.work_dtype

        self.img_gen = ImageGenerator(self.test_dir)
        #self.gauss = self.img_gen.gauss(self.image_size)
        #self.random = self.img_gen.random(self.image_size)
        self.noisy_line = self.img_gen.noisy_line(self.image_size,.7)


    @classmethod
    def tearDownClass(self):
        print('Keeping things for now')
        #if (os.path.exists(self.test_dir)):
        #    shutil.rmtree(self.test_dir)
        
    def test_img_reading(self):
        print('Will test image reading')
        
    def test_data_aggregation(self):
        with SlidingWindow(self.noisy_line) as slide_window:
            arr = slide_window._img.read(1).astype(self.dtype)
            print(arr)
            slide_window.initialize_dem()
            slide_window.aggregate_dem(self.num_aggre)
            arr_dict = slide_window._dem_data
            
            print('z: ')
            print(arr_dict.z())
            print('xz: ')
            print(arr_dict.xz())
            print('yz: ')
            print(arr_dict.yz())
            print('xxz: ')
            print(arr_dict.xxz())
            print('yyz: ')
            print(arr_dict.yyz()) 
            print('xyz: ')
            print(arr_dict.xyz())
            arr_slope_num = (np.multiply(arr_dict.z(),arr_dict.xyz()) - np.multiply(arr_dict.xz(),arr_dict.yz()))
            print("numerator: ")
            print(arr_slope_num)
            arr_slope_denom = (np.multiply(arr_dict.z(),arr_dict.xxz()) - np.multiply(arr_dict.xz(),arr_dict.xz()))
            print("denominator: ")
            print(arr_slope_denom)
            arr_slope = np.divide(arr_slope_num,arr_slope_denom)
            print("slope: ")
            print(arr_slope)

if __name__ == '__main__':
    unittest.main()
