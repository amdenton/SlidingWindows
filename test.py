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

class TestSlidingWindow(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_dir = 'test_img/'
        self.image_size = 64
        self.num_aggre = 4
        self.num_test_points = 20
        self.dtype = config.work_dtype

        self.img_gen = ImageGenerator(self.test_dir)
        self.gauss = self.img_gen.gauss(self.image_size)
        self.random = self.img_gen.random(self.image_size)


    @classmethod
    def tearDownClass(self):
        if (os.path.exists(self.test_dir)):
            shutil.rmtree(self.test_dir)

    def test_aggregation(self):
        with rasterio.open(self.random) as img:
            arr = img.read(1).astype(self.dtype)

        sum_all = aggregation.aggregate(arr, Agg_ops.add_all, 5, 0)
        partial_sum_all = aggregation.aggregate(arr, Agg_ops.add_all, 2, 0)
        partial_sum_all = aggregation.aggregate(partial_sum_all, Agg_ops.add_all, 1, 2)
        partial_sum_all = aggregation.aggregate(partial_sum_all, Agg_ops.add_all, 2, 3)
        brute_sum_all = aggregation.aggregate_brute(arr, Agg_ops.add_all, 5, 0)

        sum_bottom = aggregation.aggregate(arr, Agg_ops.add_bottom, 5, 0)
        partial_sum_bottom = aggregation.aggregate(arr, Agg_ops.add_bottom, 2, 0)
        partial_sum_bottom = aggregation.aggregate(partial_sum_bottom, Agg_ops.add_bottom, 1, 2)
        partial_sum_bottom = aggregation.aggregate(partial_sum_bottom, Agg_ops.add_bottom, 2, 3)
        brute_sum_bottom = aggregation.aggregate_brute(arr, Agg_ops.add_bottom, 5, 0)

        sum_right = aggregation.aggregate(arr, Agg_ops.add_right, 5, 0)
        partial_sum_right = aggregation.aggregate(arr, Agg_ops.add_right, 2, 0)
        partial_sum_right = aggregation.aggregate(partial_sum_right, Agg_ops.add_right, 1, 2)
        partial_sum_right = aggregation.aggregate(partial_sum_right, Agg_ops.add_right, 2, 3)
        brute_sum_right = aggregation.aggregate_brute(arr, Agg_ops.add_right, 5, 0)

        sum_main_diag = aggregation.aggregate(arr, Agg_ops.add_main_diag, 5, 0)
        partial_sum_main_diag = aggregation.aggregate(arr, Agg_ops.add_main_diag, 2, 0)
        partial_sum_main_diag = aggregation.aggregate(partial_sum_main_diag, Agg_ops.add_main_diag, 1, 2)
        partial_sum_main_diag = aggregation.aggregate(partial_sum_main_diag, Agg_ops.add_main_diag, 2, 3)
        brute_sum_main_diag = aggregation.aggregate_brute(arr, Agg_ops.add_main_diag, 5, 0)

        maximum = aggregation.aggregate(arr, Agg_ops.maximum, 5, 0)
        partial_maximum = aggregation.aggregate(arr, Agg_ops.maximum, 2, 0)
        partial_maximum = aggregation.aggregate(partial_maximum, Agg_ops.maximum, 1, 2)
        partial_maximum = aggregation.aggregate(partial_maximum, Agg_ops.maximum, 2, 3)
        brute_maximum = aggregation.aggregate_brute(arr, Agg_ops.maximum, 5, 0)

        minimum = aggregation.aggregate(arr, Agg_ops.minimum, 5, 0)
        partial_minimum = aggregation.aggregate(arr, Agg_ops.minimum, 2, 0)
        partial_minimum = aggregation.aggregate(partial_minimum, Agg_ops.minimum, 1, 2)
        partial_minimum = aggregation.aggregate(partial_minimum, Agg_ops.minimum, 2, 3)
        brute_minimum = aggregation.aggregate_brute(arr, Agg_ops.minimum, 5, 0)

        self.assertTrue(np.array_equal(sum_all, brute_sum_all))
        self.assertTrue(np.array_equal(partial_sum_all, brute_sum_all))

        self.assertTrue(np.array_equal(sum_bottom, brute_sum_bottom))
        self.assertTrue(np.array_equal(partial_sum_bottom, brute_sum_bottom))

        self.assertTrue(np.array_equal(sum_right, brute_sum_right))
        self.assertTrue(np.array_equal(partial_sum_right, brute_sum_right))

        self.assertTrue(np.array_equal(sum_main_diag, brute_sum_main_diag))
        self.assertTrue(np.array_equal(partial_sum_main_diag, brute_sum_main_diag))

        self.assertTrue(np.array_equal(maximum, brute_maximum))
        self.assertTrue(np.array_equal(partial_maximum, brute_maximum))

        self.assertTrue(np.array_equal(minimum, brute_minimum))
        self.assertTrue(np.array_equal(partial_minimum, brute_minimum))

    def test_regression(self):
        band_num_1 = 1
        band_num_2 = 2
        try:
            with SlidingWindow(self.random) as slide_window:
                slide_window.convert_image = False
                slide_window.tif_dtype = self.dtype
                regression_path = slide_window.regression(band_num_1, band_num_2, self.num_aggre)
                with rasterio.open(regression_path) as img:
                    arr_good = img.read(1)

                band_1 = slide_window._img.read(band_num_1).astype(self.dtype)
                band_2 = slide_window._img.read(band_num_2).astype(self.dtype)
                
            arr_brute = rbg.regression_brute(band_1, band_2, self.num_aggre)
            
        finally:
            if (os.path.exists(regression_path)):
                os.remove(regression_path)

        self.assertTrue(np.allclose(arr_good, arr_brute))

    def test_data_init(self):
        with SlidingWindow(self.random) as slide_window:
            arr = slide_window._img.read(1).astype(self.dtype)
            slide_window.initialize_dem()
            arr_dict = slide_window._dem_data
            arr_zeros = np.zeros_like(arr)
            self.assertTrue(np.array_equal(arr_dict.z(), arr))
            self.assertTrue(np.array_equal(arr_dict.xz(), arr_zeros))
            self.assertTrue(np.array_equal(arr_dict.yz(), arr_zeros))
            self.assertTrue(np.array_equal(arr_dict.xxz(), arr_zeros))
            self.assertTrue(np.array_equal(arr_dict.yyz(), arr_zeros)) 
            self.assertTrue(np.array_equal(arr_dict.xyz(), arr_zeros))


    def test_data_aggregation(self):
        with SlidingWindow(self.random) as slide_window:
            arr = slide_window._img.read(1).astype(self.dtype)
            slide_window.initialize_dem()
            slide_window.aggregate_dem(self.num_aggre)
            arr_dict = slide_window._dem_data
            
            self.assertTrue(np.array_equal(arr_dict.z(), aggregation.aggregate_z_brute(arr, self.num_aggre)))
            self.assertTrue(np.array_equal(arr_dict.xz(), aggregation.aggregate_xz_brute(arr, self.num_aggre)))
            self.assertTrue(np.array_equal(arr_dict.yz(), aggregation.aggregate_yz_brute(arr, self.num_aggre)))
            self.assertTrue(np.array_equal(arr_dict.xxz(), aggregation.aggregate_xxz_brute(arr, self.num_aggre)))
            self.assertTrue(np.array_equal(arr_dict.yyz(), aggregation.aggregate_yyz_brute(arr, self.num_aggre))) 
            self.assertTrue(np.array_equal(arr_dict.xyz(), aggregation.aggregate_xyz_brute(arr, self.num_aggre)))

    def test_aspect(self):
        with SlidingWindow(self.gauss) as slide_window:
            slide_window.convert_image = False
            slide_window.initialize_dem()
            slide_window.aggregate_dem(self.num_aggre)
            try:
                aspect_path = slide_window.dem_aspect()
                with rasterio.open(aspect_path) as img:
                    arr_aspect = img.read(1)

                # assumed to be square
                size = arr_aspect.shape[0]
                for y in range(size):
                    for x in range(size):
                        aspect = self.img_gen.gauss_aspect_point(x, y, size)
                        self.assertTrue(
                            math.isclose(arr_aspect[y, x] + (2 * math.pi), aspect + (2 * math.pi), rel_tol=1e-4)
                            or
                            math.isclose(arr_aspect[y, x] + (2 * math.pi), aspect, rel_tol=1e-4)
                            or
                            math.isclose(arr_aspect[y, x], aspect + (2 * math.pi), rel_tol=1e-4)
                        )

            finally:
                if (os.path.exists(aspect_path)):
                    os.remove(aspect_path)

    def test_standard(self):
        with SlidingWindow(self.gauss) as slide_window:
            slide_window.convert_image = False
            slide_window.initialize_dem()
            slide_window.aggregate_dem(self.num_aggre)
            try:
                standard_path = slide_window.dem_standard()
                with rasterio.open(standard_path) as img:
                    arr_standard = img.read(1)

                # assumed to be square
                size = arr_standard.shape[0]
                for y in range(size):
                    for x in range(size):
                        standard = self.img_gen.gauss_standard_point(x, y, size)

                math.isclose(arr_standard[y, x], standard)

            finally:
                if (os.path.exists(standard_path)):
                    os.remove(standard_path)

    def test_slope(self):
        with SlidingWindow(self.gauss) as slide_window:
            slide_window.convert_image = False
            slide_window.initialize_dem()
            slide_window.aggregate_dem(self.num_aggre)
            try:
                slope_path = slide_window.dem_slope()
                with rasterio.open(slope_path) as img:
                    arr_slope = img.read(1)
                print(arr_slope)

                # assumed to be square
                size = arr_slope.shape[0]
                for y in range(size):
                    for x in range(size):
                        slope = self.img_gen.gauss_standard_point(x, y, size)

                math.isclose(arr_slope[y, x], slope)

            finally:
                if (os.path.exists(slope_path)):
                    os.remove(slope_path)

if __name__ == '__main__':
    unittest.main()
