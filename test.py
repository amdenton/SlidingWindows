import unittest
from windowagg.sliding_window import SlidingWindow
import numpy as np
import rasterio
import math
import os

class TestSlidingWindow(unittest.TestCase):

    test_path = 'test_img/'

    def test_aggregation(self):
        with SlidingWindow(self.test_path + 'random.tif') as slide_window:
            with rasterio.open(self.test_path + 'random.tif') as img:
                arr = img.read(1).astype(float)

            sum_all = slide_window._partial_aggregation(arr, 0, 5, '++++')
            partial_sum_all = slide_window._partial_aggregation(arr, 0, 2, '++++')
            partial_sum_all = slide_window._partial_aggregation(partial_sum_all, 2, 3, '++++')
            partial_sum_all = slide_window._partial_aggregation(partial_sum_all, 3, 5, '++++')
            brute_sum_all = slide_window._aggregation_brute(arr, '++++', 5)

            sum_bottom = slide_window._partial_aggregation(arr, 0, 5, '--++')
            partial_sum_bottom = slide_window._partial_aggregation(arr, 0, 2, '--++')
            partial_sum_bottom = slide_window._partial_aggregation(partial_sum_bottom, 2, 3, '--++')
            partial_sum_bottom = slide_window._partial_aggregation(partial_sum_bottom, 3, 5, '--++')
            brute_sum_bottom = slide_window._aggregation_brute(arr, '--++', 5)

            sum_right = slide_window._partial_aggregation(arr, 0, 5, '-+-+')
            partial_sum_right = slide_window._partial_aggregation(arr, 0, 2, '-+-+')
            partial_sum_right = slide_window._partial_aggregation(partial_sum_right, 2, 3, '-+-+')
            partial_sum_right = slide_window._partial_aggregation(partial_sum_right, 3, 5, '-+-+')
            brute_sum_right = slide_window._aggregation_brute(arr, '-+-+', 5)

            sum_main_diag = slide_window._partial_aggregation(arr, 0, 5, '+--+')
            partial_sum_main_diag = slide_window._partial_aggregation(arr, 0, 2, '+--+')
            partial_sum_main_diag = slide_window._partial_aggregation(partial_sum_main_diag, 2, 3, '+--+')
            partial_sum_main_diag = slide_window._partial_aggregation(partial_sum_main_diag, 3, 5, '+--+')
            brute_sum_main_diag = slide_window._aggregation_brute(arr, '+--+', 5)

            maximum = slide_window._partial_aggregation(arr, 0, 5, 'max')
            partial_maximum = slide_window._partial_aggregation(arr, 0, 2, 'max')
            partial_maximum = slide_window._partial_aggregation(partial_maximum, 2, 3, 'max')
            partial_maximum = slide_window._partial_aggregation(partial_maximum, 3, 5, 'max')
            brute_maximum = slide_window._aggregation_brute(arr, 'max', 5)

            minimum = slide_window._partial_aggregation(arr, 0, 5, 'min')
            partial_minimum = slide_window._partial_aggregation(arr, 0, 2, 'min')
            partial_minimum = slide_window._partial_aggregation(partial_minimum, 2, 3, 'min')
            partial_minimum = slide_window._partial_aggregation(partial_minimum, 3, 5, 'min')
            brute_minimum = slide_window._aggregation_brute(arr, 'min', 5)

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
        with SlidingWindow(self.test_path + 'random4band.tif') as slide_window:
            with rasterio.open(self.test_path + 'random4band.tif') as img:
                arr1 = img.read(1).astype(float)
                arr2 = img.read(2).astype(float)
            arr_good = slide_window._regression(arr1, arr2, 5)
            arr_brute = slide_window._regression_brute(arr1, arr2, 5)

        self.assertTrue(np.allclose(arr_good, arr_brute))

    def test_dem_aggregation(self):
        with SlidingWindow(self.test_path + 'SEgradient.tif') as slide_window:
            slide_window.dem_initialize_arrays()
            slide_window.dem_aggregation_step(5)
            arr_dic = slide_window.dem_arr_dict

            slide_window.dem_initialize_arrays()
            slide_window._dem_aggregation_step_brute(5)
            arr_dic_brute = slide_window.dem_arr_dict

        for key in arr_dic:
            self.assertTrue(np.array_equal(arr_dic[key], arr_dic_brute[key]))

    def test_create_tif(self):
        with SlidingWindow(self.test_path + 'SEgradient.tif') as slide_window:
            slide_window.dem_initialize_arrays()
            slide_window.dem_aggregation_step(2)
            slide_window.dem_mean('z')
            slide_window.dem_aggregation_step(1)
            slide_window.dem_mean('z')

        with rasterio.open(self.test_path + 'SEgradient.tif') as img:
            transform_w1 = img.profile['transform']
        with rasterio.open('SEgradient_z_mean_w4.tif') as img:
            transform_w4 = img.profile['transform']
        with rasterio.open('SEgradient_z_mean_w8.tif') as img:
            transform_w8 = img.profile['transform']

        self.assertTrue(transform_w4[0] == transform_w1[0]*4)
        self.assertTrue(transform_w4[1] == transform_w1[1]*4)
        self.assertTrue(transform_w4[2] == (transform_w1[2] + (math.sqrt(transform_w1[0]**2 + transform_w1[3]**2)*3/2)))
        self.assertTrue(transform_w4[3] == transform_w1[3]*4)
        self.assertTrue(transform_w4[4] == transform_w1[4]*4)
        self.assertTrue(transform_w4[5] == (transform_w1[5] - (math.sqrt(transform_w1[1]**2 + transform_w1[4]**2)*3/2)))

        self.assertTrue(transform_w8[0] == transform_w1[0]*8)
        self.assertTrue(transform_w8[1] == transform_w1[1]*8)
        self.assertTrue(transform_w8[2] == (transform_w1[2] + (math.sqrt(transform_w1[0]**2 + transform_w1[3]**2)*7/2)))
        self.assertTrue(transform_w8[3] == transform_w1[3]*8)
        self.assertTrue(transform_w8[4] == transform_w1[4]*8)
        self.assertTrue(transform_w8[5] == (transform_w1[5] - (math.sqrt(transform_w1[1]**2 + transform_w1[4]**2)*7/2)))

        os.remove('SEgradient_z_mean_w4.tif')


def test_create_tif_skew(self):
        with SlidingWindow(self.test_path + 'SEgradient_-45skew.tif') as slide_window:
            slide_window.dem_initialize_arrays()
            slide_window.dem_aggregation_step(2)
            slide_window.dem_mean('z')
            slide_window.dem_aggregation_step(1)
            slide_window.dem_mean('z')

        with rasterio.open(self.test_path + 'SEgradient_-45skew.tif') as img:
            transform_w1 = img.profile['transform']
        with rasterio.open('SEgradient_-45skew_z_mean_w4.tif') as img:
            transform_w4 = img.profile['transform']
        with rasterio.open('SEgradient_-45skew_z_mean_w8.tif') as img:
            transform_w8 = img.profile['transform']

        self.assertTrue(transform_w4[0] == transform_w1[0]*4)
        self.assertTrue(transform_w4[1] == transform_w1[1]*4)
        self.assertTrue(transform_w4[2] == (transform_w1[2] + (math.sqrt(transform_w1[0]**2 + transform_w1[3]**2)*3/2)))
        self.assertTrue(transform_w4[3] == transform_w1[3]*4)
        self.assertTrue(transform_w4[4] == transform_w1[4]*4)
        self.assertTrue(transform_w4[5] == (transform_w1[5] - (math.sqrt(transform_w1[1]**2 + transform_w1[4]**2)*3/2)))

        self.assertTrue(transform_w8[0] == transform_w1[0]*8)
        self.assertTrue(transform_w8[1] == transform_w1[1]*8)
        self.assertTrue(transform_w8[2] == (transform_w1[2] + (math.sqrt(transform_w1[0]**2 + transform_w1[3]**2)*7/2)))
        self.assertTrue(transform_w8[3] == transform_w1[3]*8)
        self.assertTrue(transform_w8[4] == transform_w1[4]*8)
        self.assertTrue(transform_w8[5] == (transform_w1[5] - (math.sqrt(transform_w1[1]**2 + transform_w1[4]**2)*7/2)))

        os.remove('SEgradient_-45skew_z_mean_w4.tif')
        os.remove('SEgradient_-45skew_z_mean_w8.tif')

if __name__ == '__main__':
    unittest.main()
