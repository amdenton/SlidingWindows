import unittest
from windowagg.sliding_window import SlidingWindow
import numpy as np
import rasterio
import math
import os

class TestSlidingWindow(unittest.TestCase):

    fp = 'test_img/'
    create_tif_test_img = ('SEgradient', 'SEgradient_-45skew', 'Sgradient', 'Sgradient_-45skew')

    def test_aggregation(self):
        with SlidingWindow(self.fp + 'random.tif') as slide_window:
            with rasterio.open(self.fp + 'random.tif') as img:
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
        with SlidingWindow(self.fp + 'random4band.tif') as slide_window:
            with rasterio.open(self.fp + 'random4band.tif') as img:
                arr1 = img.read(1).astype(float)
                arr2 = img.read(2).astype(float)
            arr_good = slide_window._regression(arr1, arr2, 5)
            arr_brute = slide_window._regression_brute(arr1, arr2, 5)

        self.assertTrue(np.allclose(arr_good, arr_brute))

    def test_dem_aggregation(self):
        with SlidingWindow(self.fp + 'SEgradient.tif') as slide_window:
            slide_window.dem_initialize_arrays()
            slide_window.dem_aggregation_step(5)
            arr_dic = slide_window.dem_arr_dict

            slide_window.dem_initialize_arrays()
            slide_window._dem_aggregation_step_brute(5)
            arr_dic_brute = slide_window.dem_arr_dict

        for key in arr_dic:
            self.assertTrue(np.array_equal(arr_dic[key], arr_dic_brute[key]))

    def test_create_tif_transform(self):
        agg_num_1 = 2
        agg_num_2 = 5

        pix_agg_1 = 2**agg_num_1
        pix_agg_2 = 2**agg_num_2
        for path in self.create_tif_test_img:
            with self.subTest(path=path):
                paths = (self.fp+path+'.tif', path+'_z_mean_w'+str(pix_agg_1)+'.tif', path+'_z_mean_w'+str(pix_agg_2)+'.tif')
                try:
                    with SlidingWindow(paths[0]) as slide_window:
                        slide_window.dem_initialize_arrays()
                        slide_window.dem_aggregation_step(agg_num_1)
                        slide_window.dem_mean('z')
                        slide_window.dem_aggregation_step(agg_num_2 - agg_num_1)
                        slide_window.dem_mean('z')

                    with rasterio.open(paths[0]) as img:
                        transform = img.profile['transform']
                    with rasterio.open(paths[1]) as img:
                        transform_1 = img.profile['transform']
                    with rasterio.open(paths[2]) as img:
                        transform_2 = img.profile['transform']

                    map_width = math.sqrt(transform[0]**2 + transform[3]**2)
                    map_height = math.sqrt(transform[1]**2 + transform[4]**2)
                    self.assertTrue(transform_1[0] == transform[0]*pix_agg_1)
                    self.assertTrue(transform_1[1] == transform[1]*pix_agg_1)
                    self.assertTrue(transform_1[2] == (transform[2] + (map_width*(pix_agg_1 - 1)/2)))
                    self.assertTrue(transform_1[3] == transform[3]*pix_agg_1)
                    self.assertTrue(transform_1[4] == transform[4]*pix_agg_1)
                    self.assertTrue(transform_1[5] == (transform[5] - (map_height*(pix_agg_1 - 1)/2)))

                    self.assertTrue(transform_2[0] == transform[0]*pix_agg_2)
                    self.assertTrue(transform_2[1] == transform[1]*pix_agg_2)
                    self.assertTrue(transform_2[2] == (transform[2] + (map_width*(pix_agg_2 - 1)/2)))
                    self.assertTrue(transform_2[3] == transform[3]*pix_agg_2)
                    self.assertTrue(transform_2[4] == transform[4]*pix_agg_2)
                    self.assertTrue(transform_2[5] == (transform[5] - (map_height*(pix_agg_2 - 1)/2)))

                    for i in range(1, len(paths)):
                        self.assertTrue(os.path.exists(paths[i]))

                finally:
                    for i in range(1, len(paths)):
                        if os.path.exists(paths[i]):
                            os.remove(paths[i])

    def test_create_tif_export(self):
        num_aggre = 0
        path = self.fp + self.create_tif_test_img[1] + '.tif'
        new_path = self.create_tif_test_img[1] + '_export_w' + str(2**num_aggre) + '.tif'
        try:
            with SlidingWindow(path) as slide_window:
                slide_window.dem_initialize_arrays()
                slide_window.dem_aggregation_step(num_aggre)
                slide_window.dem_export_arrays()
            self.assertTrue(os.path.exists(new_path))

            with rasterio.open(path) as img:
                transform = img.profile['transform']
            with rasterio.open(new_path) as img:
                transform_new = img.profile['transform']

            with SlidingWindow(new_path) as slide_window:
                slide_window.dem_import_arrays()
                self.assertTrue(slide_window.dem_pixels_aggre == 2**num_aggre)
                self.assertTrue(len(slide_window.dem_arr_dict) == 6)

            for i in range(6):
                self.assertTrue(transform_new[i] == transform[i])
            
        finally:
            if os.path.exists(new_path):
                os.remove(new_path)

if __name__ == '__main__':
    unittest.main()
