import unittest
from windowagg.sliding_window import SlidingWindow
from image_generator import ImageGenerator
import numpy as np
import rasterio
import math
import os
import shutil

class TestSlidingWindow(unittest.TestCase):

    # used in tearDownClass method
    test_dir = 'test_img/'

    img_gen = ImageGenerator()
    img_gen.test_dir = test_dir

    # remove test folder after testing completes
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(TestSlidingWindow.test_dir)

    def test_aggregation(self):
        path = self.img_gen.random()
        with SlidingWindow(path) as slide_window:
            with rasterio.open(path) as img:
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
        path = self.img_gen.random()
        with SlidingWindow(path) as slide_window:
            with rasterio.open(path) as img:
                arr1 = img.read(1).astype(float)
                arr2 = img.read(2).astype(float)
            arr_good = slide_window._regression(arr1, arr2, 5)
            arr_brute = slide_window._regression_brute(arr1, arr2, 5)

        self.assertTrue(np.allclose(arr_good, arr_brute))

    def test_dem_aggregation(self):
        path = self.img_gen.random()
        with SlidingWindow(path) as slide_window:
            slide_window.dem_initialize_arrays()
            slide_window.dem_aggregation_step(5)
            arr_dic = slide_window.dem_arr_dict

            slide_window.dem_initialize_arrays()
            slide_window._dem_aggregation_step_brute(5)
            arr_dic_brute = slide_window.dem_arr_dict

        for key in arr_dic:
            self.assertTrue(np.array_equal(arr_dic[key], arr_dic_brute[key]))

    def test_dem_aggregation2(self):
        # image size must be even
        image_size = 4
        path = self.img_gen.random(image_size=image_size, num_bands=1)
        with rasterio.open(path) as img:
                arr = img.read(1).astype(float)
        with SlidingWindow(path) as slide_window:
            slide_window.dem_initialize_arrays()
            arr_dict = slide_window.dem_arr_dict
            new_arr_dict = {}
            new_arr_dict['z'] = arr
            for key in arr_dict:
                if (key != 'z'):
                    self.assertTrue(np.count_nonzero(arr_dict[key]) == 0)
                elif (key == 'z'):
                    self.assertTrue(np.array_equal(arr_dict[key], arr))

            slide_window.dem_aggregation_step(2)
            arr_dict = slide_window.dem_arr_dict
            new_arr_dict['z'] = np.array([np.sum(arr)/16])

            new_arr_dict['xz'] = np.zeros([1])
            new_arr_dict['xz'][0] += -1.5*arr[0][0] + -.5*arr[0][1] + .5*arr[0][2] + 1.5*arr[0][3]
            new_arr_dict['xz'][0] += -1.5*arr[1][0] + -.5*arr[1][1] + .5*arr[1][2] + 1.5*arr[1][3]
            new_arr_dict['xz'][0] += -1.5*arr[2][0] + -.5*arr[2][1] + .5*arr[2][2] + 1.5*arr[2][3]
            new_arr_dict['xz'][0] += -1.5*arr[3][0] + -.5*arr[3][1] + .5*arr[3][2] + 1.5*arr[3][3]
            new_arr_dict['xz'][0] /= 16

            new_arr_dict['xxz'] = np.zeros([1])
            new_arr_dict['xxz'][0] += (-1.5)**2*arr[0][0] + (-.5)**2*arr[0][1] + .5**2*arr[0][2] + 1.5**2*arr[0][3]
            new_arr_dict['xxz'][0] += (-1.5)**2*arr[1][0] + (-.5)**2*arr[1][1] + .5**2*arr[1][2] + 1.5**2*arr[1][3]
            new_arr_dict['xxz'][0] += (-1.5)**2*arr[2][0] + (-.5)**2*arr[2][1] + .5**2*arr[2][2] + 1.5**2*arr[2][3]
            new_arr_dict['xxz'][0] += (-1.5)**2*arr[3][0] + (-.5)**2*arr[3][1] + .5**2*arr[3][2] + 1.5**2*arr[3][3]
            new_arr_dict['xxz'][0] /= 16

            new_arr_dict['yz'] = np.zeros([1])
            new_arr_dict['yz'][0] += -1.5*arr[0][0] + -1.5*arr[0][1] + -1.5*arr[0][2] + -1.5*arr[0][3]
            new_arr_dict['yz'][0] += -.5*arr[1][0] + -.5*arr[1][1] + -.5*arr[1][2] + -.5*arr[1][3]
            new_arr_dict['yz'][0] += .5*arr[2][0] + .5*arr[2][1] + .5*arr[2][2] + .5*arr[2][3]
            new_arr_dict['yz'][0] += 1.5*arr[3][0] + 1.5*arr[3][1] + 1.5*arr[3][2] + 1.5*arr[3][3]
            new_arr_dict['yz'][0] /= 16

            new_arr_dict['yyz'] = np.zeros([1])
            new_arr_dict['yyz'][0] += (-1.5)**2*arr[0][0] + (-1.5)**2*arr[0][1] + (-1.5)**2*arr[0][2] + (-1.5)**2*arr[0][3]
            new_arr_dict['yyz'][0] += (-.5)**2*arr[1][0] + (-.5)**2*arr[1][1] + (-.5)**2*arr[1][2] + (-.5)**2*arr[1][3]
            new_arr_dict['yyz'][0] += .5**2*arr[2][0] + .5**2*arr[2][1] + .5**2*arr[2][2] + .5**2*arr[2][3]
            new_arr_dict['yyz'][0] += 1.5**2*arr[3][0] + 1.5**2*arr[3][1] + 1.5**2*arr[3][2] + 1.5**2*arr[3][3]
            new_arr_dict['yyz'][0] /= 16

            new_arr_dict['xyz'] = np.zeros([1])
            new_arr_dict['xyz'][0] += -1.5*-1.5*arr[0][0] + -1.5*-.5*arr[0][1] + -1.5*.5*arr[0][2] + -1.5*1.5*arr[0][3]
            new_arr_dict['xyz'][0] += -.5*-1.5*arr[1][0] + -.5*-.5*arr[1][1] + -.5*.5*arr[1][2] + -.5*1.5*arr[1][3]
            new_arr_dict['xyz'][0] += .5*-1.5*arr[2][0] + .5*-.5*arr[2][1] + .5*.5*arr[2][2] + .5*1.5*arr[2][3]
            new_arr_dict['xyz'][0] += 1.5*-1.5*arr[3][0] + 1.5*-.5*arr[3][1] + 1.5*.5*arr[3][2] + 1.5*1.5*arr[3][3]
            new_arr_dict['xyz'][0] /= 16

            for key in arr_dict:
                self.assertTrue(np.allclose(arr_dict[key], new_arr_dict[key]))

    def test_create_tif_transform(self):
        agg_num_1 = 2
        agg_num_2 = 5

        pix_agg_1 = 2**agg_num_1
        pix_agg_2 = 2**agg_num_2
        paths = [self.img_gen.random(), self.img_gen.random(angle=45), self.img_gen.random(angle=75)]
        # initalize to remove files later
        path1 = path2 = ''
        for i in range(len(paths)):
            try:
                with self.subTest(path=paths[i]):
                    with SlidingWindow(paths[i]) as slide_window:
                        slide_window.dem_initialize_arrays()
                        slide_window.dem_aggregation_step(agg_num_1)
                        path1 = slide_window.dem_mean('z')
                        slide_window.dem_aggregation_step(agg_num_2 - agg_num_1)
                        path2 = slide_window.dem_mean('z')

                    with rasterio.open(paths[i]) as img:
                        transform = img.profile['transform']
                    with rasterio.open(path1) as img:
                        transform_1 = img.profile['transform']
                    with rasterio.open(path2) as img:
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
            finally:
                if (os.path.exists(path1)):
                    os.remove(path1)
                if (os.path.exists(path2)):
                        os.remove(path2)

    def test_create_tif_export(self):
        num_aggre = 5
        path = self.img_gen.random()
        try:
            with SlidingWindow(path) as slide_window:
                slide_window.dem_initialize_arrays()
                slide_window.dem_aggregation_step(num_aggre)
                new_path = slide_window.dem_export_arrays()
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
            if (os.path.exists(new_path)):
                os.remove(new_path)

    def test_aspect(self):
        path = self.img_gen.se_gradient()
        with SlidingWindow(path) as slide_window:
            slide_window.dem_initialize_arrays()
            slide_window.dem_aggregation_step(5)
            aspect_path = ''
            try:
                aspect_path = slide_window.dem_aspect()
                slope_angle_path = slide_window.dem_slope_angle()
                with rasterio.open(aspect_path) as img:
                    arr_aspect = img.read(1).astype(float)
                with rasterio.open(slope_angle_path) as img:
                    arr_slope_angle = img.read(1).astype(float)

                aspect_angle_perc = (7*math.pi/4)/(2*math.pi)
                aspect_value = aspect_angle_perc*np.iinfo(self.img_gen.dtype).max
                self.assertTrue(np.all(arr_aspect == math.floor(aspect_value)))

                slope_angle_perc = math.atan(2/math.sqrt(2))/(math.pi/2)
                slope_value = slope_angle_perc*np.iinfo(self.img_gen.dtype).max
                self.assertTrue(np.all(arr_slope_angle == math.floor(slope_value)))
            finally:
                if (os.path.exists(aspect_path)):
                    os.remove(aspect_path)
                if (os.path.exists(slope_angle_path)):
                    os.remove(slope_angle_path)

    def test_profile(self):
        image_size = 300
        sigma = image_size/4
        path = self.img_gen.gauss()
        with SlidingWindow(path) as slide_window:
            slide_window.dem_initialize_arrays()
            slide_window.dem_aggregation_step(5)
            arr_profile = slide_window._profile()
            for y in range(arr_profile.shape[0]):
                for x in range(arr_profile.shape[1]):
                    test_val = 2*math.pi*math.sqrt( (math.exp((2*(x**2+y**2))/sigma**2)*sigma**8*(-sigma**2+x**2+y**2)**2) / (2*math.exp((x**2+y**2)/sigma**2)*math.pi*sigma**6+x**2+y**2)**3 )
                    my_value = arr_profile[y][x]
                    math.isclose(my_value, test_val)
                

if __name__ == '__main__':
    unittest.main()
