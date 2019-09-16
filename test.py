import unittest
from SlidingWindow import SlidingWindow
import BandEnum
import numpy as np
import rasterio

class TestSlidingWindow(unittest.TestCase):

    test_path = 'test.tif'

    def test_aggregation(self):
        slide_window = SlidingWindow(self.test_path, BandEnum.rgbIr)
        img = rasterio.open(self.test_path)
        arr = img.read(1).astype(float)[0:512, 0:512]

        arr_good_sum = slide_window._partial_aggregation(arr, 0, 7, 'sum')
        arr_good_max = slide_window._partial_aggregation(arr, 0, 7, 'max')

        arr_good_partial_sum = slide_window._partial_aggregation(arr, 0, 2, 'sum')
        arr_good_partial_sum = slide_window._partial_aggregation(arr_good_partial_sum, 2, 3, 'sum')
        arr_good_partial_sum = slide_window._partial_aggregation(arr_good_partial_sum, 3, 7, 'sum')
        arr_good_partial_max = slide_window._partial_aggregation(arr, 0, 2, 'max')
        arr_good_partial_max = slide_window._partial_aggregation(arr_good_partial_max, 2, 3, 'max')
        arr_good_partial_max = slide_window._partial_aggregation(arr_good_partial_max, 3, 7, 'max')

        arr_brute_sum = slide_window._aggregation_brute(arr, 'sum', 7)
        arr_brute_max = slide_window._aggregation_brute(arr, 'max', 7)

        self.assertTrue(np.array_equal(arr_good_sum, arr_brute_sum))
        self.assertTrue(np.array_equal(arr_good_max, arr_brute_max))
        self.assertTrue(np.array_equal(arr_good_partial_sum, arr_brute_sum))
        self.assertTrue(np.array_equal(arr_good_partial_max, arr_brute_max))

    def test_regression(self):
        slide_window = SlidingWindow(self.test_path, BandEnum.rgbIr)
        img = rasterio.open(self.test_path)
        arr1 = img.read(1).astype(float)[0:128, 0:128]
        arr2 = img.read(2).astype(float)[0:128, 0:128]

        arr_good = slide_window._regression(arr1, arr2, 6)
        arr_good = np.round(arr_good, 10)
        arr_brute = slide_window._regression_brute(arr1, arr2, 6)
        arr_brute = np.round(arr_brute, 10)

        self.assertTrue(np.array_equal(arr_good, arr_brute))

if __name__ == '__main__':
    unittest.main()
