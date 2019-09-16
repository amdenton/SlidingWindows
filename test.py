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
        arr = img.read(1).astype(float)[0:1024, 0:1024]

        arr_good = slide_window._partial_aggregation(arr, 0, 6, 'sum')
        arr_brute = slide_window._aggregation_brute(arr, 'sum', 6)

        self.assertTrue(np.array_equal(arr_good, arr_brute))

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
