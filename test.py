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
        arr = img.read(1)
        arr_vec = slide_window._aggregation(arr.astype(float), 'sum', 6)
        arr_brute = slide_window._aggregation_brute(arr.astype(float), 'sum', 6)
        self.assertTrue(np.array_equal(arr_vec, arr_brute))

if __name__ == '__main__':
    unittest.main()
