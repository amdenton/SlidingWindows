import numpy as np
from Raster import Raster
from SlidingWindow import SlidingWindow
import matplotlib.pyplot as plt

from BandEnum import BandEnum

rast = Raster("m_4509601_ne_14_1_20120705.tif")
slide_window = SlidingWindow(5)

img = np.random.randint(0, 2, (7700, 5650), np.uint8)
# img = rast.img.read(BandEnum.red.value)
# img = rast.create_binary_image(img, 127)
print('HEIGHT: ', img.shape[0])
print('WIDTH: ', img.shape[1])
imgGood = slide_window.analyze(img, 'sum')
imgBad = slide_window.analyze_bad(img, 'sum')
print('EQUAL? ', np.array_equal(imgGood, imgBad))