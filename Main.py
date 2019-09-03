import numpy as np
from SlidingWindow import SlidingWindow
from BandEnum import rgbIr

slide_window = SlidingWindow('m_4509601_ne_14_1_20120705.tif', rgbIr)

print('VEC SLIDING WINDOW')
imgGood = slide_window.sliding_window_vec('red', 'sum', 5)
print('BAD SLIDING WINDOW')
imgBad = slide_window.sliding_window_bad('red', 'sum', 5)
print('EQUAL? ', np.array_equal(imgGood, imgBad))