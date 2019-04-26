import numpy as np
import time

class SlidingWindow:
    def __init__(self, y_max, x_max, power_target):
        self.y_max = y_max
        self.x_max = x_max
        self.power_target = power_target

    # assumption: img_array is 2-dimensional
    # assumption: img_array is black and white, i.e. binary
    # TODO throw errors if assumptions are not met
    def analyze_number_one(self, img_array, operation):
        start = time.time() # TODO remove time
        y_max = img_array.shape[0]
        x_max = img_array.shape[1]
        array = np.array(img_array)
       
       # calculate sliding window for each value of delta
       # delta is in powers of 2
        for x in range(6):
            delta = 2**x
            array = array.flatten()
            size = array.shape[0]

            # create 4 offset arrays such that element wise operations operate on the four corners of a square of size delta+1
            # first array is truncated, second array is offset for top right corner, third for bottom left, fourth for bottom right
            # if that makes sense
            array1 = array[0: size - (delta*x_max + delta)]
            array2 = array[delta: size - (x_max*delta)]
            array3 = array[delta*x_max: size - (delta)]
            array4 = array[delta*x_max + delta: size]

            # operate on four arrays
            if operation.upper() == 'SUM':
                array = array1 + array2 + array3 + array4
            elif operation.upper() == 'MAX':
                array = np.maximum(np.maximum(np.maximum(array1, array2), array3), array4)
            else:
                raise Exception('not valid operation')
            
            # pad to reshape into two dimensional array
            array = np.pad(array, delta, 'constant')[delta:]
            array = np.reshape(array, (y_max-delta, x_max))
            # truncate last delta rows, which cannot create full delta+1 cubes
            array = np.delete(array, np.s_[-delta:], 1)

            # array has shrunk
            x_max -= delta
            y_max -= delta

        end = time.time() # TODO remove time
        print(end-start)
    
        return array

    # assumption: img_array is 2-dimensional
    # assumption: img_array is black and white, i.e. binary
    # TODO throw errors if assumptions are not met
    # .6 seconds faster than method one! (for a 170MB img)
    def analyze_number_two(self, img_array, operation):
        start = time.time()  # TODO remove time
        y_max = img_array.shape[0]
        x_max = img_array.shape[1]
        array = img_array.flatten()
        max_delta_power = 5
        
        # calculate sliding window for each value of delta
        # delta is in powers of 2
        for x in range(max_delta_power+1):
            delta = 2**x
            size = array.shape[0]

            # create 4 offset arrays such that element wise operations operate on the four corners of a square of size delta+1
            # first array is truncated, second array is offset for top right corner, third for bottom left, fourth for bottom right
            # if that makes sense
            array1 = array[0: size - (delta*x_max + delta)]
            array2 = array[delta: size - (x_max*delta)]
            array3 = array[delta*x_max: size - (delta)]
            array4 = array[delta*x_max + delta: size]

            # operate on four arrays
            if operation.upper() == 'SUM':
                array = array1 + array2 + array3 + array4
            elif operation.upper() == 'MAX':
                array = np.maximum(np.maximum(np.maximum(array1, array2), array3), array4)
            else:
                raise Exception('not valid operation')

            # array has shrunk
            y_max -= delta

        pad_num = 2**max_delta_power * 2 - 1
        # pad to reshape into two dimensional array
        array = np.pad(array, pad_num, 'constant')[pad_num:]
        array = np.reshape(array, (y_max, x_max))
        # truncate last pad_num rows, which are not valid
        array = np.delete(array, np.s_[-pad_num:], 1)

        end = time.time() # TODO remove time
        print(end-start)

        return array