import numpy as np
import time

class SlidingWindow:
    def __init__(self, max_delta_power):
        self.max_delta_power = max_delta_power

    # assumption: img_array is 2-dimensional
    # TODO throw errors if assumptions are not met
    def analyze_bad(self, img_array, operation):
        if (operation.upper() != 'SUM' and operation.upper() != 'MAX'):
            raise Exception('not valid operation')

        start = time.time()  # TODO remove time
        y_max = img_array.shape[0]
        x_max = img_array.shape[1]
        array = np.array(img_array).astype(float) # TODO change dtype later?

        for i in range(self.max_delta_power+1):
            delta = 2**i
            y_max -= delta
            x_max -= delta

            new_array = np.zeros((y_max, x_max), dtype=array.dtype)
            for y in range(y_max):
                for x in range(x_max):
                    if (operation.upper() == 'SUM'):
                        new_array[y, x] = array[y, x] + array[y, x+delta] + array[y+delta, x] + array[y+delta, x+delta]
                    elif (operation.upper() == 'MAX'):
                        new_array[y, x] = np.maximum(np.maximum(np.maximum(array[y, x], array[y, x+delta]), array[y+delta, x]), array[y+delta, x+delta])
            array = new_array

        end = time.time() # TODO remove time
        print('BAD TIME: ', end-start)

        return array


    # assumption: img_array is 2-dimensional
    # TODO throw errors if assumptions are not met
    def analyze(self, img_array, operation):
        if (operation.upper() != 'SUM' and operation.upper() != 'MAX'):
            raise Exception('not valid operation')
        
        start = time.time()  # TODO remove time
        y_max = img_array.shape[0]
        x_max = img_array.shape[1]
        array = img_array.flatten().astype(float) # TODO change dtype later?
        
        # calculate sliding window for each value of delta
        for x in range(self.max_delta_power+1):
            delta = 2**x
            size = array.shape[0]

            # create offset slices of the array to aggregate elements
            arrTopLeft = array[0: size - (delta*x_max + delta)]
            arrTopRight = array[delta: size - (x_max*delta)]
            arrBottomLeft = array[delta*x_max: size - (delta)]
            arrBottomRight = array[delta*x_max + delta: size]

            # operate on arrays
            if operation.upper() == 'SUM':
                array = arrTopLeft + arrTopRight + arrBottomLeft + arrBottomRight
            elif operation.upper() == 'MAX':
                array = np.maximum(np.maximum(np.maximum(arrTopLeft, arrTopRight), arrBottomLeft), arrBottomRight)

        # number of rows and columns removed from the ends of the array
        pad_num = 2**(self.max_delta_power+1) - 1
        # truncate last pad_num rows
        y_max -= pad_num
        # pad to make array square
        array = np.pad(array, (0, pad_num), 'constant')
        array = np.reshape(array, (y_max, x_max))
        # truncate last pad_num columns
        array = np.delete(array, np.s_[-pad_num:], 1)

        end = time.time() # TODO remove time
        print('GOOD TIME: ', end-start)

        return array