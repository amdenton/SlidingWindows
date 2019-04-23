import numpy as np

class SlidingWindow:
    def __init__(self, y_max, x_max, power_target):
        self.y_max = y_max
        self.x_max = x_max
        self.power_target = power_target

    def sum(self, img_array):
        return None

    def create_binary_image(self, img, threshold):
        binary_image = np.empty([self.y_max, self.x_max])
        for y in range(self.y_max):
            for x in range(self.x_max):
                binary_image[y][x] = (img[y][x] >= threshold)
        return binary_image
        
        
