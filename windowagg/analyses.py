from enum import Enum

class Analyses(Enum):
    ndvi = 1
    regression = 2
    pearson = 3
    fractal = 4
    fractal_3d = 5
    slope = 6
    aspect = 7
    standard = 8