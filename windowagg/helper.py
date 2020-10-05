import numpy as np

# get max and min of numpy data type
# returns tuple (max, min)
def get_max_min(dtype):
    max_val = 0
    min_val = 0
    if (np.issubdtype(dtype, np.floating)):
        max_val = np.finfo(dtype).max
        min_val = np.finfo(dtype).min
    else:
        max_val = np.iinfo(dtype).max
        min_val = np.iinfo(dtype).min

    return (max_val, min_val)