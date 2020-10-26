import numpy as np

# get max and min of numpy data type
# returns tuple (max, min)
def dtype_max_min(dtype):
    max_val = 0
    min_val = 0
    if (np.issubdtype(dtype, np.floating)):
        max_val = np.finfo(dtype).max
        min_val = np.finfo(dtype).min
    else:
        max_val = np.iinfo(dtype).max
        min_val = np.iinfo(dtype).min

    return (max_val, min_val)

# get max and min of numpy data type
# returns tuple (max, min)
def dtype_max(dtype):
    max_val = 0
    if (np.issubdtype(dtype, np.floating)):
        max_val = np.finfo(dtype).max
    else:
        max_val = np.iinfo(dtype).max

    return max_val

# get max and min of numpy data type
# returns tuple (max, min)
def dtype_min(dtype):
    min_val = 0
    if (np.issubdtype(dtype, np.floating)):
        min_val = np.finfo(dtype).min
    else:
        min_val = np.iinfo(dtype).min

    return min_val