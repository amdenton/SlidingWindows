"""
Last updated on Tue Dec 14

@authors: Anne Denton, David Schwarz, Rahul Gomes

License information:
https://opensource.org/licenses/GPL-3.0
"""
import numpy as np
from affine import Affine
import matplotlib.pyplot as plt
import rasterio
import copy
import os
from rasterio.windows import Window

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

def plot(file_name):
    with rasterio.open(file_name) as img:
        if img.count > 1:
            data = np.empty([img.shape[0], img.shape[1], img.count]).astype(np.uint8)
            for i in range(img.count):
                data[...,i] = arr_dtype_conversion(img.read(i + 1), np.uint8)

            plt.figure()
            plt.imshow(data)
            plt.savefig(os.path.splitext(file_name)[0] + '.png')
        else:
            data = img.read(1)
            
            plt.figure()
#            plt.title(file_name)
            shift = int(round(data.shape[0]/50))
            plt.text(0,-4*shift,file_name)
            plt.text(0,-shift,'min: '+str(round(np.amin(data),3))+' max: '+str(round(np.amax(data),3)))
            plt.imshow(data,'gray_r')

# create tif with array of numpy arrays representing image bands
# adjust geoTransform according to how many pixels were aggregated
def create_tif(arr_in, file_name, profile=None, num_aggre=0, pixelSpatialResolution = (1,1)):
    dtype = np.dtype(arr_in[0,0])
    if (profile == None):
        geotransform = (500000, 1.0, 0.0, 5000000.0, 0.0, -1.0)
        transform = Affine.from_gdal(*geotransform)
        profile = {
            'driver': 'GTiff',
            'crs': 'EPSG:26914',# Same as for Landscape file
            'dtype': dtype,
            'transform': transform,
            'count': 1,
            'height': np.size(arr_in,0),
            'width': np.size(arr_in,1)
        }
    old_transform = profile['transform']
    num_trunc = (2**num_aggre - 1)
    img_offset = num_trunc / 2
    new_transform = Affine.translation(img_offset * pixelSpatialResolution[0], img_offset  * pixelSpatialResolution[1]) * old_transform
    new_profile = copy.deepcopy(profile)

    big_tiff = 'NO'
    n_bytes = 0
    gigabyte = 1024**3
    for i in range(len(arr_in)):
        n_bytes += arr_in[i].nbytes
    if (n_bytes > (2 * gigabyte)):
        big_tiff = 'YES'

    new_profile.update({
        'dtype': dtype,
        'count': 1,
        'height': np.size(arr_in,0),
        'width': np.size(arr_in,1),
        'transform': new_transform,
    })
    with rasterio.open(file_name, 'w', **new_profile, BIGTIFF=big_tiff) as dst:
        print('Writing to: ',dst)
        print('with transform: ')
        print(new_transform)
        dst.write(arr_in,indexes=1)

# TODO fix later, not the best way to do this
# arr_in: array to be converted
# dtype: numpy type to convert to
def arr_dtype_conversion(arr_in, dtype=np.uint16, low_bound=None, high_bound=None):
    arr_max = np.amax(arr_in)
    arr_min = np.amin(arr_in)
    if (low_bound == None):
        low_bound = arr_min
    else:
        if (arr_min < low_bound):
            raise ValueError('Lower bound must be smaller than all values')
    if (high_bound == None):
        high_bound = arr_max
    else:
        if (arr_max > high_bound):
            raise ValueError('Upper bound must be greater than all values')

    new_dtype_max = dtype_max(dtype)
    if (high_bound == low_bound):
        original_dtype_max = dtype_max(arr_in.dtype)
        arr_out = (arr_in / original_dtype_max * new_dtype_max).astype(dtype)
    else:
        arr_out = ((arr_in - low_bound) / (high_bound - low_bound) * new_dtype_max).astype(dtype)
    
    return arr_out

def trim_tiff(filepath, size_width_and_height):
    with rasterio.open(filepath) as src:
        window = Window(0, 0, size_width_and_height, size_width_and_height)
        kwargs = src.meta.copy()
        kwargs.update({
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, src.transform)})

        with rasterio.open('cropped.tif', 'w', **kwargs) as dst:
            dst.write(src.read(window=window))