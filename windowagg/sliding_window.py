import windowagg.rbg as rbg
import windowagg.dem as dem
import windowagg.aggregation as aggregation
from windowagg.dem_data import Dem_data
import windowagg.helper as helper

import math
import os
import inspect

import numpy as np
import rasterio
import affine
import matplotlib.pyplot as plt

class SlidingWindow:

    # TODO create more tests
    # test geoTransform update in _create_tif()
    # test averages (e.g. xz, yz) in DEM_UTILS

    # TODO do I have all required functionality?
    # image creation with geoTransform update
    # array conversion TODO needs updating
    # ndvi
    # binary
    # regression
    # Pearson
    # fractal
    # fractal 3D TODO ensure this method is written properly (check _boxed_array() for sure)
    # DEM window mean
    # DEM array intialization
    # DEM double_w TODO this has 3 methods, do we need all 3?
    # DEM slope
    # DEM aspect
    # DEM standard curve
    # DEM profile curve
    # DEM planform curve

    # TODO research how to create python package
    # TODO add more documentation
    # TODO should all these methods use floating point? datatypes?!?!

    def __init__(self, file_path, cell_width=1, cell_height=1):
        self._file_name = os.path.split(file_path)[-1]
        self._img = rasterio.open(file_path)
        self._dem_data = None

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, traceback):
        if (self._img):
            self._img.close()
    def close(self):
        if (self._img):
            self._img.close()
    def __del__(self):
        if (self._img):
            self._img.close()

    # TODO fix later, not the best way to do this
    # arr_in: array to be converted
    # dtype: numpy type to convert to
    @staticmethod
    def _arr_dtype_conversion(arr_in, dtype=np.uint16, low_bound=None, high_bound=None):
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

        dtype_max = helper.dtype_max(dtype)
        arr_out = ((arr_in - low_bound)/(high_bound - low_bound)*dtype_max).astype(dtype)
        return arr_out

    # create tif with array of numpy arrays representing image bands
    # adjust geoTransform according to how many pixels were aggregated
    def _create_tif(self, arr_in, file_name, num_aggre=0):
        if (type(arr_in) != list):
            arr_in = [arr_in]

        dtype = arr_in[0].dtype
        shape = arr_in[0].shape
        for i in range(1, len(arr_in)):
            if (arr_in[i].dtype != dtype):
                raise ValueError('arrays must have the same dtype')
            if (arr_in[i].shape != shape):
                raise ValueError('arrays must have the same shape')

        profile = self._img.profile
        transform = profile["transform"]

        num_trunc = (2**num_aggre - 1)
        img_offset = num_trunc / 2

        x = transform[2] + ((transform[0] + transform[1]) * img_offset)
        y = transform[5] + ((transform[3] + transform[4]) * img_offset)
        transform = affine.Affine(transform[0], transform[1], x, transform[3] , transform[4], y)

        big_tiff = 'NO'
        n_bytes = 0
        gigabyte = 1024**3
        for i in range(len(arr_in)):
            n_bytes += arr_in[i].nbytes
        if (n_bytes > (2 * gigabyte)):
            big_tiff = 'YES'

        # TODO should nodata be 0?
        profile.update(
            nodata=0,
            dtype=dtype,
            count=len(arr_in),
            height=len(arr_in[0]),
            width=len(arr_in[0][0])
            )
            
        with rasterio.open(file_name, 'w', **profile, BIGTIFF=big_tiff) as dst:
            for i in range(len(arr_in)): 
                dst.write(arr_in[i], i + 1)

        return file_name

    def _create_file_name(self, algo_name, num_aggre=0):
        if (num_aggre == 0):
            file_name = self._file_name + '_' + algo_name + '.tif'
        else:
            file_name = self._file_name + '_' + algo_name + '_w=' + str(2**num_aggre) + '.tif'
        return file_name

    # create NDVI image
    def ndvi(self, red_band, ir_band):
        bands = np.array(range(self._img.count)) + 1
        if (red_band not in bands or ir_band not in bands):
            raise ValueError('bands must be in range of %r.' % bands)
        
        red = self._img.read(red_band)
        ir = self._img.read(ir_band)
        ndvi = rbg.ndvi(red, ir)

        # TODO change later?
        ndvi = self._arr_dtype_conversion(ndvi, np.uint8)

        file_name = self._create_file_name('ndvi')
        return self._create_tif(ndvi, file_name)

    # create binary image
    def binary(self, band, threshold):
        bands = np.array(range(self._img.count)) + 1
        if (band not in bands):
            raise ValueError('band must be in range of %r.' % bands)

        arr = self._img.read(band)
        arr = rbg.binary(arr, threshold)
        
        # TODO change later?
        arr = self._arr_dtype_conversion(arr, np.uint8)

        file_name = self._create_file_name('binary')
        return self._create_tif(arr, file_name)

    # create image with pixel values cooresponding to their aggregated regression slope
    def regression(self, band1, band2, num_aggre):
        bands = np.array(range(self._img.count)) + 1
        if (band1 not in bands or band2 not in bands):
            raise ValueError('bands must be in range of %r.' % bands)

        arr_a = self._img.read(band1)
        arr_b = self._img.read(band2)
        arr_m = rbg.regression(arr_a, arr_b, num_aggre)

        # TODO remove later
        arr_m = self._arr_dtype_conversion(arr_m, np.uint8)

        file_name = self._create_file_name('regression', num_aggre)
        return self._create_tif(arr_m, file_name, num_aggre)

    # TODO potentially add R squared method?

    # create image with pixel values cooresponding to their aggregated pearson correlation
    def pearson(self, band1, band2, num_aggre):
        bands = np.array(range(self._img.count)) + 1
        if (band1 not in bands or band2 not in bands):
            raise ValueError('bands must be in range of %r.' % bands)

        arr_a = self._img.read(band1)
        arr_b = self._img.read(band2)
        arr_r = rbg.pearson(arr_a, arr_b, num_aggre)

        # TODO remove later
        arr_r = self._arr_dtype_conversion(arr_r, np.uint8)

        file_name = self._create_file_name('pearson', num_aggre)
        return self._create_tif(arr_r, file_name, num_aggre)

    # create image with pixel values cooresponding to their aggregated fractal dimension
    def fractal(self, band, threshold, power_start, power_target):
        bands = np.array(range(self._img.count)) + 1
        if (band not in bands):
            raise ValueError('band must be in range of %r.' % bands)

        arr = self._img.read(band)
        arr = rbg.fractal(rbg.binary(arr, threshold), power_start, power_target)

        # TODO remove later
        arr = self._arr_dtype_conversion(arr, np.uint16)

        #file_name = self._create_file_name('fractal', num_aggre)
        return self._create_tif(arr, power_target)

    def fractal_3d(self, band, num_aggre):
        bands = np.array(range(self._img.count)) + 1
        if (band not in bands):
            raise ValueError('band must be in range of %r.' % bands)

        arr = self._img.read(band)
        arr = rbg.fractal_3d(arr, num_aggre)

        # TODO remove later
        arr = self._arr_dtype_conversion(arr, np.uint8)

        return self._create_tif(arr, num_aggre)

    def import_dem(self, file_name):
        self._dem_data = Dem_data.from_import(file_name)

    def export_dem(self, file_name):
        if (self._dem_data is None):
            print('No DEM data to export')
        else:
            self._dem_data.export(file_name)

    def initializ_dem(self, band):
        self._dem_data = Dem_data(self._img.read(band))

    def aggregate_dem(self, num_aggre=1):
        if (Dem_data is None):
            self.initializ_dem(0)

        aggregation.aggregate_dem(self._dem_data, num_aggre)

    # generate image of aggregated slope values
    def dem_slope(self):
        if (Dem_data is None):
            self.initializ_dem(0)

        slope = dem.slope(self._dem_data)
        slope = self._arr_dtype_conversion(slope, np.uint16, low_bound=0, high_bound=np.iinfo(np.uint16).max)

        file_name = self._create_file_name('slope', self._dem_data.num_aggre)
        return self._create_tif(slope, file_name, self._dem_data.num_aggre)

    # generate image of aggregated slope values
    def dem_slope_angle(self):
        if (Dem_data is None):
            self.initializ_dem(0)

        slope_angle = dem.slope_angle(self._dem_data)
        slope_angle = self._arr_dtype_conversion(slope_angle, dtype=np.uint16, low_bound=0, high_bound=math.pi/2)

        file_name = self._create_file_name('slope_angle', self._dem_data.num_aggre)
        return self._create_tif(slope_angle, file_name, self._dem_data.num_aggre)

    # generate image of aggregated angle of steepest descent, calculated as clockwise angle from north 
    def dem_aspect(self):
        if (Dem_data is None):
            self.initializ_dem(0)

        aspect = dem.aspect(self._dem_data)
        aspect = self._arr_dtype_conversion(aspect, dtype=np.uint16, low_bound=0, high_bound=(2 * math.pi))

        file_name = self._create_file_name('aspect', self._dem_data.num_aggre)
        return self._create_tif(aspect, file_name, self._dem_data.num_aggre)

    # generate image of aggregated profile curvature, second derivative parallel to steepest descent
    def dem_profile(self):
        if (Dem_data is None):
            self.initializ_dem(0)

        profile = dem.profile(self._dem_data)
        profile = self._arr_dtype_conversion(profile, np.uint16)

        file_name = self._create_file_name('profile', self._dem_data.num_aggre)
        return self._create_tif(profile, file_name, self._dem_data.num_aggre)

    # generate image of aggregated planform curvature, second derivative perpendicular to steepest descent
    def dem_planform(self):
        if (Dem_data is None):
            self.initializ_dem(0)

        planform = dem.planform(self._dem_data)
        planform = self._arr_dtype_conversion(planform, np.uint16)

        file_name = self._create_file_name('planform', self._dem_data.num_aggre)
        return self._create_tif(planform, file_name, self._dem_data.num_aggre)

    # generate image of aggregated standard curvature
    def dem_standard(self):
        if (Dem_data is None):
            self.initializ_dem(0)

        standard = dem.standard(self._dem_data)
        standard = self._arr_dtype_conversion(standard, np.uint16)
        
        file_name = self._create_file_name('standard', self._dem_data.num_aggre)
        return self._create_tif(standard, file_name, self._dem_data.num_aggre)