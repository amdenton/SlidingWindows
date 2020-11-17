import windowagg.rbg as rbg
import windowagg.dem as dem
import windowagg.aggregation as aggregation
from windowagg.dem_data import Dem_data
import windowagg.helper as helper

import math
import os

import numpy as np
import rasterio
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

    def __init__(self, file_path, map_width_to_meters=1.0, map_height_to_meters=1.0):
        file_basename = os.path.basename(file_path)
        self._file_name = os.path.splitext(file_basename)[0]
        self._img = rasterio.open(file_path)
        self._dem_data = None
        self.autoPlot = False

        transform = self._img.profile['transform']
        self.pixel_width = math.sqrt(transform[0]**2 + transform[3]**2) * map_width_to_meters
        self.pixel_height = math.sqrt(transform[1]**2 + transform[4]**2) * map_height_to_meters

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

    def _create_file_name(self, algo_name, num_aggre=0):
        if (num_aggre == 0):
            file_name = self._file_name + '_' + algo_name + '.tif'
        else:
            file_name = self._file_name + '_' + algo_name + '_w=' + str(2**num_aggre) + '.tif'
        return file_name

    def _plot(self, file_name):
        img = rasterio.open(file_name)

        data = np.empty([img.shape[0], img.shape[1], img.count])
        for i in range(img.count):
            data[...,i] = img.read(i + 1)

        plt.imshow(data)
        plt.savefig(os.path.splitext(file_name)[0] + '.png')

    # create NDVI image
    def ndvi(self, red_band, ir_band):
        bands = np.array(range(self._img.count)) + 1
        if (red_band not in bands or ir_band not in bands):
            raise ValueError('bands must be in range of %r.' % bands)
        
        red = self._img.read(red_band)
        ir = self._img.read(ir_band)
        ndvi = rbg.ndvi(red, ir)

        # TODO change later?
        ndvi = helper.arr_dtype_conversion(ndvi, np.uint8)

        file_name = self._create_file_name('ndvi')
        helper.create_tif(ndvi, file_name, self._img.profile)

        if (self.autoPlot):
            self._plot(file_name)

        return file_name

    # create binary image
    def binary(self, band, threshold):
        bands = np.array(range(self._img.count)) + 1
        if (band not in bands):
            raise ValueError('band must be in range of %r.' % bands)

        arr = self._img.read(band)
        arr = rbg.binary(arr, threshold)
        
        # TODO change later?
        arr = helper.arr_dtype_conversion(arr, np.uint8)

        file_name = self._create_file_name('binary')
        helper.create_tif(arr, file_name, self._img.profile)

        if (self.autoPlot):
            self._plot(file_name)

        return file_name

    # create image with pixel values cooresponding to their aggregated regression slope
    def regression(self, band1, band2, num_aggre):
        bands = np.array(range(self._img.count)) + 1
        if (band1 not in bands or band2 not in bands):
            raise ValueError('bands must be in range of %r.' % bands)

        arr_a = self._img.read(band1)
        arr_b = self._img.read(band2)
        arr_m = rbg.regression(arr_a, arr_b, num_aggre)

        # TODO remove later
        arr_m = helper.arr_dtype_conversion(arr_m, np.uint8)

        file_name = self._create_file_name('regression', num_aggre)
        helper.create_tif(arr_m, file_name, self._img.profile, num_aggre)
        
        if (self.autoPlot):
            self._plot(file_name)

        return file_name

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
        arr_r = helper.arr_dtype_conversion(arr_r, np.uint8)

        file_name = self._create_file_name('pearson', num_aggre)
        helper.create_tif(arr_r, file_name, self._img.profile, num_aggre)

        if (self.autoPlot):
            self._plot(file_name)

        return file_name

    # create image with pixel values cooresponding to their aggregated fractal dimension
    def fractal(self, band, threshold, num_aggre):
        bands = np.array(range(self._img.count)) + 1
        if (band not in bands):
            raise ValueError('band must be in range of %r.' % bands)

        arr = self._img.read(band)
        arr = rbg.fractal(arr, threshold, num_aggre)

        # TODO remove later
        arr = helper.arr_dtype_conversion(arr, np.uint16)

        file_name = self._create_file_name('fractal', num_aggre)
        helper.create_tif(arr, file_name, self._img.profile, num_aggre)

        if (self.autoPlot):
            self._plot(file_name)
        
        return file_name

    def fractal_3d(self, band, num_aggre):
        bands = np.array(range(self._img.count)) + 1
        if (band not in bands):
            raise ValueError('band must be in range of %r.' % bands)

        arr = self._img.read(band)
        arr = rbg.fractal_3d(arr, num_aggre)

        # TODO remove later
        arr = helper.arr_dtype_conversion(arr, np.uint8)

        file_name = self._create_file_name('fractal_3d', num_aggre)
        helper.create_tif(arr, file_name, self._img.profile, num_aggre)

        if (self.autoPlot):
            self._plot(file_name)

        return file_name

    def import_dem(self, file_name):
        self._dem_data = Dem_data.from_import(file_name)

    def export_dem(self, file_name=None):
        if (self._dem_data is None):
            print('No DEM data to export')
        else:
            if (file_name == None):
                file_name = self._file_name + '_w=' + str(2**self._dem_data.num_aggre)
            self._dem_data.export(file_name)

    def initialize_dem(self, band=1):
        self._dem_data = Dem_data(self._img.read(band))

    def aggregate_dem(self, num_aggre=1):
        if (self._dem_data is None):
            self.initialize_dem(1)

        aggregation.aggregate_dem(self._dem_data, num_aggre)

    # generate image of aggregated slope values
    def dem_slope(self):
        if (self._dem_data is None):
            self.initialize_dem(1)

        slope = dem.slope(self._dem_data, self.pixel_width, self.pixel_height)
        slope = helper.arr_dtype_conversion(slope, np.uint16)

        file_name = self._create_file_name('slope', self._dem_data.num_aggre)
        helper.create_tif(slope, file_name, self._img.profile, self._dem_data.num_aggre)

        if (self.autoPlot):
            self._plot(file_name)

        return file_name

    # generate image of aggregated slope values
    def dem_slope_angle(self):
        if (self._dem_data is None):
            self.initialize_dem(1)

        slope_angle = dem.slope_angle(self._dem_data, self.pixel_width, self.pixel_height)
        slope_angle = helper.arr_dtype_conversion(slope_angle, dtype=np.uint16)

        file_name = self._create_file_name('slope_angle', self._dem_data.num_aggre)
        helper.create_tif(slope_angle, file_name, self._img.profile, self._dem_data.num_aggre)

        if (self.autoPlot):
            self._plot(file_name)

        return file_name

    # generate image of aggregated angle of steepest descent, calculated as clockwise angle from north 
    def dem_aspect(self):
        if (self._dem_data is None):
            self.initialize_dem(1)

        aspect = dem.aspect(self._dem_data)
        aspect = helper.arr_dtype_conversion(aspect, dtype=np.uint16)

        file_name = self._create_file_name('aspect', self._dem_data.num_aggre)
        helper.create_tif(aspect, file_name, self._img.profile, self._dem_data.num_aggre)

        if (self.autoPlot):
            self._plot(file_name)

        return file_name

    # generate image of aggregated profile curvature, second derivative parallel to steepest descent
    def dem_profile(self):
        if (self._dem_data is None):
            self.initialize_dem(1)

        profile = dem.profile(self._dem_data, self.pixel_width, self.pixel_height)
        profile = helper.arr_dtype_conversion(profile, np.uint16)

        file_name = self._create_file_name('profile', self._dem_data.num_aggre)
        helper.create_tif(profile, file_name, self._img.profile, self._dem_data.num_aggre)

        if (self.autoPlot):
            self._plot(file_name)

        return file_name

    # generate image of aggregated planform curvature, second derivative perpendicular to steepest descent
    def dem_planform(self):
        if (self._dem_data is None):
            self.initialize_dem(1)

        planform = dem.planform(self._dem_data, self.pixel_width, self.pixel_height)
        planform = helper.arr_dtype_conversion(planform, np.uint16)

        file_name = self._create_file_name('planform', self._dem_data.num_aggre)
        helper.create_tif(planform, file_name, self._img.profile, self._dem_data.num_aggre)

        if (self.autoPlot):
            self._plot(file_name)

        return file_name

    # generate image of aggregated standard curvature
    def dem_standard(self):
        if (self._dem_data is None):
            self.initialize_dem(1)

        standard = dem.standard(self._dem_data, self.pixel_width, self.pixel_height)
        standard = helper.arr_dtype_conversion(standard, np.uint16)
        
        file_name = self._create_file_name('standard', self._dem_data.num_aggre)
        helper.create_tif(standard, file_name, self._img.profile, self._dem_data.num_aggre)

        if (self.autoPlot):
            self._plot(file_name)

        return file_name