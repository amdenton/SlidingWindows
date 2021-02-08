import rbg as rbg
import dem as dem
import aggregation as aggregation
from dem_data import Dem_data
import helper as helper
import config as config

import math
import os

import numpy as np
import rasterio
import matplotlib.pyplot as plt

class SlidingWindow:

    # TODO potentially add R squared method?

    def __init__(self, file_path, map_width_to_meters=1.0, map_height_to_meters=1.0):
        file_basename = os.path.basename(file_path)
        self._file_name = os.path.splitext(file_basename)[0]
        self._img = rasterio.open(file_path)
        self._dem_data = None
        self.auto_plot = False
        self.work_dtype = config.work_dtype
        self.tif_dtype = config.tif_dtype
        self.convert_image = True

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
        with rasterio.open(file_name) as img:

            data = np.empty([img.shape[0], img.shape[1], img.count]).astype(np.uint8)
            for i in range(img.count):
                data[...,i] = helper.arr_dtype_conversion(img.read(i + 1), np.uint8)

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

        if (self.convert_image):
            ndvi = helper.arr_dtype_conversion(ndvi, self.tif_dtype)

        file_name = self._create_file_name('ndvi')
        helper.create_tif(ndvi, file_name, self._img.profile)

        if (self.auto_plot):
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

        if (self.convert_image):
            arr_m = helper.arr_dtype_conversion(arr_m, self.tif_dtype)

        file_name = self._create_file_name('regression', num_aggre)
        helper.create_tif(arr_m, file_name, self._img.profile, num_aggre)
        
        if (self.auto_plot):
            self._plot(file_name)

        return file_name

    # create image with pixel values cooresponding to their aggregated pearson correlation
    def pearson(self, band1, band2, num_aggre):
        bands = np.array(range(self._img.count)) + 1
        if (band1 not in bands or band2 not in bands):
            raise ValueError('bands must be in range of %r.' % bands)

        arr_a = self._img.read(band1)
        arr_b = self._img.read(band2)
        arr_r = rbg.pearson(arr_a, arr_b, num_aggre)

        if (self.convert_image):
            arr_r = helper.arr_dtype_conversion(arr_r, self.tif_dtype)

        file_name = self._create_file_name('pearson', num_aggre)
        helper.create_tif(arr_r , file_name, self._img.profile, num_aggre)

        if (self.auto_plot):
            self._plot(file_name)

        return file_name

    # create image with pixel values cooresponding to their aggregated fractal dimension
    def fractal(self, band, threshold, num_aggre):
        bands = np.array(range(self._img.count)) + 1
        if (band not in bands):
            raise ValueError('band must be in range of %r.' % bands)

        arr = self._img.read(band)
        arr = rbg.fractal(arr, threshold, num_aggre)

        if (self.convert_image):
            arr = helper.arr_dtype_conversion(arr, self.tif_dtype)

        file_name = self._create_file_name('fractal', num_aggre)
        helper.create_tif(arr, file_name, self._img.profile, num_aggre)

        if (self.auto_plot):
            self._plot(file_name)
        
        return file_name

    # create image with pixel values cooresponding to their aggregated 3D fractal dimension
    def fractal_3d(self, band, num_aggre):
        bands = np.array(range(self._img.count)) + 1
        if (band not in bands):
            raise ValueError('band must be in range of %r.' % bands)

        arr = self._img.read(band)
        arr = rbg.fractal_3d(arr, num_aggre)

        if (self.convert_image):
            arr = helper.arr_dtype_conversion(arr, self.tif_dtype)

        file_name = self._create_file_name('fractal_3d', num_aggre)
        helper.create_tif(arr, file_name, self._img.profile, num_aggre)

        if (self.auto_plot):
            self._plot(file_name)

        return file_name

    def import_dem(self, file_name):
        self._dem_data = Dem_data.from_import(file_name)

    def export_dem(self, file_name=None):
        if (self._dem_data is None):
            print('No DEM data to export')
        else:
            if (file_name == None):
                file_name = self._file_name + '_w=' + str(2**self._dem_data.num_aggre) + '.npz'
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

        if (self.convert_image):
            slope = helper.arr_dtype_conversion(slope, self.tif_dtype)

        file_name = self._create_file_name('slope', self._dem_data.num_aggre)
        helper.create_tif(slope, file_name, self._img.profile, self._dem_data.num_aggre)

        if (self.auto_plot):
            self._plot(file_name)

        return file_name

    # generate image of aggregated slope values
    def dem_slope_angle(self):
        if (self._dem_data is None):
            self.initialize_dem(1)

        slope_angle = np.arctan(dem.slope(self._dem_data, self.pixel_width, self.pixel_height))

        if (self.convert_image):
            slope_angle = helper.arr_dtype_conversion(slope_angle, self.tif_dtype)

        file_name = self._create_file_name('slope_angle', self._dem_data.num_aggre)
        helper.create_tif(slope_angle, file_name, self._img.profile, self._dem_data.num_aggre)

        if (self.auto_plot):
            self._plot(file_name)

        return file_name

    # generate image of aggregated angle of steepest descent, calculated as clockwise angle from north 
    def dem_aspect(self):
        if (self._dem_data is None):
            self.initialize_dem(1)

        aspect = dem.aspect(self._dem_data)

        if (self.convert_image):
            aspect = helper.arr_dtype_conversion(aspect, self.tif_dtype)

        file_name = self._create_file_name('aspect', self._dem_data.num_aggre)
        helper.create_tif(aspect, file_name, self._img.profile, self._dem_data.num_aggre)

        if (self.auto_plot):
            self._plot(file_name)

        return file_name

    # generate image of aggregated standard curvature
    def dem_standard(self):
        if (self._dem_data is None):
            self.initialize_dem(1)
        if (2**self._dem_data.num_aggre <= 2):
            print('Curvature cannot be calculated on windows of size 2 or 1')
            return

        standard = dem.standard(self._dem_data, self.pixel_width, self.pixel_height)

        if (self.convert_image):
            standard = helper.arr_dtype_conversion(standard, self.tif_dtype)
        
        file_name = self._create_file_name('standard', self._dem_data.num_aggre)
        helper.create_tif(standard, file_name, self._img.profile, self._dem_data.num_aggre)

        if (self.auto_plot):
            self._plot(file_name)

        return file_name