import windowagg.rbg as rbg
import windowagg.dem as dem
import windowagg.aggregation as aggregation
from windowagg.dem_data import Dem_data
import windowagg.helper as helper

from enum import Enum
import os
import math
import shutil

import numpy as np
import affine
import rasterio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans

test_file_name = 'test.tif'
dir_prefix = ''
output_dir_postfix = 'clustered_image'

def path(file_path, postfix, file_extension=''):
    if ((file_extension != '') and (file_extension.index('.') != 0)):
        file_extension = '.' + file_extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    return dir_prefix + '_' + base_name + '_' + postfix + file_extension

def exportPath(file_path, band, num_aggre):
    return path(file_path, 'work') + '\\' + path(file_path, 'band=' + str(band) + '_w=' + str(2**num_aggre), '.npz')

class Analyses(Enum):
    regression = 1
    pearson = 2
    fractal = 3
    fractal_3d = 4
    slope = 5
    aspect = 6
    standard = 7

def gen_clustered_image(file_path, analyses, num_aggres, bands, num_clusters=3, sub_image_size=256, sub_image_start=[0, 0], map_width_to_meters=1.0, map_height_to_meters=1.0):

    with rasterio.open(file_path) as img:
        band_range = range(1, img.count + 1)
        max_band = 0
        for i in range(len(bands)):
            if (np.amax(bands[i]) > max_band):
                max_band = np.amax(bands[i])
        if (max_band not in band_range):
            raise ValueError('Band must be in range of %r.' % band_range)
        
        if (
            ((sub_image_start[0] + sub_image_size) >= img.read(1).shape[0]) or
            ((sub_image_start[1] + sub_image_size) >= img.read(1).shape[1])
        ):
            raise ValueError('Sub image doesn\'t fit within the original image')
        
        profile = img.profile
        transform = profile["transform"]
        x = transform[2] + ((transform[0] + transform[1]) * sub_image_start[1])
        y = transform[5] + ((transform[3] + transform[4]) * sub_image_start[0])
        transform = affine.Affine(transform[0], transform[1], x, transform[3] , transform[4], y)
        profile["transform"] = transform


        sub_img = np.empty((img.count, sub_image_size, sub_image_size)).astype(img.read(1).dtype)
        size = sub_image_size
        y_origin = sub_image_start[0]
        x_origin = sub_image_start[1]
        for band in band_range:
            sub_img[band - 1] = img.read(band)[y_origin:(y_origin + size), x_origin:(x_origin + size)].copy()

    maxNumAggre = np.amax(num_aggres)
    minNumAggre = np.amin(num_aggres)
    if (minNumAggre == 1):
        raise ValueError('Cannot aggregate images less than 2 times')

    try:
        if (not os.path.exists(path(file_path, 'work'))):
            os.makedirs(path(file_path, 'work'))
        else:
            raise ValueError('Work directory %s already exists' % path(file_path, 'work'))

        if (not os.path.exists(path(file_path, 'output'))):
            os.makedirs(path(file_path, 'output'))

        for band in band_range:
            dem_data = Dem_data(sub_img[band - 1])
            for num_aggre in range(maxNumAggre):
                aggregation.aggregate_dem(dem_data, 1)
                dem_data.export(exportPath(file_path, band, num_aggre + 1))

        removal_num = 2**maxNumAggre - 1
        output_length = sub_image_size - removal_num
        if (removal_num > sub_image_size):
            raise ValueError('Sub image size is too small to aggregate %s times' + str(maxNumAggre))
        adjust_file_path = path(file_path, 'output') + '\\' + path(file_path, 'adjusted', '.tif')
        adjusted_img = []
        for index in range(sub_img.shape[0]):
            adjusted_img_band = np.delete(np.delete(sub_img[index], np.s_[-removal_num::], 0), np.s_[-removal_num::], 1)
            adjusted_img_band = helper.arr_dtype_conversion(adjusted_img_band, np.uint8)
            adjusted_img.append(adjusted_img_band)
        helper.create_tif(adjusted_img, adjust_file_path, profile, maxNumAggre)
        print("Adjusted image saved to ", adjust_file_path)

        cluster_data = np.empty([(output_length)**2, 0 ]).astype(np.uint8)
        for index in range(len(analyses)):
            if analyses[index] is Analyses.regression:
                if ((not isinstance(bands[index], list)) or (len(bands[index]) != 2)):
                    raise ValueError('Unexpected band format. Regression analysis requires 2 bands')
                band1 = sub_img[bands[index][0] - 1]
                band2 = sub_img[bands[index][1] - 1]
                regression = rbg.regression(band1, band2, num_aggres[index])
                regression_removal_num = int((2**maxNumAggre - 2**num_aggres[index]) / 2)
                if (regression_removal_num > 0):
                    regression = regression[regression_removal_num:-regression_removal_num, regression_removal_num:-regression_removal_num]
                regression = helper.arr_dtype_conversion(regression, np.uint8)
                cluster_data = np.concatenate((cluster_data, regression.flatten()[:,np.newaxis]), 1)

            elif analyses[index] is Analyses.pearson:
                if ((not isinstance(bands[index], list)) or (len(bands[index]) != 2)):
                    raise ValueError('Unexpected band format. Pearson analysis requires 2 bands')
                band1 = sub_img[bands[index][0] - 1]
                band2 = sub_img[bands[index][1] - 1]
                pearson = rbg.pearson(band1, band2, num_aggres[index])
                pearson_removal_num = int((2**maxNumAggre - 2**num_aggres[index]) / 2)
                if (pearson_removal_num > 0):
                    pearson = pearson[pearson_removal_num:-pearson_removal_num, pearson_removal_num:-pearson_removal_num]
                pearson = helper.arr_dtype_conversion(pearson, np.uint8)
                cluster_data = np.concatenate((cluster_data, pearson.flatten()[:,np.newaxis]), 1)

            elif analyses[index] is Analyses.fractal:
                if (not isinstance(bands[index], int)):
                    raise ValueError('Unexpected band format. Fractal analysis requires 1 band')
                band = sub_img[bands[index] - 1]
                fractal = rbg.fractal(band, .5, num_aggres[index])
                fractal_removal_num = int((2**maxNumAggre - 2**num_aggres[index]) / 2)
                if (fractal_removal_num > 0):
                    fractal = fractal[fractal_removal_num:-fractal_removal_num, fractal_removal_num:-fractal_removal_num]
                fractal = helper.arr_dtype_conversion(fractal, np.uint8)
                cluster_data = np.concatenate((cluster_data, fractal.flatten()[:,np.newaxis]), 1)

            elif analyses[index] is Analyses.fractal_3d:
                if (not isinstance(bands[index], int)):
                    raise ValueError('Unexpected band format. Factal_3d analysis requires 1 band')
                band = sub_img[bands[index] - 1]
                fractal_3d = rbg.fractal_3d(band, num_aggres[index])
                fractal_3d_removal_num = int((2**maxNumAggre - 2**num_aggres[index]) / 2)
                if (fractal_3d_removal_num > 0):
                    fractal_3d = fractal_3d[fractal_3d_removal_num:-fractal_3d_removal_num, fractal_3d_removal_num:-fractal_3d_removal_num]
                fractal_3d = helper.arr_dtype_conversion(fractal_3d, np.uint8)
                cluster_data = np.concatenate((cluster_data, fractal_3d.flatten()[:,np.newaxis]), 1)

            elif analyses[index] is Analyses.slope:
                if (not isinstance(bands[index], int)):
                    raise ValueError('Unexpected band format. Factal_3d analysis requires 1 band')
                dem_data = Dem_data.from_import(exportPath(file_path, bands[index], num_aggres[index]))
                band = sub_img[bands[index] - 1]
                slope = dem.slope(dem_data, map_width_to_meters, map_height_to_meters)
                slope_removal_num = int((2**maxNumAggre - 2**num_aggres[index]) / 2)
                if (slope_removal_num > 0):
                    slope = slope[slope_removal_num:-slope_removal_num, slope_removal_num:-slope_removal_num]
                slope = helper.arr_dtype_conversion(slope, np.uint8)
                cluster_data = np.concatenate((cluster_data, slope.flatten()[:,np.newaxis]), 1)

            elif analyses[index] is Analyses.aspect:
                if (not isinstance(bands[index], int)):
                    raise ValueError('Unexpected band format. Factal_3d analysis requires 1 band')
                dem_data = Dem_data.from_import(exportPath(file_path, bands[index], num_aggres[index]))
                band = sub_img[bands[index] - 1]
                aspect = dem.aspect(dem_data)
                aspect_removal_num = int((2**maxNumAggre - 2**num_aggres[index]) / 2)
                if (aspect_removal_num > 0):
                    aspect = aspect[aspect_removal_num:-aspect_removal_num, aspect_removal_num:-aspect_removal_num]
                aspect = helper.arr_dtype_conversion(aspect, np.uint8)
                cluster_data = np.concatenate((cluster_data, aspect.flatten()[:,np.newaxis]), 1)

            elif analyses[index] is Analyses.standard:
                if (not isinstance(bands[index], int)):
                    raise ValueError('Unexpected band format. Factal_3d analysis requires 1 band')
                dem_data = Dem_data.from_import(exportPath(file_path, bands[index], num_aggres[index]))
                band = sub_img[bands[index] - 1]
                standard = dem.standard(dem_data, map_width_to_meters, map_height_to_meters)
                standard_removal_num = int((2**maxNumAggre - 2**num_aggres[index]) / 2)
                if (standard_removal_num > 0):
                    standard = standard[standard_removal_num:-standard_removal_num, standard_removal_num:-standard_removal_num]
                standard = helper.arr_dtype_conversion(standard, np.uint8)
                cluster_data = np.concatenate((cluster_data, standard.flatten()[:,np.newaxis]), 1)

        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(cluster_data)
        output = np.empty([(output_length)**2]).astype(np.uint8)
        step = 256 / (num_clusters - 1)
        for i in range(num_clusters):
            output[kmeans.labels_ == i] = kmeans.cluster_centers_[i].sum() / kmeans.cluster_centers_[i].size#  math.floor(step * i)
        output_file_path = path(file_path, 'output') + '\\' + path(file_path, 'output', '.tif')
        helper.create_tif(output.reshape([(output_length), (output_length)]), output_file_path, profile, maxNumAggre)
        print("Output image saved to ", output_file_path)

    finally:
        if os.path.exists(path(file_path, 'work')):
            shutil.rmtree(path(file_path, 'work'))


analyses = [Analyses.fractal, Analyses.fractal, Analyses.pearson]
num_aggres = [3, 3, 3]
bands = [2, 3, [2, 3]]

gen_clustered_image(test_file_name, analyses=analyses, num_aggres=num_aggres, bands=bands, num_clusters=10, sub_image_size=512)