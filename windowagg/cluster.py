import rbg as rbg
import dem as dem
import aggregation as aggregation
from dem_data import Dem_data
import helper as helper
from analyses import Analyses

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

def path(file_path, postfix, file_extension=''):
    if ((file_extension != '') and (file_extension.index('.') != 0)):
        file_extension = '.' + file_extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    return base_name + '_' + postfix + file_extension

def demExportPath(file_path, band, num_aggre):
    return path(file_path, 'work') + '\\' + path(file_path, 'band=' + str(band) + '_w=' + str(2**num_aggre), '.npz')

def gen_clustered_image(file_path, analyses, num_aggres, bands, num_clusters=3, sub_image_size=256, sub_image_start=[0, 0], map_width_to_meters=1.0, map_height_to_meters=1.0, output_file_path=None, average_cluster_values=False):

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

        # TODO only export required DEM data
        for band in band_range:
            dem_data = Dem_data(sub_img[band - 1])
            for num_aggre in range(maxNumAggre):
                aggregation.aggregate_dem(dem_data, 1)
                dem_data.export(demExportPath(file_path, band, num_aggre + 1))

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
            local_removal_num = int((2**maxNumAggre - 2**num_aggres[index]) / 2)

            if analyses[index] is Analyses.ndvi:
                if ((not isinstance(bands[index], list)) or (len(bands[index]) != 2)):
                    raise ValueError('Unexpected band format. NDVI requires 2 bands')
                if (num_aggres[index] != 0):
                    local_removal_num = int((2**maxNumAggre - 1) / 2)
                    raise Warning('NDVI does not aggregate')
                analysis = rbg.ndvi(sub_img[bands[index][0] - 1], sub_img[bands[index][1] - 1])

            elif analyses[index] is Analyses.regression:
                if ((not isinstance(bands[index], list)) or (len(bands[index]) != 2)):
                    raise ValueError('Unexpected band format. Regression analysis requires 2 bands')
                analysis = rbg.regression(sub_img[bands[index][0] - 1], sub_img[bands[index][1] - 1], num_aggres[index])

            elif analyses[index] is Analyses.pearson:
                if ((not isinstance(bands[index], list)) or (len(bands[index]) != 2)):
                    raise ValueError('Unexpected band format. Pearson analysis requires 2 bands')
                analysis = rbg.pearson(sub_img[bands[index][0] - 1], sub_img[bands[index][1] - 1], num_aggres[index])

            elif analyses[index] is Analyses.fractal:
                # TODO make threshold variable
                analysis = rbg.fractal(sub_img[bands[index] - 1], .5, num_aggres[index])

            elif analyses[index] is Analyses.fractal_3d:
                analysis = rbg.fractal_3d(sub_img[bands[index] - 1], num_aggres[index])

            elif analyses[index] is Analyses.slope:
                if (not isinstance(bands[index], int)):
                    raise ValueError('Unexpected band format. Factal_3d analysis requires 1 band')
                dem_data = Dem_data.from_import(demExportPath(file_path, bands[index], num_aggres[index]))
                analysis = dem.slope(dem_data, map_width_to_meters, map_height_to_meters)

            elif analyses[index] is Analyses.aspect:
                if (not isinstance(bands[index], int)):
                    raise ValueError('Unexpected band format. Factal_3d analysis requires 1 band')
                dem_data = Dem_data.from_import(demExportPath(file_path, bands[index], num_aggres[index]))
                analysis = dem.aspect(dem_data)

            elif analyses[index] is Analyses.standard:
                if (not isinstance(bands[index], int)):
                    raise ValueError('Unexpected band format. Factal_3d analysis requires 1 band')
                dem_data = Dem_data.from_import(demExportPath(file_path, bands[index], num_aggres[index]))
                analysis = dem.standard(dem_data, map_width_to_meters, map_height_to_meters)

            if (local_removal_num > 0):
                analysis = analysis[local_removal_num:-local_removal_num, local_removal_num:-local_removal_num]
            analysis = helper.arr_dtype_conversion(analysis, np.uint8)
            cluster_data = np.concatenate((cluster_data, analysis.flatten()[:,np.newaxis]), 1)

        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(cluster_data)
        output = np.empty([(output_length)**2]).astype(np.uint8)
        step = 256 / (num_clusters - 1)
        for i in range(num_clusters):
            if average_cluster_values:
                output[kmeans.labels_ == i] = kmeans.cluster_centers_[i].sum() / kmeans.cluster_centers_[i].size
            else:
                output[kmeans.labels_ == i] =  math.floor(step * i)
        if (output_file_path is None):
            output_file_path = path(file_path, 'output') + '\\' + path(file_path, 'output', '.tif')
        helper.create_tif(output.reshape([(output_length), (output_length)]), output_file_path, profile, maxNumAggre)
        print("Output image saved to ", output_file_path)

    finally:
        if os.path.exists(path(file_path, 'work')):
            shutil.rmtree(path(file_path, 'work'))


analyses = [Analyses.standard]
num_aggres = [7]
bands = [3]

for i in range(2,8):
    output_file_path = path(test_file_name, 'output') + "\\standard_blue_numAggre=" + str(i) + ".tif"
    num_aggres = [i]
    gen_clustered_image(test_file_name, analyses=analyses, num_aggres=num_aggres, bands=bands, num_clusters=10, sub_image_size=512, output_file_path=output_file_path, average_cluster_values=True)