import windowagg.rbg as rbg
import windowagg.dem as dem
import windowagg.aggregation as aggregation
from windowagg.dem_data import Dem_data
from windowagg.agg_ops import Agg_ops
import windowagg.helper as helper
from windowagg.analyses import Analyses

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
import seaborn as sns
import pandas as pd

def _path(file_path, postfix, file_extension=''):
    if ((file_extension != '') and (file_extension.index('.') != 0)):
        file_extension = '.' + file_extension
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    return base_name + '_' + postfix + file_extension

def _demExportPath(file_path, band, num_aggre):
    return _path(file_path, 'work') + '\\' + _path(file_path, 'band=' + str(band) + '_w=' + str(2**num_aggre), '.npz')

def gen_clustered_img(file_path, analyses, num_aggres, bands, num_clusters=3, sub_img_size=256, sub_img_start=[0, 0], map_width_to_meters=1.0, map_height_to_meters=1.0, output_file_path=None, average_cluster_values=False):

    if (output_file_path is None):
        output_file_path = _path(file_path, 'output') + '\\' + _path(file_path, 'output', '.tif')

    band_range, profile, sub_img = _extract_img_info(file_path, bands, sub_img_start, sub_img_size)

    maxNumAggre = np.amax(num_aggres)
    if (np.amin(num_aggres) == 1):
        raise ValueError('Cannot aggregate images less than 2 times')

    try:
        if (not os.path.exists(_path(file_path, 'work'))):
            os.makedirs(_path(file_path, 'work'))
        else:
            raise ValueError('Work directory %s already exists' % _path(file_path, 'work'))

        if (not os.path.exists(_path(file_path, 'output'))):
            os.makedirs(_path(file_path, 'output'))

        if ((2**maxNumAggre - 1) > sub_img_size):
            raise ValueError('Sub image size is too small to aggregate %s times' + str(maxNumAggre))
        
        _export_dem_data(file_path, sub_img, band_range, maxNumAggre)
        _create_adjusted_img(file_path, sub_img, maxNumAggre, profile, np.uint8)
        _create_clustered_img(analyses, num_aggres, bands, sub_img, num_clusters, average_cluster_values, file_path, output_file_path, profile, map_width_to_meters, map_height_to_meters)

    finally:
        if os.path.exists(_path(file_path, 'work')):
            shutil.rmtree(_path(file_path, 'work'))

def gen_pairplot_img(file_path, analyses, num_aggres, bands, num_clusters=3, sub_img_size=256, sub_img_start=[0, 0], map_width_to_meters=1.0, map_height_to_meters=1.0, output_file_path=None):

    if (output_file_path is None):
        output_file_path = _path(file_path, 'output') + '\\' + _path(file_path, 'output', '.png')

    band_range, profile, sub_img = _extract_img_info(file_path, bands, sub_img_start, sub_img_size)

    maxNumAggre = np.amax(num_aggres)
    if (np.amin(num_aggres) == 1):
        raise ValueError('Cannot aggregate images less than 2 times')

    try:
        if (not os.path.exists(_path(file_path, 'work'))):
            os.makedirs(_path(file_path, 'work'))
        else:
            raise ValueError('Work directory %s already exists' % _path(file_path, 'work'))

        if (not os.path.exists(_path(file_path, 'output'))):
            os.makedirs(_path(file_path, 'output'))

        if ((2**maxNumAggre - 1) > sub_img_size):
            raise ValueError('Sub image size is too small to aggregate %s times' + str(maxNumAggre))
        
        _export_dem_data(file_path, sub_img, band_range, maxNumAggre)
        _create_adjusted_img(file_path, sub_img, maxNumAggre, profile, np.uint8)
        _create_pairplot_img(analyses, num_aggres, bands, sub_img, num_clusters, file_path, output_file_path, profile, map_width_to_meters, map_height_to_meters)

    finally:
        if os.path.exists(_path(file_path, 'work')):
            shutil.rmtree(_path(file_path, 'work'))

def _extract_img_info(file_path, bands, sub_img_start, sub_img_size):
    with rasterio.open(file_path) as img:
        band_range = range(1, img.count + 1)
        max_band = 0
        for i in range(len(bands)):
            if (np.amax(bands[i]) > max_band):
                max_band = np.amax(bands[i])
                
        if (max_band not in band_range):
            raise ValueError('Band must be in range of %r.' % band_range)
        
        if (
            ((sub_img_start[0] + sub_img_size) >= img.read(1).shape[0]) or
            ((sub_img_start[1] + sub_img_size) >= img.read(1).shape[1])
        ):
            raise ValueError('Sub image doesn\'t fit within the original image')
        
        profile = img.profile
        transform = profile["transform"]
        x = transform[2] + ((transform[0] + transform[1]) * sub_img_start[1])
        y = transform[5] + ((transform[3] + transform[4]) * sub_img_start[0])
        transform = affine.Affine(transform[0], transform[1], x, transform[3] , transform[4], y)
        profile["transform"] = transform


        sub_img = np.empty((img.count, sub_img_size, sub_img_size)).astype(img.read(1).dtype)
        size = sub_img_size
        y_origin = sub_img_start[0]
        x_origin = sub_img_start[1]
        for band in band_range:
            sub_img[band - 1] = img.read(band)[y_origin:(y_origin + size), x_origin:(x_origin + size)].copy()

    return (band_range, profile, sub_img)

def _export_dem_data(file_path, sub_img, band_range, num_aggre):
        # TODO only export required DEM data
        for band in band_range:
            dem_data = Dem_data(sub_img[band - 1])
            for i in range(num_aggre):
                aggregation.aggregate_dem(dem_data, 1)
                dem_data.export(_demExportPath(file_path, band, i + 1))


def _create_clustered_img(analyses, num_aggres, bands, sub_img, num_clusters, average_cluster_values, file_path, output_file_path, profile, map_width_to_meters, map_height_to_meters):
    cluster_data = _gen_cluster_data(analyses, num_aggres, bands, sub_img, file_path, map_width_to_meters, map_height_to_meters, np.uint8)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(cluster_data)
    maxNumAggre = np.amax(num_aggres)
    removal_num = 2**maxNumAggre - 1
    output_length = sub_img[0].shape[0] - removal_num
    output = np.empty([output_length**2]).astype(np.uint8)
    step = 256 / (num_clusters - 1)

    for i in range(num_clusters):
        if average_cluster_values:
            output[kmeans.labels_ == i] = kmeans.cluster_centers_[i].sum() / kmeans.cluster_centers_[i].size
        else:
            output[kmeans.labels_ == i] =  math.floor(step * i)

    helper.create_tif(output.reshape([(output_length), (output_length)]), output_file_path, profile, maxNumAggre)
    print("Output image saved to ", output_file_path)

def _create_pairplot_img(analyses, num_aggres, bands, sub_img, num_clusters, file_path, output_file_path, profile, map_width_to_meters, map_height_to_meters):
    labels = []
    for i in range(len(analyses)):
        concat_bands = ""
        if isinstance(bands[i], int):
            concat_bands = str(bands[i])
        else:
            for j in range(len(bands[i])):
                if (j != 0):
                    concat_bands += "+"
                concat_bands += str(bands[i][j])
        labels.append(str(analyses[i].name) + "_b" + concat_bands + "_w" + str(2**num_aggres[i]))

    cluster_data = _gen_cluster_data(analyses, num_aggres, bands, sub_img, file_path, map_width_to_meters, map_height_to_meters)
    data = pd.DataFrame(cluster_data, columns=labels)
    sns_plot = sns.pairplot(data, kind="hist")
    sns_plot.savefig(output_file_path)
    print("Output image saved to ", output_file_path)

def _create_adjusted_img(file_path, sub_img, num_aggre, profile, dtype=None):
        trun_num = int((2**num_aggre - 2) / 2)
        adjusted_file_path = _path(file_path, 'output') + '\\' + _path(file_path, 'adjusted', '.tif')
        adjusted_img = []

        for index in range(sub_img.shape[0]):
            arr = sub_img[index]

            old_dtype = arr.dtype
            adjusted_img_band = arr.astype(np.float64)
            adjusted_img_band = aggregation.aggregate(adjusted_img_band, Agg_ops.add_all, 1)
            adjusted_img_band = (adjusted_img_band / 4).astype(old_dtype)

            adjusted_img_band = adjusted_img_band[trun_num:-trun_num:1, trun_num:-trun_num:1]
            if (not dtype is None):
                adjusted_img_band = helper.arr_dtype_conversion(adjusted_img_band, dtype)
            
            adjusted_img.append(adjusted_img_band)

        helper.create_tif(adjusted_img, adjusted_file_path, profile, num_aggre)
        print("Adjusted image saved to ", adjusted_file_path)

def _gen_cluster_data(analyses, num_aggres, bands, sub_img, file_path, map_width_to_meters, map_height_to_meters, dtype=None):
    maxNumAggre = np.amax(num_aggres)
    removal_num = 2**maxNumAggre - 1
    cluster_data = np.empty([(sub_img[0].shape[0] - removal_num)**2, 0 ]).astype(np.uint8)

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
            dem_data = Dem_data.from_import(_demExportPath(file_path, bands[index], num_aggres[index]))
            analysis = dem.slope(dem_data, map_width_to_meters, map_height_to_meters)

        elif analyses[index] is Analyses.aspect:
            if (not isinstance(bands[index], int)):
                raise ValueError('Unexpected band format. Factal_3d analysis requires 1 band')
            dem_data = Dem_data.from_import(_demExportPath(file_path, bands[index], num_aggres[index]))
            analysis = dem.aspect(dem_data)

        elif analyses[index] is Analyses.standard:
            if (not isinstance(bands[index], int)):
                raise ValueError('Unexpected band format. Factal_3d analysis requires 1 band')
            dem_data = Dem_data.from_import(_demExportPath(file_path, bands[index], num_aggres[index]))
            analysis = dem.standard(dem_data, map_width_to_meters, map_height_to_meters)

        if (local_removal_num > 0):
            analysis = analysis[local_removal_num:-local_removal_num, local_removal_num:-local_removal_num]
        if (not dtype is None):
            analysis = helper.arr_dtype_conversion(analysis, dtype)
        cluster_data = np.concatenate((cluster_data, analysis.flatten()[:,np.newaxis]), 1)

    return cluster_data