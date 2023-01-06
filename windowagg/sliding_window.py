"""
Last updated on Tue Dec 14

@authors: Anne Denton, David Schwarz, Rahul Gomes

License information:
https://opensource.org/licenses/GPL-3.0
"""
import windowagg.dem as dem
import windowagg.aggregation as aggregation
from windowagg.dem_data import Dem_data
import windowagg.helper as helper
import windowagg.config as config
from osgeo import gdal
import os
import rasterio


class SlidingWindow:

    # TODO potentially add R squared method?

    def __init__(self, file_path, map_width_to_meters=1.0, map_height_to_meters=1.0, save_jpg=False):
        self._file_name = os.path.splitext(file_path)[0]
        self._img = rasterio.open(file_path)
        self._orig_profile = self._img.profile
        self._dem_data = None
        self.auto_plot = False
        self.work_dtype = config.work_dtype
        self.tif_dtype = config.tif_dtype
        self._jpg_options = "-ot Byte -of JPEG -b 1 -scale"
        self.save_jpg = save_jpg
        self.getPixelSpatialResolution()

#       Algorith is derived using assumption of square windows
#        transform = self._img.profile['transform']
#        self.pixel_width = math.sqrt(transform[0]**2 + transform[3]**2) * map_width_to_meters
#        self.pixel_height = math.sqrt(transform[1]**2 + transform[4]**2) * map_height_to_meters

        if (self.auto_plot):
            helper.plot(file_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if (self._img):
            self.close()

    def close(self):
        if (self._img):
            try:
                self._img.close()
                if(self.save_jpg):
                    gdal.Translate(
                        f'{self._file_name}.jpg',
                        f'{self._file_name}.tif',
                        options=self._jpg_options
                    )
            except:
                x = 1
                # Failed to close

    def __del__(self):
        if (self._img):
            self.close()

    def _create_file_name(self, algo_name, num_aggre=0):
        if (num_aggre == 0):
            file_name = self._file_name + '_' + algo_name + '.tif'
        else:
            file_name = self._file_name + '_' + algo_name + \
                '_w=' + str(2**num_aggre) + '.tif'
        return file_name

    def import_dem(self, file_name):
        self._dem_data = Dem_data.from_import(file_name)

    def export_dem(self, file_name=None):
        if (self._dem_data is None):
            print('No DEM data to export')
        else:
            if (file_name == None):
                file_name = self._file_name + '_w=' + \
                    str(2**self._dem_data.num_aggre) + '.npz'
            self._dem_data.export(file_name)

    def initialize_dem(self, band=1):
        self._dem_data = Dem_data(self._img.read(band))

    def aggregate_dem(self, num_aggre=1):
        if (self._dem_data is None):
            self.initialize_dem(1)

        aggregation.aggregate_dem(self._dem_data, num_aggre)

    def aggregate_basic(self, num_aggre=1):
        if (self._dem_data is None):
            self.initialize_dem(1)

        aggregation.aggregate_basic(self._dem_data, num_aggre)

    def aggregate_basic_brute(self, num_aggre=1):
        if (self._dem_data is None):
            self.initialize_dem(1)

        aggregation.aggregate_basic_brute(self._dem_data, num_aggre)

    def dem_slope(self):
        if (self._dem_data is None):
            self.initialize_dem(1)

        slope = dem.slope(self._dem_data)

        file_name = self._create_file_name('slope', self._dem_data.num_aggre)
        helper.create_tif(slope, file_name, self._orig_profile,
                          self._dem_data.num_aggre, self.pixelSpatialResolution)

        if (self.auto_plot):
            helper.plot(file_name)

        return file_name

    # generate image of aggregated standard curvature
    def dem_profile(self):
        if (self._dem_data is None):
            self.initialize_dem(1)

        if (2**self._dem_data.num_aggre <= 2):
            print('Profile curvature cannot be calculated on windows of size 2 or 1')
            return

        profile = dem.profile(self._dem_data)

        file_name = self._create_file_name('profile', self._dem_data.num_aggre)
        # profile means two completely different things in the following line
        helper.create_tif(profile, file_name, self._orig_profile,
                          self._dem_data.num_aggre, self.pixelSpatialResolution)

        if (self.auto_plot):
            helper.plot(file_name)

        return file_name

    def dem_tangential(self):
        if (self._dem_data is None):
            self.initialize_dem(1)

        if (2**self._dem_data.num_aggre <= 2):
            print('Profile curvature cannot be calculated on windows of size 2 or 1')
            return

        tangential = dem.tangential(self._dem_data)

        file_name = self._create_file_name(
            'tangential', self._dem_data.num_aggre)
        helper.create_tif(tangential, file_name, self._orig_profile,
                          self._dem_data.num_aggre, self.pixelSpatialResolution)

        if (self.auto_plot):
            helper.plot(file_name)

        return file_name

    def dem_contour(self):
        if (self._dem_data is None):
            self.initialize_dem(1)

        if (2**self._dem_data.num_aggre <= 2):
            print('Profile curvature cannot be calculated on windows of size 2 or 1')
            return

        contour = dem.contour(self._dem_data)

        file_name = self._create_file_name('contour', self._dem_data.num_aggre)
        helper.create_tif(contour, file_name, self._orig_profile,
                          self._dem_data.num_aggre, self.pixelSpatialResolution)

        if (self.auto_plot):
            helper.plot(file_name)

        return file_name

    def dem_proper_profile(self):
        if (self._dem_data is None):
            self.initialize_dem(1)

        if (2**self._dem_data.num_aggre <= 2):
            print('Profile curvature cannot be calculated on windows of size 2 or 1')
            return

        proper_profile = dem.proper_profile(self._dem_data)

        file_name = self._create_file_name(
            'proper_profile', self._dem_data.num_aggre)
        # profile means two completely different things in the following line
        helper.create_tif(proper_profile, file_name, self._orig_profile,
                          self._dem_data.num_aggre, self.pixelSpatialResolution)

        if (self.auto_plot):
            helper.plot(file_name)

        return file_name

    def dem_proper_tangential(self):
        if (self._dem_data is None):
            self.initialize_dem(1)

        if (2**self._dem_data.num_aggre <= 2):
            print('Profile curvature cannot be calculated on windows of size 2 or 1')
            return

        proper_tangential = dem.proper_tangential(self._dem_data)

        file_name = self._create_file_name(
            'proper_tangential', self._dem_data.num_aggre)
        helper.create_tif(proper_tangential, file_name, self._orig_profile,
                          self._dem_data.num_aggre, self.pixelSpatialResolution, self.pixelSpatialResolution)

        if (self.auto_plot):
            helper.plot(file_name)

        return file_name

    def getPixelSpatialResolution(self):
        bnds = self._img.bounds
        xPixelSize = (bnds.right - bnds.left) / \
            self._orig_profile['width']
        yPixelSize = (bnds.bottom - bnds.top) / \
            self._orig_profile['height']
        self.pixelSpatialResolution = (xPixelSize, yPixelSize)
