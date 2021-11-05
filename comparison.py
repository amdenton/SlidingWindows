from windowagg.sliding_window import SlidingWindow
import windowagg.rbg as rbg
import windowagg.dem as dem
from windowagg.agg_ops import Agg_ops
import windowagg.aggregation as aggregation
from image_generator import ImageGenerator
import windowagg.config as config
import windowagg.helper as helper

import unittest
import math
import os
import shutil
import random
import time

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
from mpl_toolkits import mplot3d

class SlidingWindowComparison():
    
    def __init__(self):
        #self.img_dir = 'img/'
        #self.img_dir = 'img_old/'
        self.num_aggre = 6
        #self.sigma = self.image_size / 16
        #self.num_test_points = 20
        self.dtype = config.work_dtype
        self.noise_true = True
        self.our_rsquared = np.empty([5,4])
        self.arc_rsquared = np.empty([5,4])
        self.our_mape = np.empty([5,4])
        self.arc_mape = np.empty([5,4])

        self.artificial = True#False
       
        if self.artificial:
            self.img_dir = 'img_proposed/'
            self.image_size = 512
            self.img_gen = ImageGenerator(self.noise_true,self.img_dir)
            if self.noise_true:
                #self.file_desig = '_n01'
                self.file_desig = '_n001'
                self.arc_desig = 'Arc_Final_Output/Noise/Arc_multi_noise_random_gauss'
                self.our_desig = 'multi_random_gauss_n001'
            else:    
                self.file_desig = ''
                self.arc_desig = 'Arc_Final_Output/NoNoise/Arc_multi_random_gauss'
                self.our_desig = 'multi_random_gauss'
            #self.multi_random_gauss = self.img_gen.multi_random_gauss(self.image_size)
            self.multi_random_gauss = self.img_gen.multi_random_gauss_bare(self.image_size)
        else:
            #self.img_dir = 'img_rrv/NewArea1_500_Ours/'
            self.img_dir = 'img_rrv/NewArea1/'
            self.image_size = 5000
            #self.landscape = self.img_gen.landscape(self.img_dir+'FilledElevationQGIS.tif')
            self.img_gen = ImageGenerator(self.noise_true,self.img_dir)
            self.landscape = self.img_gen.landscape(self.img_dir+'Large_ND_Image')
            #self.image_height = 1271
            #self.image_height = 1103
            #self.image_width = 1273
            #self.image_width = 5000
            #self.image_height = 5000
            #self.image_size = 500
            #self.arc_desig = 'NewArea1_500/Large_ND_Image'
            #self.our_desig = 'Large_ND_Image'
            #self.image_width = 1400
            #self.image_height = 1400
            #self.arc_desig = 'NewArea2/Small_ND_Image'
            #self.landscape = self.img_gen.landscape(self.img_dir+self.arc_desig)
            #self.landscape = self.img_gen.landscape(self.img_dir+'NewArea3/River_ND_Image')
            #self.landscape_ours = self.img_gen.landscape(self.img_dir+self.our_desig)

        #self.gauss = self.img_gen.gauss(self.image_size)
        #self.gauss = self.img_gen.gauss(self.image_size,self.image_size/2,self.sigma,0)
        #self.random = self.img_gen.random(self.image_size)
        #self.parabola = self.img_gen.parabola(self.image_size)
        #self.multi_random_gauss = self.img_gen.multi_random_gauss_av_geotiff(self.image_size) #not working
        #self.multi_random_gauss = '/home/adenton/code/SlidingWindows/img/multi_random_gauss_n01_geotiff.tif'
        
    def create_geotiff(self):
        self.multi_random_gauss_geotiff = self.img_gen.multi_random_gauss_geotiff(self.image_size)
        #self.multi_random_gauss_av_geotiff = self.img_gen.multi_random_gauss_av_geotiff(self.image_size)
         
    def compare_multi(self,num_aggre):
        with SlidingWindow(self.multi_random_gauss) as slide_window:
            print('self.multi_random_gauss: ',self.multi_random_gauss)
#        with SlidingWindow(self.parabola) as slide_window:
            slide_window.convert_image = False
            slide_window.initialize_dem()
            slide_window.aggregate_dem(num_aggre)
            try:
                #arr_img = slide_window._dem_data.z()
                #print('image array')
                #print(arr_img)
                #print('max of image array: ',arr_img.max())
                # The following only gets the filenamee
                print(self.img_gen.multi_random_gauss_arr(0,self.image_size,0))
                #self.img_gen.multi_random_gauss(self.image_size)                

                slope_path = slide_window.dem_slope()
                with rasterio.open(slope_path) as img:
                    arr_slope = img.read(1)
                print('Numerically calculated slope')
                print(arr_slope)

                # assumed to be square
                size = arr_slope.shape[0]
                slope = self.img_gen.multi_random_gauss_arr(1,size,num_aggre)                
#                slope = self.img_gen.parabola_arr(1,size,num_aggre)    
                print('Analytically calculated slope')
                print(slope)
                #for y in range(size):
                #    for x in range(size):
                #        slope[x,y] = self.img_gen.gauss_slope_point(x, y, size, size/2, self.image_size/8)
                print('max of abs(slope array-analytical slope)')
                print(np.absolute(arr_slope-slope).max())
                print('math.sqrt(np.average(np.square(arr_slope-slope)))/math.sqrt(np.average(np.square(arr_slope)))')
                print(math.sqrt(np.average(np.square(arr_slope-slope)))/math.sqrt(np.average(np.square(arr_slope))))
                #print('analytical slope')
                #print(slope)
                print('min of slope array: ',arr_slope.min())
                print('max of slope array: ',arr_slope.max())
                print('min of analytical slope: ',slope.min())
                print('max of analytical slope: ',slope.max())
                print('number of aggregations:',num_aggre)
                print('image size: ',self.image_size)
                print('processed size: ',size)
            
                
                profile_path = slide_window.dem_profile()
                with rasterio.open(profile_path) as img:
                    arr_profile = img.read(1)
                print('Numerically calculated profile')
                print(arr_profile)

                # assumed to be square
                size = arr_profile.shape[0]
                profile = self.img_gen.multi_random_gauss_arr(3,size,num_aggre)                
#                profile = self.img_gen.parabola_arr(3,size,num_aggre) #np.ones(arr_standard.shape)/(self.image_size/8)**4                
                print('Analytically calculated profile')
                print(profile)
                print('numerically / analytically calculated profile')
                print(arr_profile/profile)
                print('max of abs(profile array-analytical profile curvature)')
                print(np.absolute(arr_profile-profile).max())
                print('math.sqrt(np.average(np.square(arr_standard-standard)))/math.sqrt(np.average(np.square(arr_standard)))')
                print(math.sqrt(np.average(np.square(arr_profile-profile)))/math.sqrt(np.average(np.square(arr_profile))))
                #print('analytical slope')
                #print(slope)
                print('min of profile array: ',arr_profile.min())
                print('max of profile array: ',arr_profile.max())
                print('min of analytical profile curvature: ',profile.min())
                print('max of analytical profile curvature: ',profile.max())
                print('number of aggregations:',num_aggre)
                print('image size: ',self.image_size)
                print('processed size: ',size)

                #profile_path = slide_window.dem_profile()
                #with rasterio.open(profile_path) as img:
                #    arr_profile = img.read(1)
                #print(arr_profile)

                planform_path = slide_window.dem_planform()
                with rasterio.open(planform_path) as img:
                    arr_planform = img.read(1)
                print('Analytically calculated planform')
                print(arr_planform)

                # assumed to be square
                size = arr_planform.shape[0]
                planform = self.img_gen.multi_random_gauss_arr(4,size,num_aggre)                
#                profile = self.img_gen.parabola_arr(3,size,num_aggre) #np.ones(arr_standard.shape)/(self.image_size/8)**4                
                print('Analytically calculated planform')
                print(planform)
                print('numerically / analytically calculated planform')
                print(arr_planform/planform)
                
                print('max of abs(profile array-analytical profile curvature)')
                print(np.absolute(arr_planform-planform).max())
                print('math.sqrt(np.average(np.square(arr_standard-standard)))/math.sqrt(np.average(np.square(arr_standard)))')
                print(math.sqrt(np.average(np.square(arr_planform-planform)))/math.sqrt(np.average(np.square(arr_planform))))
                #print('analytical slope')
                #print(slope)
                print('min of planform array: ',arr_planform.min())
                print('max of planform array: ',arr_planform.max())
                print('min of analytical planform curvature: ',planform.min())
                print('max of analytical planform curvature: ',planform.max())
                print('number of aggregations:',num_aggre)
                print('image size: ',self.image_size)
                print('processed size: ',size)

                horizontal_path = slide_window.dem_horizontal()
                with rasterio.open(horizontal_path) as img:
                    arr_horizontal = img.read(1)
                print('Analytically calculated planform')
                print(arr_horizontal)

                # assumed to be square
                size = arr_horizontal.shape[0]
                horizontal= self.img_gen.multi_random_gauss_arr(6,size,num_aggre)                
#                profile = self.img_gen.parabola_arr(3,size,num_aggre) #np.ones(arr_standard.shape)/(self.image_size/8)**4                
                print('Analytically calculated horizontal')
                print(horizontal)
                print('numerically / analytically calculated planform')
                print(arr_horizontal/horizontal)
                

            finally:
                #if (os.path.exists(slope_path)):
                #    os.remove(slope_path)
                print('Done')

    def remove_frame(self,arr,frame):
        print('in remove_frame, frame is: ',frame)
        print('arr.shape[0] is: ',arr.shape[0])
        print('arr.shape[1] is: ',arr.shape[1])
        out_arr = np.empty([arr.shape[0]-2*frame,arr.shape[1]-2*frame])
        for y in range (0,arr.shape[1]-2*frame):
            for x in range (0,arr.shape[0]-2*frame):
                out_arr[x,y] = arr[x+frame,y+frame]
        print('out_arr.shape[0] is: ',out_arr.shape[0])
        print('out_arr.shape[1] is: ',out_arr.shape[1])

        return out_arr #arr
                
    def arc_error(self,level_index=0,frame=-1):
        file_desig = self.file_desig 
        arc_desig = self.arc_desig
        print('arc_desig: ',arc_desig)
        with_arc = True#False
        verbose = True
# slope
        if with_arc:
            arc_levels = ['3','8','16','32','64']#3, 7, 12, 23
            qc = arc_levels[level_index]
            if frame == -1:
                frame = int(int(qc)/2)+1
        w_levels = ['4','8','16','32','64']
        wc = w_levels[level_index]
        print('frame: '+str(frame))
        print("=== SLOPE === wc = "+wc)
        if with_arc:
            with rasterio.open(self.img_dir+arc_desig+'_slope_w'+qc+'.tif') as img:
                arc_array = self.remove_frame(img.read(1),frame)
                if verbose: 
                    print(arc_array)
                    print('min of arc array: ',arc_array.min())
                    print('max of arc array: ',arc_array.max())
            with rasterio.open(self.img_dir+'multi_random_gauss_slope_analytical.tif') as img:
                analytical_array = self.remove_frame(img.read(1),frame)
                if verbose: 
                    print(analytical_array)
                    print('min of analytical array: ',analytical_array.min())
                    print('max of analytical array: ',analytical_array.max())
        with rasterio.open(self.img_dir+'multi_random_gauss_slope_analytical_w='+wc+'.tif') as img:
            analytical_wc = img.read(1)
            if verbose: 
                print(analytical_wc)
                print('min of analytical_wc array: ',analytical_wc.min())
                print('max of analytical_wc array: ',analytical_wc.max())
        with rasterio.open(self.img_dir+'multi_random_gauss'+file_desig+'_slope_w='+wc+'.tif') as img:
            wc_array = img.read(1)
            if verbose: 
                print(wc_array)
                print('min of wc array: ',wc_array.min())
                print('max of wc array: ',wc_array.max())
        
        if with_arc:
            print('MAPE of arc_array and analytical_array')
            mape = 100*np.average(np.abs((arc_array-analytical_array)/analytical_array))
            print(mape)
            self.arc_mape[level_index,0] = mape
            correlation_matrix = np.corrcoef(arc_array,analytical_array)
            r_squared = correlation_matrix[0,1]**2
            if verbose:
                print('correlation matrix')
                print(correlation_matrix)
                print('R squared: ',r_squared)
            self.arc_rsquared[level_index,0] = r_squared
            #plt.figure()
            #plt.scatter(arc_array,analytical_array)

        print('MAPE of wc_array and analytical_array')
        mape = 100 * np.average(np.abs((wc_array-analytical_wc)/analytical_wc))
        print(mape)
        self.our_mape[level_index,0] = mape
        correlation_matrix = np.corrcoef(wc_array,analytical_wc)
        r_squared = correlation_matrix[0,1]**2
        print('correlation matrix')
        print(correlation_matrix)
        print('R squared: ',r_squared)
        self.our_rsquared[level_index,0] = r_squared
        #plt.figure()
        #plt.scatter(wc_array,analytical_wc)



# profile
        print("=== PROFILE ===")
        if with_arc:
            with rasterio.open(self.img_dir+arc_desig+'_prof_w'+qc+'.tif') as img:
                arc_array = -0.01*self.remove_frame(img.read(1),frame)
                if verbose: 
                    print(arc_array)
                print('min of arc array: ',arc_array.min())
                print('max of arc array: ',arc_array.max())
                    #plt.figure()
                    #plt.imshow(arc_array)
            with rasterio.open(self.img_dir+'multi_random_gauss_profile_analytical.tif') as img:
                analytical_array = self.remove_frame(img.read(1),frame)
                if verbose: 
                    print('analytical_array')
                    print(analytical_array)
                    print('min of analytical array: ',analytical_array.min())
                    print('max of analytical array: ',analytical_array.max())
                    print("arc_array/analytical_array")
                    print(np.divide(arc_array,analytical_array))
        with rasterio.open(self.img_dir+'multi_random_gauss_profile_analytical_w='+wc+'.tif') as img:
            analytical_wc = img.read(1)
            if verbose: 
                print('analytical_wc')
                print(analytical_wc)
                print('min of analytical_wc array: ',analytical_wc.min())
                print('max of analytical_wc array: ',analytical_wc.max())
        with rasterio.open(self.img_dir+'multi_random_gauss'+file_desig+'_profile_w='+wc+'.tif') as img:
            wc_array = img.read(1)
            if verbose: 
                print('wc_array')
                print(wc_array)
                print('min of wc_array: ',wc_array.min())
                print('max of wc_array: ',wc_array.max())
                print("wc_array/analytical_array_wc")
                print(np.divide(wc_array,analytical_wc))
        
        if with_arc:
            print('MAPE of arc_array and analytical_array')
            mape = 100 * np.average(np.abs((arc_array-analytical_array)/analytical_array))
            print(mape)
            self.arc_mape[level_index,1] = mape
            correlation_matrix = np.corrcoef(arc_array,analytical_array)
            r_squared = correlation_matrix[0,1]**2
            print('correlation matrix')
            print(correlation_matrix)
            print('R squared: ',r_squared)
            self.arc_rsquared[level_index,1] = r_squared

        print('MAPE of wc_array and analytical_array')
        mape = 100 * np.average(np.abs((wc_array-analytical_wc)/analytical_wc))
        print(mape)
        self.our_mape[level_index,1] = mape
        correlation_matrix = np.corrcoef(wc_array,analytical_wc)
        r_squared = correlation_matrix[0,1]**2
        print('correlation matrix')
        print(correlation_matrix)
        print('R squared: ',r_squared)
        self.our_rsquared[level_index,1] = r_squared

# planform
        print("=== PLANFORM ===")
        if with_arc:
            with rasterio.open(self.img_dir+arc_desig+'_plan_w'+qc+'.tif') as img:
                arc_array = -0.01*self.remove_frame(img.read(1),frame)
                if verbose: 
                    print('arc_array')
                    print(arc_array)
                    print('min of arc array: ',arc_array.min())
                    print('max of arc array: ',arc_array.max())
                    #plt.figure()
                    #plt.imshow(arc_array)
            with rasterio.open(self.img_dir+'multi_random_gauss_planform_analytical.tif') as img:
                analytical_array = self.remove_frame(img.read(1),frame)
                if verbose: 
                    print('analytical_array')
                    print(analytical_array)
                    print('min of analytical array: ',analytical_array.min())
                    print('max of analytical array: ',analytical_array.max())
                    print("arc_array/analytical_array")
                    print(np.divide(arc_array,analytical_array))
        with rasterio.open(self.img_dir+'multi_random_gauss_planform_analytical_w='+wc+'.tif') as img:
            analytical_wc = img.read(1)
            if verbose: 
                print('analytical_wc')
                print(analytical_wc)
                print('min of analytical_wc array: ',analytical_wc.min())
                print('max of analytical_wc array: ',analytical_wc.max())
        with rasterio.open(self.img_dir+'multi_random_gauss'+file_desig+'_planform_w='+wc+'.tif') as img:
            wc_array = img.read(1)
            if verbose: 
                print('wc_array')
                print(wc_array)
                print('min of wc_array: ',wc_array.min())
                print('max of wc_array: ',wc_array.max())
        
        if with_arc:
            print('MAPE of arc_array and analytical_array')
            mape = 100 * np.average(np.abs((arc_array-analytical_array)/analytical_array))
            print(mape)
            self.arc_mape[level_index,2] = mape
            correlation_matrix = np.corrcoef(arc_array,analytical_array)
            r_squared = correlation_matrix[0,1]**2
            print('correlation matrix')
            print(correlation_matrix)
            print('R squared: ',r_squared)
            self.arc_rsquared[level_index,2] = r_squared

        print('MAPE of wc_array and analytical_array')
        mape = 100 * np.average(np.abs((wc_array-analytical_wc)/analytical_wc))
        print(mape)
        self.our_mape[level_index,2] = mape
        correlation_matrix = np.corrcoef(wc_array,analytical_wc)
        r_squared = correlation_matrix[0,1]**2
        print('correlation matrix')
        print(correlation_matrix)
        print('R squared: ',r_squared)
        self.our_rsquared[level_index,2] = r_squared

        print("=== HORIZONTAL ===")
        if with_arc:
            with rasterio.open(self.img_dir+arc_desig+'_plan_w'+qc+'.tif') as img:
                arc_array = 0.01*self.remove_frame(img.read(1),frame)
                if verbose: 
                    print('arc_array')
                    print(arc_array)
                    print('min of arc array: ',arc_array.min())
                    print('max of arc array: ',arc_array.max())
                    #plt.figure()
                    #plt.imshow(arc_array)
            with rasterio.open(self.img_dir+'multi_random_gauss_horizontal_analytical.tif') as img:
                analytical_array = self.remove_frame(img.read(1),frame)
                if verbose: 
                    print('analytical_array')
                    print(analytical_array)
                    print('min of analytical array: ',analytical_array.min())
                    print('max of analytical array: ',analytical_array.max())
                    print("arc_array/analytical_array")
                    print(np.divide(arc_array,analytical_array))
        with rasterio.open(self.img_dir+'multi_random_gauss_horizontal_analytical_w='+wc+'.tif') as img:
            analytical_wc = img.read(1)
            if verbose: 
                print('analytical_wc')
                print(analytical_wc)
                print('min of analytical_wc array: ',analytical_wc.min())
                print('max of analytical_wc array: ',analytical_wc.max())
        with rasterio.open(self.img_dir+'multi_random_gauss'+file_desig+'_horizontal_w='+wc+'.tif') as img:
            wc_array = img.read(1)
            if verbose: 
                print('wc_array')
                print(wc_array)
                print('min of wc_array: ',wc_array.min())
                print('max of wc_array: ',wc_array.max())
        
        if with_arc:
            print('MAPE of arc_array and analytical_array')
            mape = 100 * np.average(np.abs((arc_array-analytical_array)/analytical_array))
            print(mape)
            self.arc_mape[level_index,3] = mape
            correlation_matrix = np.corrcoef(arc_array,analytical_array)
            r_squared = correlation_matrix[0,1]**2
            print('correlation matrix')
            print(correlation_matrix)
            print('R squared: ',r_squared)
            self.arc_rsquared[level_index,3] = r_squared

        print('MAPE of wc_array and analytical_array')
        mape = 100 * np.average(np.abs((wc_array-analytical_wc)/analytical_wc))
        print(mape)
        self.our_mape[level_index,3] = mape
        
        correlation_matrix = np.corrcoef(wc_array,analytical_wc)
        r_squared = correlation_matrix[0,1]**2
        print('correlation matrix')
        print(correlation_matrix)
        print('R squared: ',r_squared)
        self.our_rsquared[level_index,3] = r_squared


    #not finished
    def file_aggregator(self):
        wc = 4
        file_desig = '_n01'
        with rasterio.open(self.img_dir+'multi_random_gauss'+file_desig+'_slope_w='+wc+'.tif') as img:
            slope_array = img.read(1)
        with rasterio.open(self.img_dir+'multi_random_gauss'+file_desig+'_standard_w='+wc+'.tif') as img:
            standard_array = img.read(1)
        with rasterio.open(self.img_dir+'multi_random_gauss'+file_desig+'_profile_w='+wc+'.tif') as img:
            profile_array = img.read(1)
        with rasterio.open(self.img_dir+'multi_random_gauss'+file_desig+'_planform_w='+wc+'.tif') as img:
            planform_array = img.read(1)
        w_aggs = ['8','16','32','64']
        for wc in w_aggs:
            with rasterio.open(self.img_dir+'multi_random_gauss'+file_desig+'_slope_w='+wc+'.tif') as img:
                slope_array = slope_array + img.read(1)
            with rasterio.open(self.img_dir+'multi_random_gauss'+file_desig+'_standard_w='+wc+'.tif') as img:
                standard_array = standard_array + img.read(1)
            with rasterio.open(self.img_dir+'multi_random_gauss'+file_desig+'_profile_w='+wc+'.tif') as img:
                profile_array = profile_array + img.read(1)
            with rasterio.open(self.img_dir+'multi_random_gauss'+file_desig+'_planform_w='+wc+'.tif') as img:
                planform_array = planform_array + img.read(1)

        fn = self.img_dir+'multi_random_gauss'+file_desig+'_slope_w='+wc+'agg.tif'
        helper.create_tif(slope_array,fn)
        fn = self.img_dir+'multi_random_gauss'+file_desig+'_standard_w='+wc+'agg.tif'
        helper.create_tif(standard_array,fn)
        fn = self.img_dir+'multi_random_gauss'+file_desig+'_profile_w='+wc+'agg.tif'
        helper.create_tif(profile_array,fn)
        fn = self.img_dir+'multi_random_gauss'+file_desig+'_planform_w='+wc+'agg.tif'
        helper.create_tif(planform_array,fn)        

    def plot_tiff_arc(self):
        int_columns = 3
        image_frame = 10
        arc_desig = self.arc_desig
        row_desig = ['3','8','16','32','64']#3, 7, 12, 23
        #row_desig = ['3','15','30','63']
        rows = (self.image_size + 2 * image_frame) * len(row_desig)
        print('rows: ',rows)
        columns = (self.image_size + 2 * image_frame) * int_columns
        print('colums: ',columns)
        out_array = np.zeros([rows,columns])
        for i, num_id in enumerate(row_desig):
            with rasterio.open(self.img_dir+arc_desig+'_slope_w'+num_id+'.tif') as img:
                #arc_array0 = img.read(1) 
                frame = int(int(num_id)/2)+1
                arc_array0 = 0.01*self.remove_frame(img.read(1),frame)
                arc_array0 = helper.arr_dtype_conversion(arc_array0)
                image_loc = self.image_size - 2*frame
                for j in np.arange(image_loc):
                    for k in np.arange(image_loc):
                        out_array[image_frame+frame+i*(self.image_size+2*image_frame)+k,image_frame+frame+j] = arc_array0[k,j]
#        for i, num_id in enumerate(row_desig):
#            with rasterio.open(self.img_dir+arc_desig+'_slope_w'+qc+'.tif') as img:
#                #arc_array1 = img.read(1) 
#                arc_array1 = helper.arr_dtype_conversion(img.read(1))
#                for j in np.arange(self.image_size):
#                    for k in np.arange(self.image_size):
#                        out_array[image_frame+i*(self.image_size+2*image_frame)+k,image_frame+(self.image_size+2*image_frame)+j] = arc_array1[k,j]
#                plt.figure()
#                plt.imshow(arc_array1)
#                plt.figure()
#                plt.imshow(out_array)
        for i, num_id in enumerate(row_desig):
            with rasterio.open(self.img_dir+arc_desig+'_prof_w'+num_id+'.tif') as img:
                #arc_array2 = img.read(1) 
                frame = int(int(num_id)/2)+1
                arc_array2 = -0.01*self.remove_frame(img.read(1),frame)
                arc_array2 = helper.arr_dtype_conversion(arc_array2)
                print('Size of arc_array2: ',np.shape(arc_array2))
                image_loc = self.image_size - 2*frame
                for j in np.arange(image_loc):
                    for k in np.arange(image_loc):
                        out_array[image_frame+frame+i*(self.image_size+2*image_frame)+k,image_frame+frame+self.image_size+2*image_frame+j] = arc_array2[k,j]
        for i, num_id in enumerate(row_desig):
            with rasterio.open(self.img_dir+arc_desig+'_plan_w'+num_id+'.tif') as img:
                #arc_array3 = img.read(1) 
                frame = int(int(num_id)/2)+1
                arc_array3 = 0.01*self.remove_frame(img.read(1),frame)
                arc_array3 = helper.arr_dtype_conversion(arc_array3)
                image_loc = self.image_size - 2*frame
                for j in np.arange(image_loc):
                    for k in np.arange(image_loc):
                        out_array[image_frame+frame+i*(self.image_size+2*image_frame)+k,image_frame+frame+2*(self.image_size+2*image_frame)+j] = arc_array3[k,j]
        plt.figure()
#        frame1 = plt.gca()
#        frame1.axes.get_xaxis().set_visible(False)
#        frame1.axes.get_yaxis().set_visible(False)
        plt.axis('off')
        plt.imshow(out_array)
#        plt.savefig('fig_artificial_arc_with_noise.png',dpi=300,bbox_inches='tight',pad_inches = 0)
        plt.savefig('fig_landscape_arc.png',dpi=300,bbox_inches='tight',pad_inches = 0)
        #fn = 'out.tif'
        #helper.create_tif(out_array,fn)

    def plot_tiff_ours(self):
        int_columns = 3
        image_frame = 10
        row_desig = ['4','8','16','32','64']
        #row_desig = ['4']
        #row_desig = ['3','15','30','63']
        rows = (self.image_size + 2 * image_frame) * len(row_desig)
        print('rows: ',rows)
        columns = (self.image_size + 2 * image_frame) * int_columns
        print('colums: ',columns)
        out_array = np.zeros([rows,columns])
        for i, num_id in enumerate(row_desig):
            with rasterio.open(self.img_dir+self.our_desig+'_slope_w='+num_id+'.tif') as img:
                #arc_array0 = img.read(1) 
                frame = int(int(num_id)/2)
                array0 = helper.arr_dtype_conversion(img.read(1))
                plt.figure()
                plt.imshow(array0)
                image_loc = self.image_size - 2*frame
                for j in np.arange(image_loc):
                    for k in np.arange(image_loc):
                        out_array[image_frame+frame+i*(self.image_size+2*image_frame)+k,image_frame+frame+j] = array0[k,j]
#        for i, num_id in enumerate(row_desig):
#            with rasterio.open(self.img_dir+arc_desig+'_slope_w'+qc+'.tif') as img:
#                #arc_array1 = img.read(1) 
#                arc_array1 = helper.arr_dtype_conversion(img.read(1))
#                for j in np.arange(self.image_size):
#                    for k in np.arange(self.image_size):
#                        out_array[image_frame+i*(self.image_size+2*image_frame)+k,image_frame+(self.image_size+2*image_frame)+j] = arc_array1[k,j]
#                plt.figure()
#                plt.imshow(arc_array1)
#                plt.figure()
#                plt.imshow(out_array)
        for i, num_id in enumerate(row_desig):
            with rasterio.open(self.img_dir+self.our_desig+'_profile_w='+num_id+'.tif') as img:
                #arc_array2 = img.read(1) 
                frame = int(int(num_id)/2)
                array2 = helper.arr_dtype_conversion(img.read(1))
                image_loc = self.image_size - 2*frame
                for j in np.arange(image_loc):
                    for k in np.arange(image_loc):
                        out_array[image_frame+frame+i*(self.image_size+2*image_frame)+k,image_frame+frame+self.image_size+2*image_frame+j] = array2[k,j]
        for i, num_id in enumerate(row_desig):
            with rasterio.open(self.img_dir+self.our_desig+'_horizontal_w='+num_id+'.tif') as img:
                #arc_array3 = img.read(1) 
                frame = int(int(num_id)/2)
                array3 = helper.arr_dtype_conversion(img.read(1))
                image_loc = self.image_size - 2*frame
                for j in np.arange(image_loc):
                    for k in np.arange(image_loc):
                        out_array[image_frame+frame+i*(self.image_size+2*image_frame)+k,image_frame+frame+2*(self.image_size+2*image_frame)+j] = array3[k,j]
        plt.figure()
        #frame1 = plt.gca()
        #frame1.axes.get_xaxis().set_visible(False)
        #frame1.axes.get_yaxis().set_visible(False)
        plt.axis('off')
        plt.imshow(out_array)
        #plt.savefig('fig_artificial_ours_with_noise.png',dpi=300,bbox_inches='tight',pad_inches = 0)
        plt.savefig('fig_landscape_ours.png',dpi=300,bbox_inches='tight',pad_inches = 0)

        #fn = 'fig_artificial_ours_with_noise.tif'
        #helper.create_tif(out_array,fn)



    def intro(self):
#        with rasterio.open(self.img_dir+'.tif') as img:
        image_size = 40
        window_size = 16
        fit = np.empty([window_size,window_size])
        x_window = np.empty([window_size,window_size])
        y_window = np.empty([window_size,window_size])
        arr = np.zeros([image_size*image_size])
        x_arr = np.zeros([image_size*image_size])
        y_arr = np.zeros([image_size*image_size])
        i = 0
        for y in range (window_size):
            for x in range (window_size):
                value = 1 - (((x-0.2*image_size)/image_size)**2 + ((y-0.55*image_size)/image_size)**2)
                fit[x][y] = value
                x_window[x][y] = 0.5 * image_size + x
                y_window[x][y] = 0.15 * image_size + y
                i += 1
        print(i)
        print(x_window)
        print(y_window)
        print(fit)
        i=0
        for y in range (image_size):
            for x in range (image_size):
                value = 1 - (((x-0.7*image_size)/image_size)**2 + ((y-0.7*image_size)/image_size)**2) 
                r = 0.015 * np.random.normal()
                if ((r > 0) or (x < 0.5 * image_size) or (x > 0.5 * image_size + window_size) or (y < 0.15 * image_size) or (y > 0.15 * image_size + window_size)):
                    arr[i] = value + r
                    x_arr[i] = x
                    y_arr[i] = y
                    i += 1
        print(i)
        print(x_arr)
        print(y_arr)
        print(arr)
        plt.figure(figsize=(10,10))
        ax = plt.axes(projection='3d')
            
#        x_lin = np.linspace(0,arr.shape[0]-1,arr.shape[0])
#        y_lin = np.linspace(0,arr.shape[1]-1,arr.shape[1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z');
        
        ax.scatter3D(x_arr,y_arr,arr,c='black')
#        ax.scatter3D(x_window,y_window,fit,c='red')
        ax.plot_surface(x_window,y_window,fit,cmap='viridis')
        plt.savefig('intro.png')
        
    def all_artificial_images(self):
        
        int_columns = 2
        int_rows = 2
        image_frame = 10
        size = self.image_size
        num_aggre = 0
        rows = (self.image_size + 2 * image_frame) * int_rows
        print('rows: ',rows)
        columns = (self.image_size + 2 * image_frame) * int_columns
        print('colums: ',columns)
        out_array = np.zeros([rows,columns])

        orig = helper.arr_dtype_conversion(self.img_gen.multi_random_gauss_arr(0,self.image_size,0))
        print('Analytical image')
        print(orig)
        print('min of original image: ',orig.min())
        print('max of original image: ',orig.max())
        plt.figure()
        plt.imshow(orig)
        for j in np.arange(self.image_size):
            for k in np.arange(self.image_size):
                out_array[image_frame+k,image_frame+j] = orig[k,j]
        
        slope = helper.arr_dtype_conversion(self.img_gen.multi_random_gauss_arr(1,self.image_size,0))              
        print('Analytically calculated slope')
        print(slope)
        print('min of analytical slope: ',slope.min())
        print('max of analytical slope: ',slope.max())
        plt.figure()
        plt.imshow(slope)
        for j in np.arange(self.image_size):
            for k in np.arange(self.image_size):
                out_array[image_frame+k,self.image_size+3*image_frame+j] = slope[k,j]
        
#        aspect = helper.arr_dtype_conversion(self.img_gen.multi_random_gauss_arr(5,self.image_size,0))               
#        print('Analytically calculated aspect')
#        print(aspect)
#        print('min of analytical aspect: ',aspect.min())
#        print('max of analytical aspect: ',aspect.max())
#        plt.figure()
#        plt.imshow(aspect)
#        for j in np.arange(self.image_size):
#            for k in np.arange(self.image_size):
#                out_array[image_frame+k,2*self.image_size+5*image_frame+j] = aspect[k,j]
        
#        curv = helper.arr_dtype_conversion(self.img_gen.multi_random_gauss_arr(2,self.image_size,0))
#        print('Analytical curvature')
#        print(curv)
#        print('min of analtyical curvature: ',curv.min())
#        print('max of analtyical curvature: ',curv.max())
#        plt.figure()
#        plt.imshow(curv)
#        for j in np.arange(self.image_size):
#            for k in np.arange(self.image_size):
#                out_array[self.image_size+3*image_frame+k,image_frame+j] = curv[k,j]
        
        profile = helper.arr_dtype_conversion(self.img_gen.multi_random_gauss_arr(3,self.image_size,0))               
        print('Analytically calculated profile')
        print(profile)
        print('min of analytical profile: ',profile.min())
        print('max of analytical profile: ',profile.max())
        plt.figure()
        plt.imshow(profile)
        for j in np.arange(self.image_size):
            for k in np.arange(self.image_size):
                out_array[self.image_size+3*image_frame+k,image_frame+j] = profile[k,j]
        
        horizontal = helper.arr_dtype_conversion(self.img_gen.multi_random_gauss_arr(6,self.image_size,0))              
        print('Analytically calculated horizontal')
        plt.figure()
        plt.imshow(horizontal)
        for j in np.arange(self.image_size):
            for k in np.arange(self.image_size):
                out_array[self.image_size+3*image_frame+k,self.image_size+3*image_frame+j] = horizontal[k,j]
        plt.figure()
        plt.axis('off')
        #plt.savefig('image.png', bbox_inches='tight',pad_inches = 0)
        #frame1 = plt.gca()
        #frame1.axes.get_xaxis().set_visible(False)
        #frame1.axes.get_yaxis().set_visible(False)
        plt.imshow(out_array)
        plt.savefig('fig_artificial_analytical.png',dpi=500,bbox_inches='tight',pad_inches = 0)
        fn = 'fig_artificial_analytical.tif'
        helper.create_tif(out_array,fn)
     
    def landscape_analyses_ours(self):
        stretch_factor = 4.53/5.00
        num_aggre = self.num_aggre
        int_columns = 2
        int_rows = 2
        image_frame = 10
        #size = self.image_size
        num_aggre = self.num_aggre
        our_frame = 2**(num_aggre-1)
        rows = (self.image_height + 2 * image_frame) * int_rows
        print('rows: ',rows)
        columns = (self.image_width + 2 * image_frame) * int_columns
        print('colums: ',columns)
        out_array = np.zeros([rows,columns])
        with SlidingWindow(self.landscape) as slide_window:
            print('self.landscape: ',self.landscape)
            slide_window.convert_image = False
            slide_window.initialize_dem()
            slide_window.aggregate_dem(num_aggre)
            try:
                orig_path = self.landscape
                with rasterio.open(orig_path) as img:
                    arr_orig = helper.arr_dtype_conversion(self.remove_frame(img.read(1),our_frame))
                print('Original Image')
                print(arr_orig)
                print('min of orig array: ',arr_orig.min())
                print('max of orig array: ',arr_orig.max())
                print('number of aggregations:',num_aggre)
                print('arr_orig.shape[0]: ',arr_orig.shape[0])
                print('arr_orig.shape[1]: ',arr_orig.shape[1])
                plt.figure()
                plt.imshow(arr_orig)
                for j in np.arange(arr_orig.shape[1]):
                    for k in np.arange(arr_orig.shape[0]):
                        out_array[image_frame+our_frame+k,image_frame+our_frame+j] = arr_orig[k,j]

                slope_path = slide_window.dem_slope()
                print('slope_path: ',slope_path)
                with rasterio.open(slope_path) as img:
                    arr_slope = helper.arr_dtype_conversion(img.read(1))
                print('Numerically calculated slope')
                print(arr_slope)
                size = arr_slope.shape[0]
                print('min of slope array: ',arr_slope.min())
                print('max of slope array: ',arr_slope.max())
                print('number of aggregations:',num_aggre)
                print('arr_slope.shape[0]: ',arr_slope.shape[0])
                print('arr_slope.shape[1]: ',arr_slope.shape[1])
                plt.figure()
                plt.imshow(arr_slope)
                for j in np.arange(arr_slope.shape[1]-2*our_frame):
                    for k in np.arange(arr_orig.shape[0]-2*our_frame):
                        out_array[image_frame+2*our_frame+k,self.image_width+our_frame+3*image_frame+j] = arr_slope[k,j]
            
#                aspect_path = slide_window.dem_aspect()
#                with rasterio.open(aspect_path) as img:
#                    arr_aspect = helper.arr_dtype_conversion(img.read(1))
#                print('Numerically calculated aspect')
#                print(arr_aspect)
#                print('min of aspect: ',arr_aspect.min())
#                print('max of aspect: ',arr_aspect.max())
#                plt.figure()
#                plt.imshow(arr_aspect)
#                for j in np.arange(size):
#                    for k in np.arange(size):
#                        out_array[self.image_size+3*image_frame+k,image_frame+j] = arr_aspect[k,j]
#        
#                curv_path = slide_window.dem_standard()
#                with rasterio.open(curv_path) as img:
#                    arr_curv = helper.arr_dtype_conversion(img.read(1))
#                print('Numerically calculated curvature')
#                print(arr_curv)
#                print('min of curvature: ',arr_curv.min())
#                print('max of curvature: ',arr_curv.max())
#                plt.figure()
#                plt.imshow(arr_curv)
#                for j in np.arange(size):
#                    for k in np.arange(size):
#                        out_array[self.image_size+3*image_frame+k,self.image_size+3*image_frame+j] = arr_curv[k,j]
        
                profile_path = slide_window.dem_profile()
                with rasterio.open(profile_path) as img:
                    arr_profile = helper.arr_dtype_conversion(img.read(1))
                print('Numerically calculated profile curvature')
                print(arr_profile)
                print('min of profile: ',arr_profile.min())
                print('max of profile: ',arr_profile.max())
                plt.figure()
                plt.imshow(arr_profile)
                for j in np.arange(arr_profile.shape[1]-2*our_frame):
                    for k in np.arange(arr_profile.shape[0]-2*our_frame):
                        out_array[self.image_height+3*image_frame+our_frame+k,image_frame+2*our_frame+j] = arr_profile[k,j]
        
                planform_path = slide_window.dem_planform()
                with rasterio.open(planform_path) as img:
                    arr_planform = helper.arr_dtype_conversion(img.read(1))
                print('Numerically calculated planform curvature')
                print(arr_planform)
                print('min of planform: ',arr_planform.min())
                print('max of planform: ',arr_planform.max())
                plt.figure()
                plt.imshow(arr_planform)
                for j in np.arange(arr_planform.shape[1]-2*our_frame):
                    for k in np.arange(arr_planform.shape[0]-2*our_frame):
                        out_array[self.image_height+3*image_frame+our_frame+k,self.image_width+3*image_frame+our_frame+j] = arr_planform[k,j]

                stretched_out = np.zeros([int(out_array.shape[0]*stretch_factor),out_array.shape[1]])
                for j in np.arange(out_array.shape[1]):
                    for k in np.arange(out_array.shape[0]):
                        stretched_out[int(k*stretch_factor),j] = out_array[k,j]
                        
                plt.figure()
                plt.axis('off')
                plt.imshow(stretched_out)
                plt.savefig('fig_landscape_ours.png',dpi=600,bbox_inches='tight',pad_inches = 0)
                fn = 'fig_landscape_ours.tif'
                helper.create_tif(stretched_out,fn)
               

            finally:
                #if (os.path.exists(slope_path)):
                #    os.remove(slope_path)
                print('Done')

    def landscape_analyses_arc(self):
        stretch_factor = 4.53/5.00
        num_aggre = self.num_aggre
        int_columns = 2
        int_rows = 2
        image_frame = 10
        num_aggre = self.num_aggre
        arc_frame = 2**(self.num_aggre-1)
        desigs = ['3','8','16','32','64']
        file_desig = desigs[num_aggre-2]
        image_height_loc = int(self.image_height*stretch_factor)
        rows = (image_height_loc + 2 * image_frame) * int_rows
        print('rows: ',rows)
        columns = (self.image_width + 2 * image_frame) * int_columns
        print('colums: ',columns)
        out_array = np.zeros([rows,columns])
        orig_path = self.landscape
        with rasterio.open(orig_path) as img:
            arr_orig = helper.arr_dtype_conversion(self.remove_frame(img.read(1),arc_frame))
        print('Original Image')
        print(arr_orig)
        print('arr_orig.shape[0]',arr_orig.shape[0])
        print('arr_orig.shape[1]',arr_orig.shape[1])
        print('min of orig array: ',arr_orig.min())
        print('max of orig array: ',arr_orig.max())
        print('number of aggregations:',num_aggre)
        plt.figure()
        plt.imshow(arr_orig)
        for k in np.arange(arr_orig.shape[0]):
            for j in np.arange(arr_orig.shape[1]):
                out_array[image_frame+arc_frame+int(stretch_factor*k),image_frame+arc_frame+j] = arr_orig[k,j]
                
        slope_path = self.img_dir+self.arc_desig+'_slope_w'+file_desig+'.tif'
        print('slope_path: ',slope_path)
        with rasterio.open(slope_path) as img:
            arr_slope = helper.arr_dtype_conversion(self.remove_frame(img.read(1),arc_frame))
        print('Arc slope')
        print(arr_slope)
        print('arr_slope.shape[0]',arr_slope.shape[0])
        print('arr_slope.shape[1]',arr_slope.shape[1])
        print('min of slope array: ',arr_slope.min())
        print('max of slope array: ',arr_slope.max())
        print('number of aggregations:',num_aggre)
        plt.figure()
        plt.imshow(arr_slope)
        for k in np.arange(arr_slope.shape[0]):
            for j in np.arange(arr_slope.shape[1]):
                out_array[image_frame+arc_frame+k,self.image_width+3*image_frame+arc_frame+j] = arr_slope[k,j]
            
#        aspect_path = slide_window.dem_aspect()
#        with rasterio.open(aspect_path) as img:
#            arr_aspect = helper.arr_dtype_conversion(img.read(1))
#            print('Numerically calculated aspect')
#            print(arr_aspect)
#            print('min of aspect: ',arr_aspect.min())
#            print('max of aspect: ',arr_aspect.max())
#            plt.figure()
#            plt.imshow(arr_aspect)
#            for j in np.arange(size):
#                for k in np.arange(size):
#                    out_array[self.image_size+3*image_frame+k,image_frame+j] = arr_aspect[k,j]
#        
#                curv_path = slide_window.dem_standard()
#                with rasterio.open(curv_path) as img:
#                    arr_curv = helper.arr_dtype_conversion(img.read(1))
#                print('Numerically calculated curvature')
#                print(arr_curv)
#                print('min of curvature: ',arr_curv.min())
#                print('max of curvature: ',arr_curv.max())
#                plt.figure()
#                plt.imshow(arr_curv)
#                for j in np.arange(size):
#                    for k in np.arange(size):
#                        out_array[self.image_size+3*image_frame+k,self.image_size+3*image_frame+j] = arr_curv[k,j]

        profile_path = self.img_dir+self.arc_desig+'_prof_w'+file_desig+'.tif'
        with rasterio.open(profile_path) as img:
            arr_profile = helper.arr_dtype_conversion(self.remove_frame(img.read(1),arc_frame))
        print('Arc profile curvature')
        print(arr_profile)
        print('min of profile: ',arr_profile.min())
        print('max of profile: ',arr_profile.max())
        print('arr_profile.shape[0]',arr_profile.shape[0])
        print('arr_profile.shape[1]',arr_profile.shape[1])
        plt.figure()
        plt.imshow(arr_profile)
        for k in np.arange(arr_profile.shape[0]):
            for j in np.arange(arr_profile.shape[1]):
                out_array[image_height_loc+3*image_frame+arc_frame+k,image_frame+arc_frame+j] = arr_profile[k,j]
        
        planform_path = self.img_dir+self.arc_desig+'_plan_w'+file_desig+'.tif'
        with rasterio.open(planform_path) as img:
            arr_planform = helper.arr_dtype_conversion(self.remove_frame(img.read(1),arc_frame))
        print('Arc planform curvature')
        print(arr_planform)
        print('min of planform: ',arr_planform.min())
        print('max of planform: ',arr_planform.max())
        print('arr_planform.shape[0]',arr_planform.shape[0])
        print('arr_planform.shape[1]',arr_planform.shape[1])
        plt.figure()
        plt.imshow(arr_planform)
        for k in np.arange(arr_planform.shape[0]):
            for j in np.arange(arr_planform.shape[1]):
                out_array[image_height_loc+3*image_frame+arc_frame+k,self.image_width+3*image_frame+arc_frame+j] = arr_planform[k,j]

        plt.figure()
        plt.axis('off')
        plt.imshow(out_array)
        plt.savefig('fig_landscape_arc.png',dpi=600,bbox_inches='tight',pad_inches = 0)
        fn = 'fig_landscape_arc.tif'
        helper.create_tif(out_array,fn)
        
    def basic_speed(self,num_aggre):
        time_array = np.zeros([num_aggre,2])
        with SlidingWindow(self.multi_random_gauss) as slide_window:
            slide_window.convert_image = False
            slide_window.initialize_dem()
            time_start = time.time_ns()
            i = 0
            time_start = time.time_ns()
            for num_aggre_loc in range(1,num_aggre+1):
                slide_window.aggregate_basic()
                time_array[i,0] = time.time_ns()-time_start
                i += 1
        with SlidingWindow(self.multi_random_gauss) as slide_window:
            slide_window.convert_image = False
            slide_window.initialize_dem()
            time_start = time.time_ns()
            i = 0
            time_start = time.time_ns()
            for num_aggre_loc in range(1,num_aggre+1):
                slide_window.aggregate_basic_brute(num_aggre_loc)
                time_array[i,1] = time.time_ns() - time_start
                time_start = time.time_ns()
                i += 1
               
        print('aggregation: ',time_array)
        plt.figure(figsize=(5.5, 3.5))
        plt.xscale("log")
        plt.yscale("log")
        #plt.xlim([3, 80])
        #plt.ylim([0, 18])
        plt.xticks([2,4,8,16,32],['2','4','8','16','32'])
        plt.xlabel('Window Size', fontsize='large')
        plt.ylabel('Runtime [s]', fontsize='large')
        plt.minorticks_off()
        plt.plot([2,4,8,16,32],time_array[:,0]*1E-9,'b',label='Iterative')
        plt.plot([2,4,8,16,32],time_array[:,1]*1E-9,'c',label='Brute Force')
        plt.legend(loc='upper left', shadow=True, fontsize='large')
        plt.savefig('fig_time.png',dpi=500,bbox_inches='tight',pad_inches = 0)
            
        return time_array
        
    def compare_speed(self,num_aggre):
        time_array = np.zeros([num_aggre-1,6])
        with SlidingWindow(self.landscape) as slide_window:
            slide_window.convert_image = False
            slide_window.initialize_dem()
            time_start = time.time_ns()
            i = 0
            time_start = time.time_ns()
            slide_window.aggregate_dem()
            time0 = time.time_ns() - time_start
            time_start = time.time_ns()
            for num_aggre_loc in range(2,num_aggre+1):
                slide_window.aggregate_dem()
                time_array[i,0] = time.time_ns() - time_start
                if i>=0:
                    time_array[i,1] = time_array[i,0] + time_array[i-1,1]
                else:
                    time_array[i,1] = time_array[i,0] + time0
                time_start = time.time_ns()
                slope = slide_window.dem_slope_arr()
                time_array[i,2] = time.time_ns() - time_start
                time_start = time.time_ns()
                profile = slide_window.dem_profile_arr()
                time_array[i,3] = time.time_ns() - time_start
                time_start = time.time_ns()
                horizontal = slide_window.dem_horizontal_arr()
                time_array[i,4] = time.time_ns() - time_start
                time_start = time.time_ns()
                slope, profile, horizontal = slide_window.dem_combination_arr()
                time_array[i,5] = time.time_ns() - time_start
                time_start = time.time_ns()
                i += 1
               
            slope_time = time_array[:,1] + time_array[:,2]
            profile_time = time_array[:,1] + time_array[:,3]
            horizontal_time = time_array[:,1] + time_array[:,4]
            horizontalprofile_time = time_array[:,1] + time_array[:,5]
            print('aggregation: ',time_array[:,0])
            print('cummulative aggregation: ',time_array[:,1])
            print('slope_time: ',slope_time)
            print('profile_time: ',profile_time)
            print('horizontal_time: ',horizontal_time)
            print('horizontalprofile_time: ',horizontalprofile_time)
            plt.figure(figsize=(5.5, 3.5))
            plt.xscale("log")
            plt.xlim([3, 80])
            plt.ylim([0, 18])
            plt.xticks([4,8,16,32,64],['4','8','16','32','64'])
            plt.xlabel('Window Size', fontsize='large')
            plt.ylabel('Runtime [s]', fontsize='large')
            plt.plot([4,8,16,32,64],slope_time*1E-9,'b',label='Slope')
            plt.plot([4,8,16,32,64],profile_time*1E-9,'c',label='Profile')
            plt.plot([4,8,16,32,64],horizontal_time*1E-9,'g',label='Horizontal')
            plt.plot([4,8,16,32,64],(horizontalprofile_time+ time_array[:,2])*1E-9,'m',label="All")
            plt.legend(loc='upper left', shadow=True, fontsize='large')
            plt.minorticks_off()
            plt.savefig('fig_time.png',dpi=500,bbox_inches='tight',pad_inches = 0)
            
        return time_array
   
    def landscape_compute_ours(self,num_aggre):
        with SlidingWindow(self.landscape_ours) as slide_window:
            print('self.landscape_ours: ',self.landscape_ours)
#        with SlidingWindow(self.parabola) as slide_window:
            slide_window.convert_image = False
            slide_window.initialize_dem()
            slide_window.aggregate_dem(num_aggre)
            try:
                #print(self.img_gen.landscape(0,self.image_size,0))
                #self.img_gen.multi_random_gauss(self.image_size)                

                slope_path = slide_window.dem_slope()
                with rasterio.open(slope_path) as img:
                    arr_slope = img.read(1)
                    fn = 'Large_ND_Image_slope_w='+str(2**num_aggre)+'.tif'

                helper.create_tif(arr_slope,'img_rrv/NewArea1_500_Ours/'+fn)
                            
                profile_path = slide_window.dem_profile()
                with rasterio.open(profile_path) as img:
                    arr_profile = img.read(1)
                    fn = 'Large_ND_Image_profile_w='+str(2**num_aggre)+'.tif'

                helper.create_tif(arr_profile,'img_rrv/NewArea1_500_Ours/'+fn)
               # assumed to be square

                horizontal_path = slide_window.dem_horizontal()
                with rasterio.open(horizontal_path) as img:
                    arr_horizontal = img.read(1)
                    fn = 'Large_ND_Image_horizontal_w='+str(2**num_aggre)+'.tif'

                helper.create_tif(arr_horizontal,'img_rrv/NewArea1_500_Ours/'+fn)

            finally:
                #if (os.path.exists(slope_path)):
                #    os.remove(slope_path)
                print('Done')
                     
    def landscape_extract_subimages_arc(self):
        desigs = ['w3','w8','w16','w32','w64']
#        sub_image_size = 1000
#        sub_image_start_x = 2200
#        sub_image_start_y= 3500
        sub_image_size = 500
        sub_image_start_x = 2300
        sub_image_start_y= 3700
        fn = 'Large_ND_Image.tif'
        with rasterio.open('img_rrv/NewArea1/'+fn) as img:
            arr = img.read(1)
            out_array = np.empty([sub_image_size,sub_image_size])
            for k in np.arange(sub_image_size):
                for j in np.arange(sub_image_size):
                    out_array[j,k] = arr[j+sub_image_start_y,k+sub_image_start_x]
            helper.create_tif(out_array,'img_rrv/NewArea1_500/'+fn)
        for index in range(0,5):
            file_desig = desigs[index]
            for proc_type in ['slope_','prof_','plan_']:
                fn = 'Large_ND_Image_'+proc_type+file_desig+'.tif'
                with rasterio.open('img_rrv/NewArea1/'+fn) as img:
                    arr = img.read(1)
                out_array = np.empty([sub_image_size,sub_image_size])
                for k in np.arange(sub_image_size):
                    for j in np.arange(sub_image_size):
                        out_array[j,k] = arr[j+sub_image_start_y,k+sub_image_start_x]
                helper.create_tif(out_array,'img_rrv/NewArea1_500/'+fn)
                
    def plot_original(self):
        with rasterio.open('img_rrv/NewArea1_500_Ours/Large_ND_Image.tif') as img:
            arr = img.read(1)
        plt.figure()
        #frame1 = plt.gca()
        #frame1.axes.get_xaxis().set_visible(False)
        #frame1.axes.get_yaxis().set_visible(False)
        plt.axis('off')
        plt.imshow(arr)
        #plt.savefig('fig_artificial_ours_no_noise.png',dpi=600,bbox_inches='tight',pad_inches = 0)
        plt.savefig('fig_landscape_image.png',dpi=600,bbox_inches='tight',pad_inches = 0)


self = SlidingWindowComparison()
#self.create_geotiff()
#for num_loc in range(2,self.num_aggre+1):
#    self.compare_multi(num_loc)
#self.qgis_error(1)
#self.intro()
#self.plot_tiff_arc()
#self.plot_tiff_ours()
#self.all_artificial_images()

if self.artificial and False:
    for num_loc in range(2,self.num_aggre+1):
        self.arc_error(num_loc-2)
    
    print('arc_mape')
    print(self.arc_mape)
    print('our_mape')
    print(self.our_mape)
#    fig, axs = plt.subplots(4,1,sharex=True,sharey=False)
    
#    axs[0].scatter([3,8,16,32,64],self.arc_mape[:,0],color="b")
#    axs[0].scatter([4,8,16,32,64],self.our_mape[:,0],color="r")
#    axs[1].scatter([3,8,16,32,64],self.arc_mape[:,1],color="b")
#    axs[1].scatter([4,8,16,32,64],self.our_mape[:,1],color="r")
#    axs[2].scatter([3,8,16,32,64],self.arc_mape[:,2],color="b")
#    axs[2].scatter([4,8,16,32,64],self.our_mape[:,2],color="r")
#    axs[3].scatter([3,8,16,32,64],self.arc_mape[:,3],color="b")
#    axs[3].scatter([4,8,16,32,64],self.our_mape[:,3],color="r")

#    plt.xscale("log")
#    plt.xlim([2.5, 80])
#    plt.xticks([4,8,16,32,64],['4','8','16','32','64'])
#    plt.xlabel('Window Size')
#
#   plt.minorticks_off()
#    if self.noise_true:
#        plt.ylim([-0.05,1.05])
#        axs[0].set_ylim([0,200])
#        axs[1].set_ylim([0,200])
#        axs[2].set_ylim([0,200])
#        axs[3].set_ylim([0,200])
#        plt.suptitle("MAPE With Noise")
#    else:
#        axs[0].set_ylim([-1,10])
#        axs[1].set_ylim([-5,105])
#        axs[2].set_ylim([-5,105])
#        axs[3].set_ylim([-5,105])
#
#        plt.ylim([0.9996, 1.00001])
#        plt.yticks([0.9997,1.000],['0.9997','1.000'])
#        plt.suptitle("MAPE No Noise")
        
#    plt.savefig('fig_mape_no_noise.png',dpi=300,bbox_inches='tight',pad_inches = 0)

    fig, axs = plt.subplots(4,1,sharex=True,sharey=False)
    
    axs[0].scatter([3,8,16,32,64],self.arc_mape[:,0],color="b")
    axs[0].scatter([4,8,16,32,64],self.our_mape[:,0],marker='x',color="r")
    axs[1].scatter([3,8,16,32,64],self.arc_mape[:,1],marker='o',color="b")
    axs[1].scatter([4,8,16,32,64],self.our_mape[:,1],marker='x',color="r")
    axs[2].scatter([3,8,16,32,64],self.arc_mape[:,2],marker='o',color="b")
    axs[2].scatter([4,8,16,32,64],self.our_mape[:,2],marker='x',color="r")
    axs[3].scatter([3,8,16,32,64],self.arc_mape[:,3],marker='o',color="b")
    axs[3].scatter([4,8,16,32,64],self.our_mape[:,3],marker='x',color="r")

    plt.xscale("log")
    plt.xlim([2.5, 80])
    plt.xticks([4,8,16,32,64],['4','8','16','32','64'])
    plt.xlabel('Window Size')

    plt.minorticks_off()
    if self.noise_true:
#        plt.ylim([-0.05,1.05])
        axs[0].set_ylim([0,200])
        axs[1].set_ylim([0,200])
        axs[2].set_ylim([0,200])
        axs[3].set_ylim([0,200])
        plt.suptitle("MAPE With Noise")
    else:
        axs[0].set_ylim([-1,10])
        axs[1].set_ylim([-5,105])
        axs[2].set_ylim([-5,105])
        axs[3].set_ylim([-5,105])

#        plt.ylim([0.9996, 1.00001])
#        plt.yticks([0.9997,1.000],['0.9997','1.000'])
        plt.suptitle("MAPE No Noise")
        plt.savefig('fig_mape_no_noise.png',dpi=300,bbox_inches='tight',pad_inches = 0)

    fig, axs = plt.subplots(4,1,sharex=True,sharey=False)

    
    axs[0].scatter([3,8,16,32,64],self.arc_rsquared[:,0],marker='o',color="b")
    axs[0].scatter([4,8,16,32,64],self.our_rsquared[:,0],marker='x',color="r")
    axs[1].scatter([3,8,16,32,64],self.arc_rsquared[:,1],marker='o',color="b")
    axs[1].scatter([4,8,16,32,64],self.our_rsquared[:,1],marker='x',color="r")
    axs[2].scatter([3,8,16,32,64],self.arc_rsquared[:,2],marker='o',color="b")
    axs[2].scatter([4,8,16,32,64],self.our_rsquared[:,2],marker='x',color="r")
    axs[3].scatter([3,8,16,32,64],self.arc_rsquared[:,3],marker='o',color="b")
    axs[3].scatter([4,8,16,32,64],self.our_rsquared[:,3],marker='x',color="r")

    plt.xscale("log")
    plt.xlim([2.5, 80])
    plt.xticks([4,8,16,32,64],['4','8','16','32','64'])
    plt.xlabel('Window Size')

    plt.minorticks_off()
    if self.noise_true:
#        plt.ylim([-0.05,1.05])
        axs[0].set_ylim([-0.05,1.05])
        axs[1].set_ylim([-0.05,1.05])
        axs[2].set_ylim([-0.05,1.05])
        axs[3].set_ylim([-0.05,1.05])
        plt.suptitle("R Squared With Noise")
        plt.savefig('fig_rsquared_with_noise.png',dpi=300,bbox_inches='tight',pad_inches = 0)
    else:
        axs[0].set_ylim([-0.05,1.05])
        axs[1].set_ylim([-0.05,1.05])
        axs[2].set_ylim([-0.05,1.05])
        axs[3].set_ylim([-0.05,1.05])

#        plt.ylim([0.9996, 1.00001])
#        plt.yticks([0.9997,1.000],['0.9997','1.000'])
        plt.suptitle("R Squared No Noise")
    
#    print('arc_mape')
#    print(self.arc_mape)
#    print('our_mape')
#    print(self.our_mape)

# landscape#print(self.compare_speed(6))

#self.landscape_analyses_ours()
#self.landscape_analyses_arc()
#print('Compare Speed Result')
#print(self.compare_speed(6))
print(self.basic_speed(5))

#self.landscape_extract_subimages_arc()
#self.plot_tiff_arc()
#for num_loc in range(2,self.num_aggre+1):
#    self.landscape_compute_ours(num_loc)
#self.plot_tiff_ours()
#self.plot_original()