# Assumes square pixels!
import gdal #osgeo.gdal as gdal
import os
import sys
import math
from skimage.external import tifffile as tiff
#import Image
import numpy as np
#import time
#from skimage import io

#global filename, width, height, z, xz, yz, GeoT
global filename, width, height, z, xz, yz, xxz, yyz, xyz, GeoT
input_dir = "/Volumes/DataDisk/_Data/_Input/"
output_dir = "/Volumes/DataDisk/_Data/_Output/"
maxuint16 = 65535
xz_signs = np.array((-1,1,-1,1))[:,None]
yz_signs = np.array((-1,-1,1,1))[:,None]
xyz_signs = np.array((1,-1,-1,1))[:,None]

def read_dem_file(fn):
	# read DEM file
    # source_image=Image.open(fn)
    # width, height = source_image.size
    #height = 500
    #width = 500
    #z = np.multiply(np.ones((height,width)),range(0,width))
    z = tiff.imread(fn).astype(float)
    ##z = io.imread(filename)
    width,height = z.shape
    print(width)
    print(height)
    print(z)

    #source_image.close()
    return z, width, height

#def write_dem_file(fn,z,w):
#    np.reshape((width-w+1,height-w+1),z)
#    im = Image.fromarray(z)
#    im.save(fn)


def initialize_arrays(fn,z):
	# initialize global arrays z, xz, yz, xxz, yyz, xyz
	xz = np.zeros(z.shape)
	yz = np.zeros(z.shape)
	xxz = np.zeros(z.shape)
	yyz = np.zeros(z.shape)
	xyz = np.zeros(z.shape)

	src = gdal.Open(fn)
	GeoT = src.GetGeoTransform()
	src = None

	return xz, yz, xxz, yyz, xyz, GeoT
	#return xz, yz, GeoT

def double_w(w_out,z,xz,yz,xxz,yyz,xyz):
    print(w_out)
    delta = int(w_out/2)
    col_extent = width - w_out + 1
    row_extent = height - w_out + 1
    z_loc = np.zeros((row_extent,col_extent))
    xz_loc = np.zeros((row_extent,col_extent))
    yz_loc = np.zeros((row_extent,col_extent))
    xxz_loc = np.zeros((row_extent,col_extent))
    yyz_loc = np.zeros((row_extent,col_extent))
    xyz_loc = np.zeros((row_extent,col_extent))
    for j in range (0, height-w_out+1):
        for i in range (0, width-w_out+1):
			#xyz[width*j+i] = 0.25 * (xyz[width*j+i] + xyz[width*j+(i+delta)] + xyz[width*(j+delta)+i] + xyz[width*(j+delta)+ (i+delta)]
			#	+ 0.25*delta * (-xz[width*j+i] - xz[width*j+(i+delta)] + xz[width*(j+delta)+i] + xz[width*(j+delta)+ (i+delta)])
			#	+ 0.25*delta * (-yz[width*j+i] + yz[width*j+(i+delta)] + yz[width*(j+delta)+i] + yz[width*(j+delta)+ (i+delta)])
			#	+ 0.0625*delta*delta * (z[width*j+i] - z[width*j+(i+delta)] - z[width*(j+delta)+i] + z[width*(j+delta)+ (i+delta)]))
            xxz_loc[j,i] = 0.25 * (xxz[j,i] + xxz[j,i+delta] + xxz[j+delta,i] + xxz[j+delta,i+delta] + 
               delta * (-xz[j,i] + xz[j,i+delta] - xz[j+delta,i] + xz[j+delta,i+delta]) + 
               0.25*delta*delta * (z[j,i] + z[j,i+delta] + z[j+delta,i] + z[j+delta,i+delta]))
            yyz_loc[j,i] = 0.25 * (yyz[j,i] + yyz[j,i+delta] + yyz[j+delta,i] + yyz[j+delta,i+delta] + 
               delta * (-yz[j,i] - yz[j,i+delta] + yz[j+delta,i] + yz[j+delta,i+delta]) + 
               0.25*delta*delta * (z[j,i] + z[j,i+delta] + z[j+delta,i] + z[j+delta,i+delta]))
            xyz_loc[j,i] = 0.25 * (xyz[j,i] + xyz[j,i+delta] + xyz[j+delta,i] + xyz[j+delta,i+delta] + 
               0.5 * delta * (-xz[j,i] - xz[j,i+delta] + xz[j+delta,i] + xz[j+delta,i+delta] - yz[j,i] + yz[j,i+delta] - yz[j+delta,i] + yz[j+delta,i+delta]) + 
               0.25*delta*delta * (z[j,i] - z[j,i+delta] - z[j+delta,i] + z[j+delta,i+delta]))
            xz_loc[j,i] = 0.25 * (xz[j,i] + xz[j,i+delta] + xz[j+delta,i] + xz[j+delta,i+delta] + 
              0.5*delta * (-z[j,i] + z[j,i+delta] - z[j+delta,i] + z[j+delta,i+delta]))
            yz_loc[j,i] = 0.25 * (yz[j,i] + yz[j,i+delta] + yz[j+delta,i] + yz[j+delta,i+delta] + 
              0.5*delta * (-z[j,i] - z[j,i+delta] + z[j+delta,i] + z[j+delta,i+delta]))
            z_loc[j,i] = 0.25 * (z[j,i] + z[j,i+delta] + z[j+delta,i] + z[j+delta,i+delta])
    return z_loc, xz_loc, yz_loc, xxz_loc, yyz_loc, xyz_loc

def double_w_array(w_out,z,xz,yz,xxz,yyz,xyz):
    #print(w_out)
    delta = int(w_out/2)
    z.flatten()
    col_extent_old = width - delta + 1
    col_extent = width - w_out + 1
    row_extent = height - w_out + 1
    # Get Starting block indices
    selector = np.array((0,delta,(delta*col_extent_old),(delta*col_extent_old+delta)))
    # Get offsetted indices across the height and width of input array
    output_index_set = np.arange(row_extent)[:,None]*col_extent_old + np.arange(col_extent)
    full_selector = selector.ravel()[:,None] + output_index_set.ravel()
    big_array_z = np.take(z,full_selector)
    #print(big_array_z)
    #print(big_array_z.shape)
    #print(big_array_z.shape[1])
    z_loc = big_array_z.mean(axis=0)

    big_array_xz = np.take(xz,full_selector) 
    big_array_yz = np.take(yz,full_selector) 
    big_array_xxz = np.take(xxz,full_selector) + 2 * delta * np.multiply(big_array_xz,xz_signs)
    #print("delta: "+str(delta))
    #print("np.multiply(big_array_xz,xz_signs)")
    #print(np.multiply(big_array_xz,xz_signs))
    #print("np.take(xxz,full_selector)")
    #print(np.take(xxz,full_selector))
    xxz_loc = big_array_xxz.mean(axis=0) + 0.25 * delta * delta * z_loc
    big_array_yyz = np.take(yyz,full_selector) + 2 * delta * np.multiply(big_array_yz,yz_signs)
    yyz_loc = big_array_yyz.mean(axis=0) + 0.25 * delta * delta * z_loc
    big_array_xyz = np.take(xyz,full_selector) + delta * (np.multiply(big_array_xz,yz_signs) + np.multiply(big_array_yz,xz_signs)) + 0.25 * delta * delta * np.multiply(big_array_z,xyz_signs)
    xyz_loc = big_array_xyz.mean(axis=0)
    #big_array_xyz = np.take(xyz,full_selector) + \
    #0.5 * delta * (np.multiply(big_array_xz,yz_signs) + np.multiply(big_array_yz,xz_signs)) + \
    #0.25 * delta * delta * np.multiply(big_array_z,xyz_signs)
    #xyz_loc = big_array_xyz.mean(axis=0)
    
    big_array_xz += 0.5 * delta * np.multiply(big_array_z,xz_signs)
    xz_loc = big_array_xz.mean(axis=0)
    big_array_yz += 0.5 * delta * np.multiply(big_array_z,yz_signs)
    yz_loc = big_array_yz.mean(axis=0)

    
    xx = (w*w - 1) / 12.
    #print("xx: "+str(xx))
    #print("xz")
    #print(xz_loc)
    #print(xz_loc[int(xz_loc.size/2)])
    #print("xxz")
    #print(xxz_loc)
    #print(xxz_loc[int(xxz_loc.size/2)])
    #print("yyz")
    #print(yyz_loc)
    #print(yyz_loc[int(xxz_loc.size/2)])
    #print("z")
    #print(z_loc)
    #print(z_loc[int(xxz_loc.size/2)])

    
    return z_loc.reshape((row_extent,col_extent)), xz_loc.reshape((row_extent,col_extent)), yz_loc.reshape((row_extent,col_extent)), \
    xxz_loc.reshape((row_extent,col_extent)), yyz_loc.reshape((row_extent,col_extent)), xyz_loc.reshape((row_extent,col_extent))
			
def window_mean(w,z,fn,GeoT):
    #mean_image = Image.new('F',(width-w+1,height-w+1))
                
    #im = Image.fromarray(mean_array)
    #mean_image.save(fn_loc)
    print("z")
    print(z)
    print(z[int((height-w+1)/2),int((width-w+1)/2)])    
    z_min = np.min(z)
    z_max = np.max(z)
    n = ((z - z_min) / (z_max - z_min) * maxuint16).astype('uint16')
    fn_loc = os.path.splitext(fn)[0] +'_mean_w'+str(w)+'.tif'
    tiff.imsave(fn_loc,n)

    #print_tfw(fn,fn_loc,w,GeoT)
    print_display_tfw(fn,fn_loc,w,GeoT,0)

def slope(w,z,xz,yz,fn,GeoT,prefactor):
    #slope_image = Image.new('F',(width-w+1,height-w+1))
    slope_array = np.zeros((height-w+1,width-w+1))
    xx_inv = 12. / (w*w - 1)
    for j in range (0, height-w+1):
        for i in range (0, width-w+1):
            value = math.atan(prefactor * xx_inv * math.sqrt(xz[j,i]**2 + yz[j,i]**2)/abs(GeoT[1]))
            #print j
			#print value
            #slope_image.putpixel((i,j),value)
            slope_array[j,i]=value

    #im = Image.fromarray(slope_array)
    #slope_image.save(fn_loc)
    print("slope_array")
    print(xx_inv*xz) #print(slope_array)
    print(slope_array[int((height-w+1)/2),int((width-w+1)/2)])
    slope_min = np.min(slope_array)
    slope_max = np.max(slope_array)
    slope_array = (slope_array - slope_min) / (slope_max - slope_min) * maxuint16
    fn_loc = os.path.splitext(fn)[0] +'_slope_w'+str(w)+'.tif'
    n = slope_array.astype('uint16')
    tiff.imsave(fn_loc,n)
    #print_tfw(fn,fn_loc,w,GeoT)
    print_display_tfw(fn,fn_loc,w,GeoT,3)

# Sign of pixel directions considered here and in directional curvaturen terms
def aspect(w,z,xz,yz,fn,GeoT):
    #aspect_image = Image.new('F',(width-w+1,height-w+1))
    aspect_array = np.zeros((height-w+1,width-w+1))
    factor = maxuint16 / (2*math.pi)
    for j in range (0, height-w+1):
        for i in range (0, width-w+1):
            xz_loc = xz[j,i]
            yz_loc = yz[j,i]
            if yz_loc == 0:
                if xz_loc < 0:
                    value = 1.5*math.pi
                else:
                    value = 0.5*math.pi
            else:
                value = -math.atan(xz_loc/yz_loc)
                if yz_loc < 0:
                     value += math.pi
                elif xz_loc > 0:
                    value += 2*math.pi
            #aspect_image.putpixel((i,j),value)
            aspect_array[j,i]=value*factor
    #im = Image.fromarray(aspect_array)
    #im.save(fn_loc)
    print("aspect_array")
    print(aspect_array)
    fn_loc = os.path.splitext(fn)[0] +'_aspect_w'+str(w)+'.tif'
    n = aspect_array.astype('uint16')
    tiff.imsave(fn_loc,n)
    #print_tfw(fn,fn_loc,w,GeoT)
    print_display_tfw(fn,fn_loc,w,GeoT,1)

def norm_curv(w,z,xz,yz,xxz,yyz,fn,GeoT):
    curv_array = np.empty((height-w+1,width-w+1))
    
    xx = (w*w - 1) / 12.
    inv_x4mxx2 = 180. / (w*w*w*w - 5*w*w + 4)
    print("inv_x4mxx2: "+str(inv_x4mxx2))
    print("curv_array")
    curv_array = (0.5*(xxz + yyz) - np.multiply(xx,z)) * inv_x4mxx2
    #curv_array = (xxz - np.multiply(xx,z)) * inv_x4mxx2
    print(curv_array)
    print(curv_array[int((height-w+1)/2),int((width-w+1)/2)])
    curv_min = np.min(curv_array)
    print("curv_min: "+str(curv_min))
    curv_max = np.max(curv_array)
    curv_array = (curv_array - curv_min) / (curv_max - curv_min) * maxuint16
    
    fn_loc = os.path.splitext(fn)[0] +'_curv_w'+str(w)+'.tif'
    n = curv_array.astype('uint16')
    tiff.imsave(fn_loc,n)
    #print_tfw(fn,fn_loc,w,GeoT)
    print_display_tfw(fn,fn_loc,w,GeoT,2)

def all_curv(w,z,xz,yz,xxz,yyz,xyz,fn,GeoT):
    curv_array = np.empty((height-w+1,width-w+1))
    
    xx = (w*w - 1) / 12.
    inv_xx_sq = 1./(xx*xx)
    inv_x4mxx2 = 180. / (w*w*w*w - 5*w*w + 4)
    print("inv_x4mxx2: "+str(inv_x4mxx2))
    print("curv_array")
    curv_array = (0.5*(xxz + yyz) - np.multiply(xx,z)) * inv_x4mxx2
    #curv_array = (xxz - np.multiply(xx,z)) * inv_x4mxx2
    print(curv_array)
    print(curv_array[int((height-w+1)/2),int((width-w+1)/2)])
    
    print("some tests")
    print(np.divide(xz*xz+xz*xz,xz*xz+xz*xz))
    print("z: "+str(z))
    print("xz: "+str(xz))
    print("yz: "+str(yz))
    print("xxz: "+str(xxz))
    print("yyz: "+str(yyz))
    print("xyz: "+str(xyz))

    pmp_array = np.divide((0.5 * (xxz - yyz) * inv_x4mxx2 * (xz * xz - yz * yz) + xyz * inv_xx_sq * xz * yz), (xz * xz + yz * yz)) 
    #pmp_array = np.divide((0.5 * (xxz - yyz) * (xz * xz - yz * yz) + xyz * xz * yz), (inv_x4mxx2 *(xz * xz + yz * yz))) 
    print("pmp: "+str(pmp_array))
    
    curv_min = np.amin(curv_array)
    print("curv_min: "+str(curv_min))
    curv_max = np.amax(curv_array)    
    print("curv_max: "+str(curv_max))
    fn_loc = os.path.splitext(fn)[0] +'_curv_w'+str(w)+'.tif'
    n = ((curv_array - curv_min) / (curv_max - curv_min) * maxuint16).astype('uint16')
    tiff.imsave(fn_loc,n)
    #print_tfw(fn,fn_loc,w,GeoT)
    print_display_tfw(fn,fn_loc,w,GeoT,2)
    
    prof_array = curv_array + pmp_array
    prof_min = np.amin(prof_array)
    print("prof_min: "+str(prof_min))
    prof_max = np.amax(prof_array)    
    print("prof_max: "+str(prof_max))
    fn_loc = os.path.splitext(fn)[0] +'_prof_w'+str(w)+'.tif'
    n = ((prof_array - prof_min) / (prof_max - prof_min) * maxuint16).astype('uint16')
    tiff.imsave(fn_loc,n)
    #print_tfw(fn,fn_loc,w,GeoT)
    print_display_tfw(fn,fn_loc,w,GeoT,2)

    plan_array = curv_array - pmp_array
    plan_min = np.amin(plan_array)
    print("plan_min: "+str(plan_min))
    plan_max = np.amax(plan_array)    
    print("plan_max: "+str(plan_max))
    fn_loc = os.path.splitext(fn)[0] +'_plan_w'+str(w)+'.tif'
    n = ((plan_array - plan_min) / (plan_max - plan_min) * maxuint16).astype('uint16')
    tiff.imsave(fn_loc,n)
    #print_tfw(fn,fn_loc,w,GeoT)
    print_display_tfw(fn,fn_loc,w,GeoT,2)

def profile(w,z,xz,yz,fn,GeoT):
    curv_array = np.empty((height-w+1,width-w+1))
    
    xx = (w*w - 1) / 12.
    inv_x4mxx2 = 180. / (w*w*w*w - 5*w*w + 4)
    print("inv_x4mxx2: "+str(inv_x4mxx2))
    print("curv_array")
    curv_array = (0.5*(xxz + yyz) - np.multiply(xx,z)) * inv_x4mxx2
    #curv_array = (xxz - np.multiply(xx,z)) * inv_x4mxx2
    print(curv_array)
    print(curv_array[int((height-w+1)/2),int((width-w+1)/2)])
    curv_min = np.mina(curv_array)
    print("curv_min: "+str(curv_min))
    curv_max = np.maxa(curv_array)
    curv_array = (curv_array - curv_min) / (curv_max - curv_min) * maxuint16
    
    fn_loc = os.path.splitext(fn)[0] +'_profile_w'+str(w)+'.tif'
    n = curv_array.astype('uint16')
    tiff.imsave(fn_loc,n)
    #print_tfw(fn,fn_loc,w,GeoT)
    print_display_tfw(fn,fn_loc,w,GeoT,3)

def planform(w,z,xz,yz,fn,GeoT):
    curv_array = np.empty((height-w+1,width-w+1))
    
    xx = (w*w - 1) / 12.
    inv_x4mxx2 = 180. / (w*w*w*w - 5*w*w + 4)
    print("inv_x4mxx2: "+str(inv_x4mxx2))
    print("curv_array")
    curv_array = (0.5*(xxz + yyz) - np.multiply(xx,z)) * inv_x4mxx2
    #curv_array = (xxz - np.multiply(xx,z)) * inv_x4mxx2
    print(curv_array)
    print(curv_array[int((height-w+1)/2),int((width-w+1)/2)])
    curv_min = np.min(curv_array)
    print("curv_min: "+str(curv_min))
    curv_max = np.max(curv_array)
    curv_array = (curv_array - curv_min) / (curv_max - curv_min) * maxuint16
    
    fn_loc = os.path.splitext(fn)[0] +'_plan_w'+str(w)+'.tif'
    n = curv_array.astype('uint16')
    tiff.imsave(fn_loc,n)
    #print_tfw(fn,fn_loc,w,GeoT)
    print_display_tfw(fn,fn_loc,w,GeoT,4)

#def print_file():
	# for each value of w print mu (z), sigma, slope, normcurv, profile, planform

def print_tfw(fn_orig,fn,w,GeoT):
    topLeftPixelCenterX=GeoT[0]+(w*GeoT[1])/2 # Top left corner of original image is now in center of window
    topLeftPixelCenterY=GeoT[3]+(w*GeoT[5])/2 # Note that this still works for w=1, because TFW requires center of first pixel

    tfw = open(os.path.splitext(fn)[0] +'.tfw', 'wt')
    tfw.write("%0.8f\n" % GeoT[1]) # pixel width
    tfw.write("%0.8f\n" % GeoT[2]) # 0 for unrotated frame
    tfw.write("%0.8f\n" % GeoT[4]) # 0 for unrotated frame
    tfw.write("%0.8f\n" % GeoT[5]) # pixel height
    tfw.write("%0.8f\n" % topLeftPixelCenterX) 
    tfw.write("%0.8f\n" % topLeftPixelCenterY)
    tfw.close()

def print_display_tfw(fn_orig,fn,w,GeoT,column):
    display_diff = 20
    topLeftPixelCenterX=GeoT[0]+(w*GeoT[1])/2 # Top left corner of original image is now in center of window
    topLeftPixelCenterY=GeoT[3]+(w*GeoT[5])/2 # Note that this still works for w=1, because TFW requires center of first pixel

    row = int(round(math.log2(w))-1)
    print("row: "+str(row))
    tfw = open(os.path.splitext(fn)[0] +'.tfw', 'wt')
    tfw.write("%0.8f\n" % GeoT[1]) # pixel width
    tfw.write("%0.8f\n" % GeoT[2]) # 0 for unrotated frame
    tfw.write("%0.8f\n" % GeoT[4]) # 0 for unrotated frame
    tfw.write("%0.8f\n" % GeoT[5]) # pixel height
    tfw.write("%0.8f\n" % (topLeftPixelCenterX + (width+display_diff)*GeoT[1]*column))
    tfw.write("%0.8f\n" % (topLeftPixelCenterY + (height+display_diff)*GeoT[5]*row))
    tfw.close()

input_fn = input_dir+sys.argv[1]
output_fn = output_dir+sys.argv[1]
w_end = int(sys.argv[2])
# To Do: Check power of two

print(input_fn)
print(output_fn)
print(w_end)

z, width, height = read_dem_file(input_fn)
#width = 100
#height = 100
print(width)
print(height)
xz, yz, xxz, yyz, xyz, GeoT = initialize_arrays(input_fn,z)
#xz, yz, xxz, yyz, xyz, GeoT = initialize_arrays(filename,z)
#xz, yz, xxz, yyz, xyz, GeoT = initialize_arrays('DEM_reraster_small.tif',z)

w = 1
while w < w_end:
    w*=2
    #z, xz, yz, xxz, yyz = double_w_array(w,z,xz,yz,xxz,yyz)
    z, xz, yz, xxz, yyz, xyz = double_w(w,z,xz,yz,xxz,yyz,xyz)
    window_mean(w,z,output_fn,GeoT)
    #write_dem_file(filename,z,w)
    slope(w,z,xz,yz,output_fn,GeoT,1)
    aspect(w,z,xz,yz,output_fn,GeoT)
    if w>2:
        #norm_curv(w,z,xz,yz,xxz,yyz,output_fn,GeoT)
        all_curv(w,z,xz,yz,xxz,yyz,xyz,output_fn,GeoT)

