# Note: The obvious choice of classes / objects for encapsulating arrays with their respective image and window size was not used to simplify migration of time critical functions to c
# Instead, all 2-dimensional arrays are stored as 1-dimensional arrays with indexes into the array calculated on the fly.  Only the existing numbers are stored, not the frame, so the sizes x_diff and y_diff start out as x_max and y_max, but then they shrink in each iteration.
# Similarly:  Some of the array operations would have been more straightforward using Python's array offset capabilities, but that too would have created compatibility issues

from PIL import Image
from skimage.external import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

#import pylab

# Creates pixel array from RapidEye images
# x_start, y_start: pixel indices in image where reading begins
# x_max, y_max: array dimensions
def readImageNoConv (fn,a,x_start,y_start,x_max,y_max):

    source_image=Image.open(fn)

    for j in range (0, y_max):
        for i in range (0, x_max):
            value = source_image.getpixel((x_start+i,y_start+j))
            a[x_max*j+i] = value

# Creates pixel array from Landsat images
# x_start, y_start: pixel indices in image where reading begins
# x_max, y_max: array dimensions
def readImageConv (fn,a,x_start,y_start,x_max,y_max):
    
    source_image=Image.open(fn)
    source_image_conv = source_image.convert("F")
    
    for j in range (0, y_max):
        for i in range (0, x_max):
            value = source_image_conv.getpixel((x_start+i,y_start+j))
            a[x_max*j+i] = value

# Convert continuous input into binary based on mean
def thresholdArray (a_in):

    a_thresh = np.mean(a_in) #sorted(a_in)[int(round(a_in.size*percent))]
    print(a_thresh)
    a_out = np.zeros(a_in.size)
    a_out[(a_in >= a_thresh)] = 1
    return a_out

# Convert continuous input into binary based on explicit threshold
def thresholdArrayMan (a_in,a_thresh):
    
    #a_thresh = np.mean(a_in) #sorted(a_in)[int(round(a_in.size*percent))]
    print(a_thresh)
    a_out = np.zeros(a_in.size)
    a_out[(a_in >= a_thresh)] = 1
    return a_out

# Same as previous but inverted image is output
def thresholdArrayNeg (a_in,a_thresh):
    
    #a_thresh = np.mean(a_in) #sorted(a_in)[int(round(a_in.size*percent))]
    print(a_thresh)
    a_out = np.zeros(a_in.size)
    a_out[(a_in < a_thresh)] = 1
    return a_out

# This is for the 3D fractal dimension that is between 2 and 3, but it isn't tested yet
def boxedArray(a_in,power_target):

    a_min = np.min(a_in)
    a_max = np.max(a_in)
    a_out = np.zeros(a_in.size)
    if (a_max > a_min):
        a_out = np.zeros(a_in.size)
        n_boxes = 2**power_target-1
        buffer = (a_in-a_min)/(a_max-a_min)
        a_out = np.floor(n_boxes * buffer)
    return a_out

# Selects those pixels in an integer array that have one of a set of values, used for landcover
def selectArray (a_in,select_set):
    
    print(select_set)
    a_out = np.zeros(a_in.size)
    for a_value in select_set:
        a_out[(a_in == a_value)] = 1
    
    return a_out

# shows how many of each landcover were in the image
def selectStatistics (a_in):
    
    for i in range(0,256):
        print((i,np.count_nonzero(a_in == i)))

def printCorrelation (a_in,b_in):
    
    for i in range(0,256):
        count = np.count_nonzero(a_in == i)
        print((i, count, np.sum(a_in[a_in == i]*b_in[a_in == i])))

# Creates composite graphs with x_rep different images left to right and y_rep top to bottom
# Each view assumes that the original image had width x_max wide and height y_max
def createImageRGB (x_max,y_max,x_rep,y_rep):
    out_image = Image.new('RGB',(x_max*x_rep,y_max*y_rep))
    out_max = 255
    for j in range (0, y_max*y_rep):
        for i in range (0, x_max*x_rep):
            out_image.putpixel((i,j),(out_max,out_max,out_max))

    return out_image

#def createImageL (x_max,y_max,x_rep,y_rep):
#    out_image = Image.new('L',(x_max*x_rep,y_max*y_rep))
#    out_max = 255
#    for j in range (0, y_max*y_rep):
#        for i in range (0, x_max*x_rep):
#            out_image.putpixel((i,j),out_max)

#    return out_image

# Writes one window-based image to the composite image for the replicate with indices x_index / y_index
# It is assumed that the original image was of size x_max, y_max
# For window-size > 1 a border of width w/2 is created to compensate for the smaller size of the image
# Assumption: input values between 0 and 1
def writeToImage(out_image,red,green,blue,x_max,y_max,w,x_index,y_index):
    x_diff = x_max-w+1
    y_diff = y_max-w+1
    out_max = 255
    if (w>1):
        offset_x = x_max*x_index + w//2-1
        offset_y = y_max*y_index + w//2-1
    else:
        offset_x = x_max*x_index
        offset_y = y_max*y_index

    red_clip = np.empty(red.size)
    #print "red",red.size
    #print "red_clip",red_clip.size
    green_clip = np.empty(green.size)
    blue_clip = np.empty(blue.size)
    np.clip(red,0,1,red_clip)
    np.clip(green,0,1,green_clip)
    np.clip(blue,0,1,blue_clip)
    np.nan_to_num(red_clip)
    np.nan_to_num(green_clip)
    np.nan_to_num(blue_clip)
    #print np.isnan(green_clip)
    for j in range (0, y_diff):
        for i in range (0, x_diff):
            out_image.putpixel((offset_x+i,offset_y+j),(int(round(out_max*red_clip[x_diff*j+i])),int(round(out_max*green_clip[x_diff*j+i])),int(round(out_max*blue_clip[x_diff*j+i]))))

# Does one window-aggregation step of a pixel array a_in that had been processed up to window size w_out/2 creating one with window size w_out
# The relevant size remains x_max, y_max throughout, although the actual storage of a_in and a_out is smaller because windows cannot overlap right or bottom of image
def windowSum (a_in,x_max,y_max,w_out):
    w_in = w_out//2
    x_diff_in = x_max-w_in+1
    x_diff_out = x_max-w_out+1
    y_diff_out = y_max-w_out+1
    a_out = np.empty(x_diff_out*y_diff_out)
    for j in range (0, y_diff_out):
        for i in range (0, x_diff_out):
            a_out[x_diff_out*j+i] = a_in[x_diff_in*j+i] + a_in[x_diff_in*j+(i+w_in)] + a_in[x_diff_in*(j+w_in)+i] + a_in[x_diff_in*(j+w_in)+(i+w_in)]
    return a_out

# This is for the window-based fractal dimension
def windowMax (a_in,x_max,y_max,w_out):
    w_in = w_out//2
    x_diff_in = x_max-w_in+1
    x_diff_out = x_max-w_out+1
    y_diff_out = y_max-w_out+1
    a_out = np.empty(x_diff_out*y_diff_out)
    for j in range (0, y_diff_out):
        for i in range (0, x_diff_out):
            a_out[x_diff_out*j+i] = max(max(a_in[x_diff_in*j+i],a_in[x_diff_in*j+(i+w_in)]),max(a_in[x_diff_in*(j+w_in)+i],a_in[x_diff_in*(j+w_in)+(i+w_in)]))
    return a_out

def windowMin (a_in,x_max,y_max,w_out):
    w_in = w_out//2
    x_diff_in = x_max-w_in+1
    x_diff_out = x_max-w_out+1
    y_diff_out = y_max-w_out+1
    a_out = np.empty(x_diff_out*y_diff_out)
    for j in range (0, y_diff_out):
        for i in range (0, x_diff_out):
            a_out[x_diff_out*j+i] = min(min(a_in[x_diff_in*j+i],a_in[x_diff_in*j+(i+w_in)]),min(a_in[x_diff_in*(j+w_in)+i],a_in[x_diff_in*(j+w_in)+(i+w_in)]))
    return a_out

# This is for finding shorelines
def windowXOR (a_in,x_max,y_max,w_out):
    eps = 0.001
    w_in = w_out//2
    x_diff_in = x_max-w_in+1
    x_diff_out = x_max-w_out+1
    y_diff_out = y_max-w_out+1
    a_out = np.empty(x_diff_out*y_diff_out)
    for j in range (0, y_diff_out):
        for i in range (0, x_diff_out):
            asum = a_in[x_diff_in*j+i] + a_in[x_diff_in*j+(i+w_in)] + a_in[x_diff_in*(j+w_in)+i] + a_in[x_diff_in*(j+w_in)+(i+w_in)]
            amax = max(max(a_in[x_diff_in*j+i],a_in[x_diff_in*j+(i+w_in)]),max(a_in[x_diff_in*(j+w_in)+i],a_in[x_diff_in*(j+w_in)+(i+w_in)]))
            a_out[x_diff_out*j+i] = (4*amax - asum) > eps #and (asum - amax) > eps
    return a_out

# Second step of counting pixels as part of the comparison algorithm
def windowCollectSumWindowBrute(a_in,x_max,y_max,power_start,power_target):
    w_start = 2**power_start
    w_target = 2**power_target
    
    x_diff_in = x_max-w_start+1
    x_diff_out = x_max-w_target+1
    y_diff_out = y_max-w_target+1
    a_sum = np.empty(x_diff_out*y_diff_out)
    for j in range (0, y_diff_out):
        for i in range (0, x_diff_out):
            aw = np.float(0)
            
            for jw in range (0, int(round(w_target/w_start))):
                for iw in range (0, int(round(w_target/w_start))):
                    aw += a_in[x_diff_in*(j+w_start*jw)+(i+w_start*iw)]
    
            a_sum[x_diff_out*j+i] = aw
            #print x_diff_out, y_diff_out, j, i, jw, iw, x_diff_out*j+i
    a_sum = np.maximum(a_sum,1)
    return a_sum

# First step of the comparison algorithm
def windowMaxBrute(a_in,x_max,y_max,w):
    x_diff_out = x_max-w+1
    y_diff_out = y_max-w+1
    a_out = np.empty(x_diff_out*y_diff_out)
    for jglobal in range(0,y_max-w+1):
        for iglobal in range(0,x_max-w+1):
            max_loc = 0
            for ispec in range(0,w):
                for jspec in range(0,w):
                    loc_value = a_in[x_max*(jglobal+jspec)+iglobal+ispec]
                    max_loc = max(max_loc,loc_value)
            a_out[x_diff_out*jglobal+iglobal] = max_loc
    return a_out

# Sum aggregation of the windows used for calculating fractal dimensions, applied to the whole image
def windowCollectSum (a,x_max,y_max,w):
    loc_sum = 0
    x_diff = x_max-w+1
    y_diff = y_max-w+1

    for j in range (0, y_max/w):
        for i in range (0, x_max/w):
            loc_sum += a[x_diff*w*j+w*i]
    return loc_sum

# Aggregation portion of the fractal dimension for sliding window algorithm
# The minimum is 1 to avoid log(0)
def windowCollectSumWindow (a_in,x_max,y_max,power_start,power_target):
    w_out = 2**(power_start+1)
    for i in range (power_start,power_target):
        a_out = windowSum(a_in,x_max,y_max,w_out)
        
        w_out *= 2
        a_in = a_out
    
    a_out = np.maximum(a_out,1)
    return a_out

# Same as above without removing 0s
def windowCollectSumWindowNoMin (a_in,x_max,y_max,power_start,power_target):
    w_out = 2**(power_start+1)
    for i in range (power_start,power_target):
        a_out = windowSum(a_in,x_max,y_max,w_out)
        
        w_out *= 2
        a_in = a_out
    
    return a_out

# Regression paper
# Create an array of window-based regression slopes for precalculated sums of x (a) and y (b), sums of squares of x (aa) and sums of xy (ab)
# Uses the original image dimensions x_max and y_max and window size w.  Note that the actual array sizes are smaller than x_max*y_max
def windowRegression(a,b,aa,ab,x_max,y_max,w):
    x_diff = x_max-w+1
    y_diff = y_max-w+1
    size = x_diff*y_diff
    count = w*w
    m = np.empty(size)
    if (size != a.size) or (size != b.size) or (size != aa.size) or (size != ab.size) or (size != b.size):
        print(size)
        print((a.size))
        raise ValueError('In windowRegression: x_max and/or y_max inconsistent with length of a, b, aa, and / or ab')

    numerator = count * ab - a * b
    denominator = count * aa - a * a
    m = numerator/denominator
    return m

# Regression paper
# Create an array of window-based R quared for precalculated sums of x (a) and y (b), sums of squares of x (aa) and y (bb) and sums of xy (ab)
# Uses the original image dimensions x_max and y_max and window size w.  Note that the actual array sizes are smaller than x_max*y_max
def windowRSquared(a,b,aa,ab,bb,m,x_max,y_max,w):
    x_diff = x_max-w+1
    y_diff = y_max-w+1
    size = x_diff*y_diff
    count = w*w
    r2 = np.empty(size)
    if (size != a.size) or (size != b.size) or (size != aa.size) or (size != ab.size) or (size != b.size) :
        raise ValueError('In windowRSquared: x_max and/or y_max inconsistent with length of a, b, aa, ab, and / or bb')
    
    s_reg = 2*m * (count*ab - a*b) - m*m * (count*aa - a*a)
    s_tot = count*bb - b*b
    r2 = s_reg/s_tot
    return r2

# Regression paper
# Create an array of window-based R quared for precalculated sums of x (a) and y (b), sums of squares of x (aa) and y (bb) and sums of xy (ab)
# Uses the original image dimensions x_max and y_max and window size w.  Note that the actual array sizes are smaller than x_max*y_max
def windowPearson(a,b,aa,ab,bb,x_max,y_max,w):
    x_diff = x_max-w+1
    y_diff = y_max-w+1
    size = x_diff*y_diff
    count = w*w
    pearson = np.empty(count)
    if (size != a.size) or (size != b.size) or (size != aa.size) or (size != ab.size) or (size != b.size) :
        raise ValueError('In windowPearson: x_max and/or y_max inconsistent with length of a, b, aa, ab, and / or bb')
    
    numerator = count*ab - a*b
    denominator = np.sqrt(count * aa - a*a) * np.sqrt(count * bb - b*b)
    pearson = numerator / denominator
    return pearson

# Regression paper
# Do all aggregations up to power_target and return the regression slope
def windowRegressionArray(a_in,b_in,x_max,y_max,power_target):
    aa_in = a_in*a_in
    bb_in = b_in*b_in
    ab_in = a_in*b_in
    
    w_out = 1
    for result_index in range(0,power_target):
        w_out *= 2

        a_out = windowSum(a_in,x_max,y_max,w_out)
        b_out = windowSum(b_in,x_max,y_max,w_out)
        aa_out = windowSum(aa_in,x_max,y_max,w_out)
        ab_out = windowSum(ab_in,x_max,y_max,w_out)
        bb_out = windowSum(bb_in,x_max,y_max,w_out)
        
        a_in = a_out
        b_in = b_out
        aa_in = aa_out
        ab_in = ab_out
        bb_in = bb_out

    m = windowRegression(a_out,b_out,aa_out,ab_out,x_max,y_max,w_out)
    return m

def windowFractalArrayBrute(a_in,x_max,y_max,power_start,power_target):
    print((power_start, power_target))
    b_in = thresholdArray(a_in)
    x_diff_out = x_max-2**power_target+1
    y_diff_out = y_max-2**power_target+1
    denom_regress = np.empty(power_target-power_start)
    num_regress = np.zeros((power_target-power_start,x_diff_out*y_diff_out))
    
    if 0 < power_start:
        print("0<power_start")
        b_in = windowMaxBrute(b_in,x_max,y_max,2**power_start)
    
    for result_index in range(power_start,power_target):
        print(("brute result_index",result_index))
    
        b_out = windowMaxBrute(b_in,x_max,y_max,2**result_index)
        #print "brute b_out", b_out
        sum_array = windowCollectSumWindowBrute(b_out,x_max,y_max,result_index,power_target)
        #print "brute algorithm: ", np.reshape(sum_array,(x_max-2**power_target+1,-1))
        num_array = np.log(sum_array)/np.log(2)
        #print "brute logarithm: ", np.reshape(num_array,(x_max-2**power_target+1,-1))
        denom = power_target-result_index
        #print sum_b_out, numerator, denominator, numerator/denominator, epsilon
        denom_regress[result_index-power_start] = denom
        num_regress[result_index-power_start,] = np.transpose(num_array)
        #print "brute num_regress", num_regress
    
    #print "denom_regress", denom_regress
    #print "num_regress", num_regress
    #print "polyfit", np.polyfit(denom_regress,num_regress,1,full=True)
    f_out, error_out, rank_out, singular_out, rcond_out = np.polyfit(denom_regress,num_regress,1,full=True)
    #print "fractal dimension ", f_out #np.reshape(f_out,(x_max-2**power_target+1,-1))
    return f_out, error_out

def windowFractalArray(b_out,x_max,y_max,power_start,power_target):
    print((power_start, power_target))
    w_out = 1
    x_diff_out = x_max-2**power_target+1
    y_diff_out = y_max-2**power_target+1
    denom_regress = np.empty(power_target-power_start)
    num_regress = np.empty((power_target-power_start,x_diff_out*y_diff_out))
    
    for result_index in range(0,power_start):
        b_in = b_out
        w_out *= 2
        b_out = windowMax(b_in,x_max,y_max,w_out)
        print(w_out)
        print((np.reshape(b_out,(x_max-w_out+1,-1))))

    time_sum_array = np.empty(power_target-power_start)
    time_max_array = np.empty(power_target-power_start)
    time_max_array[0] = 0

    start_time = time.time()

    for result_index in range(power_start,power_target):
        #print "result_index",result_index
        print(w_out)
        #print "input: ", np.reshape(b_out,(x_max-w_out+1,-1))
        #sum_array = windowCreateSumWindow(b_out,x_max,y_max,result_index,power_target)
        #print "comparison: ", np.reshape(sum_array,(x_max-2**power_target+1,-1))
        sum_array = windowCollectSumWindow(b_out,x_max,y_max,result_index,power_target)
        
        time_sum_array[result_index-power_start] = time.time() - start_time
        print(("result_index",result_index,"time", time_sum_array[result_index-power_start]))
        start_time = time.time()
        
        #print "new algorithm: ", np.reshape(sum_array,(x_max-2**power_target+1,-1))
        num_array = np.log(sum_array)/np.log(2)
        #print "logarithm: ", np.reshape(num_array,(x_max-2**power_target+1,-1))
        denom = power_target-result_index
        #print sum_b_out, numerator, denominator, numerator/denominator, epsilon
        denom_regress[result_index-power_start] = denom
        num_regress[result_index-power_start,] = np.transpose(num_array)
        #print "num_regress", num_regress
        if result_index < power_target-1:
            b_in = b_out
            w_out *= 2
            b_out = windowMax(b_in,x_max,y_max,w_out)
        
            time_max_array[result_index-power_start+1] = time.time() - start_time
            print(("result_index",result_index,"time", time_max_array[result_index-power_start+1]))
            start_time = time.time()

    print(time_sum_array)
    print(time_max_array)

    #print "denom_regress", denom_regress
    #print "num_regress", num_regress
    #print "polyfit", np.polyfit(denom_regress,num_regress,1,full=True)
    f_out, error_out, rank_out, singular_out, rcond_out = np.polyfit(denom_regress,num_regress,1,full=True)
    #print "fractal dimension ", f_out #np.reshape(f_out,(x_max-2**power_target+1,-1))
    return f_out, error_out

# Not tested, not part of the paper
def windowFractal3d(a_in,x_max,y_max,power_target):
    w_out = 1
    x_diff_out = x_max-2**power_target+1
    y_diff_out = y_max-2**power_target+1
    b_min = boxedArray(a_in,power_target)
    print(b_min)
    b_max = np.copy(b_min)
    denom_regress = np.empty(power_target-1)
    num_regress = np.empty((power_target-1,x_diff_out*y_diff_out))
    
    for result_index in range(1,power_target):
        print(w_out)
        print(b_min)
        print(b_max)
        print((b_max-b_min+1))
        w_out *= 2
        b_min = windowMin(b_min,x_max,y_max,w_out)
        b_max = windowMax(b_max,x_max,y_max,w_out)
        sum_array = windowCollectSumWindow(b_max-b_min+1,x_max,y_max,result_index,power_target)
        num_array = np.log(sum_array)/np.log(2)
        denom = power_target-result_index
        denom_regress[result_index-1] = denom
        num_regress[result_index-1,] = np.transpose(num_array)
        b_min = b_min / 2
        b_max = b_max / 2 

    print(denom_regress)
    print(num_regress)
    f_out, error_out, rank_out, singular_out, rcond_out = np.polyfit(denom_regress,num_regress,1,full=True)
    print(f_out)
    return f_out, error_out

# Regression paper
def windowRegressionArrayBrute(a_in,b_in,x_max,y_max,power_target):
    w_out = 2**power_target

    x_diff_out = x_max-w_out+1
    y_diff_out = y_max-w_out+1
    m = np.empty(x_diff_out*y_diff_out)
    for j in range (0, y_diff_out):
        for i in range (0, x_diff_out):
            aw = np.float(0)
            bw = np.float(0)
            aaw = np.float(0)
            abw = np.float(0)
            bbw = np.float(0)
            
            for jw in range (0, w_out):
                for iw in range (0, w_out):
                    al = a_in[x_max*(j+jw)+(i+iw)]
                    bl = b_in[x_max*(j+jw)+(i+iw)]
                    aw += al
                    bw += bl
                    aaw += al*al
                    abw += al*bl
                    bbw += bl*bl
        
            numerator = w_out*w_out * abw - aw * bw
            denominator = w_out*w_out * aaw - aw * aw
            m[x_diff_out*j+i] = numerator/denominator

    return m

# Regression paper
# Function for testing purposes; compares logarithmic approach against brute force approach
def testWindowRegressionArray():
    x_max = 10
    y_max = 10
    power_target = 2
    a_in = np.empty(x_max*y_max)
    b_in = np.empty(x_max*y_max)
    z = np.zeros(x_max*y_max)
    
    readImageNoConv('r322_14b_red.tif',a_in,0,0,x_max,y_max)
    readImageNoConv('r322_14b_nir.tif',b_in,0,0,x_max,y_max)
    # Timing code to be added
    m1 = windowRegressionArray(a_in,b_in,x_max,y_max,power_target)
    print(m1)
    m2 = windowRegressionArrayBrute(a_in,b_in,x_max,y_max,power_target)
    print(m2)

# Used for deciding on color
def high(f_out0,f_error):
    #return f_out0*np.exp(-10*f_error)*np.sin(f_out0*np.pi)**2
    return np.exp(-10*f_error)*np.sin(f_out0*np.pi)**2

def sin_filter(f_out0,f_error):
    return np.exp(-10*f_error)*np.sin(f_out0*np.pi)

def low(f_out0,f_error):
    return (2-f_out0)*np.exp(-10*f_error)*np.sin(f_out0*np.pi)**2

def blue(f_out0,f_error):
    return 1-(2*np.exp(-10*f_error)*np.sin(f_out0*np.pi)**2)


def fractality(f_out0,f_error):
    return 2*np.exp(-10*f_error)*np.sin(f_out0*np.pi)**2

# Example Application
def createTestPlot():
    x_max = 300
    y_max = 300
    power_target = 5
    a_in = np.empty(x_max*y_max)
    b_in = np.empty(x_max*y_max)
    z = np.zeros(x_max*y_max)

    readImageNoConv('r322_14b_red.tif',a_in,0,0,x_max,y_max)
    readImageNoConv('r322_14b_nir.tif',b_in,0,0,x_max,y_max)
    out_image = createImageRGB(x_max,y_max,power_target,3)
    aa_in = a_in*a_in
    bb_in = b_in*b_in
    ab_in = a_in*b_in

    w_out = 2
    for result_index in range(0,power_target):
        a_out = windowSum(a_in,x_max,y_max,w_out)
        b_out = windowSum(b_in,x_max,y_max,w_out)
        aa_out = windowSum(aa_in,x_max,y_max,w_out)
        ab_out = windowSum(ab_in,x_max,y_max,w_out)
        bb_out = windowSum(bb_in,x_max,y_max,w_out)
        m = windowRegression(a_out,b_out,aa_out,ab_out,x_max,y_max,w_out)
        #print 'm= ' + str(m)
        r2 = windowRSquared(a_out,b_out,aa_out,ab_out,bb_out,m,x_max,y_max,w_out)
        pearson = windowPearson(a_out,b_out,aa_out,ab_out,bb_out,x_max,y_max,w_out)

        #print 'NDVI ='+str((b_out-a_out)/(b_out+a_out))
        writeToImage(out_image,z,((b_out-a_out)/(b_out+a_out)),z,x_max,y_max,w_out,result_index,0)
        writeToImage(out_image,0.1*m,-0.1*m,z,x_max,y_max,w_out,result_index,1)
        writeToImage(out_image,pearson,-pearson,z,x_max,y_max,w_out,result_index,2)

        w_out *= 2
        a_in = a_out
        b_in = b_out
        aa_in = aa_out
        ab_in = ab_out
        bb_in = bb_out

    out_image.save('result17Jan31.tif')

def testFractalColors():
    x_max = 256#4096#729#1024
    y_max = 256#4096#int(round(0.5*np.sqrt(3)*x_max))#1024#729#1024

    out_image = createImageRGB(x_max,y_max,1,1)
    

    f_out0 = np.tile(np.asarray(list(range(0,x_max)))*(1./(0.5*x_max)),y_max)
    f_error = np.repeat(np.asarray(list(range(0,y_max)))*(1./(2*y_max)),x_max)
    print(("f_out0",f_out0))
    print(("f_out0.size",f_out0.size))
    print(("f_error",f_error))
    print(("f_error.size",f_error.size))

    out_image = createImageRGB(x_max,y_max,1,1)
    writeToImage(out_image,red(f_out0,f_error),green(f_out0,f_error),blue(f_out0,f_error),x_max,y_max,1,0,0)
    out_image.save('fractalColorTest.tif')

# Fractal dimension of image
def testFractal():
    x_max = 32#1024#729#1024
    y_max = 32#1024#int(round(0.5*np.sqrt(3)*x_max))#1024#729#1024
    power_target = 5
    a_in = np.empty(x_max*y_max)
    b_in = np.empty(x_max*y_max)
    c_in = np.empty(x_max*y_max)
    z = np.zeros(x_max*y_max)
    #readImageNoConv('SierpinskiImage.tif',c_in,0,0,x_max,y_max)
    #readImageNoConv('LineImage.tif',c_in,0,0,x_max,y_max)
    #readImageNoConv('Menger2Image.tif',c_in,0,0,x_max,y_max)
    readImageNoConv('r322_14b_red.tif',a_in,50,15,x_max,y_max)
    readImageNoConv('r322_14b_nir.tif',b_in,50,15,x_max,y_max)
    c_in = (b_in-a_in)/(b_in+a_in)
    c_in = thresholdArray(c_in,0.5)
    a_out = c_in
    b_out = c_in
    out_image = createImageRGB(x_max,y_max,power_target+1,2)
    w_out = 1
    x_regress = np.empty(power_target+1)
    y_regress = np.empty(power_target+1)
    
    for result_index in range(0,power_target+1):
        writeToImage(out_image,a_out,a_out,a_out,x_max,y_max,w_out,result_index,0)
        writeToImage(out_image,b_out,b_out,b_out,x_max,y_max,w_out,result_index,1)
        sum_b_out = windowCollectSum(b_out,x_max,y_max,w_out)
        #alt_sum_b = windowCreateSum(c_in,x_max,y_max,w_out)
        numerator = np.log(sum_b_out)/np.log(2)
        denominator = np.log(x_max)/np.log(2)-result_index
        epsilon = 2**(-denominator)
        print((sum_b_out, numerator, denominator, numerator/denominator, epsilon))
        #if result_index > 0:
        x_regress[result_index] = denominator #epsilon #result_index
        y_regress[result_index] = numerator #numerator/denominator
        a_in = a_out
        b_in = b_out
        w_out *= 2
        a_out = windowSum(a_in,x_max,y_max,w_out)/4
        b_out = windowMax(b_in,x_max,y_max,w_out)
        
    
    print((x_regress, y_regress))
    print((np.polyfit(x_regress,y_regress,1)))
    print(("Sierpinski", np.log(3)/np.log(2)))
    print(("Menger2d", np.log(8)/np.log(3)))
    
    out_image.save('resultFractal.tif')
    
    fig = plt.figure()
    font = {'family' : 'normal','weight' : 'normal','size'   : 16}
    plt.rc('font', **font)
    plt.plot(x_regress,y_regress,'ro')
    plt.xticks([0,1,2,3,4,5])
    plt.xlabel('Binary Logarithm of Inverse Window Size')
    plt.ylabel('Binary Logarithm of Sum')
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
    plt.show()
    fig.savefig('fig_fractalRegression.pdf')

# Main method for determining a window-based fractal dimension
def testFractalWindow():
    x_max = 1024#4096#729#1024
    y_max = 1024#4096#int(round(0.5*np.sqrt(3)*x_max))#1024#729#1024
    power_start = 0
    power_target = 5
    a_thresh = 0.5
    a_in = np.empty(x_max*y_max)
    ab_in = np.empty(x_max*y_max)
    ac_in = np.empty(x_max*y_max)
    
    #readImageNoConv('SierpinskiImage.tif',a_in,0,0,x_max,y_max)
    #print np.reshape(a_in,(x_max,-1))
     
    #readImageNoConv('r322_14b_red.tif',ab_in,2048,3074,x_max,y_max)
    #readImageNoConv('r322_14b_nir.tif',ac_in,2048,3074,x_max,y_max)
    readImageNoConv('r322_red_reraster.tif',ab_in,0,0,x_max,y_max)
    readImageNoConv('r322_nir_reraster.tif',ac_in,0,0,x_max,y_max)

    a_in = (ac_in-ab_in)/(ac_in+ab_in)
    print(a_in)
    ndvi_image = createImageRGB(x_max,y_max,1,1)
    a_rev = 1-a_in
    writeToImage(ndvi_image,a_rev,a_rev,a_rev,x_max,y_max,1,0,0)
    ndvi_image.save('ndvitest_new.tif')

    b_out = thresholdArrayMan(a_in,a_thresh)
    ndvi_thresh_image = createImageRGB(x_max,y_max,1,1)
    b_rev = 1-b_out
    writeToImage(ndvi_image,b_rev,b_rev,b_rev,x_max,y_max,1,0,0)
    ndvi_image.save('ndvi_thresh_test_new.tif')

    f_out, f_error = windowFractalArray(b_out,x_max,y_max,power_start,power_target)
    print(("f_out",f_out))
    print(("f_out.size",f_out.size))
    print(("f_error",f_error))
    out_image = createImageRGB(x_max,y_max,2,1)
    #writeToImage(out_image,1-high(f_out[0,],f_error),1-low(f_out[0,],f_error),blue(f_out[0,],f_error),x_max,y_max,2**power_target,0,0)
    a_rev = 1-a_in
    writeToImage(out_image,a_rev,a_rev,a_rev,x_max,y_max,1,0,0)
    # This produces the plot (check what high and low are...)
    # f_error = residual; f_out = slope
    writeToImage(out_image,1-high(f_out[0,],f_error),1-low(f_out[0,],f_error),blue(f_out[0,],f_error),x_max,y_max,2**power_target,1,0)

    out_image.save('ndviFractalOutNew.tif')
    print(("x_max", x_max))
    print(("y_max", y_max))

# Uses landuse classes
def testFractalWindowClass():
    x_max = 4096#1024#4096#729#1024
    y_max = 4096#1024#4096#int(round(0.5*np.sqrt(3)*x_max))#1024#729#1024
    power_start = 0
    power_target = 5
    #select_set = set([141,190,195]) #deciduous forest, woody wetlands, herbatious wetlands
    a_thresh = 0.5#0.3#0.5
    #select_value = 88 #nonag / undefined
    #select_value = 1 #corn
    a_in = np.empty(x_max*y_max)
    ab_in = np.empty(x_max*y_max)
    ac_in = np.empty(x_max*y_max)
    lc_in = np.empty(x_max*y_max)
    print(("x_max*y_max",x_max*y_max))
    print(("(x_max-2**power_target +1)*(y_max-2**power_target +1)",(x_max-2**power_target +1)*(y_max-2**power_target +1)))
    print(("(x_max-2**(power_target+1) +1)*(y_max-2**(power_target+1) +1)",(x_max-2**(power_target+1) +1)*(y_max-2**(power_target+1) +1)))
    
    readImageNoConv('r322_red_reraster.tif',ab_in,0,0,x_max,y_max)
    readImageNoConv('r322_nir_reraster.tif',ac_in,0,0,x_max,y_max)
    a_in = (ac_in-ab_in)/(ac_in+ab_in)
    print(a_in)
    ndvi_image = createImageRGB(x_max,y_max,1,1)
    writeToImage(ndvi_image,a_in,a_in,a_in,x_max,y_max,1,0,0)
    ndvi_image.save('ndvitest.tif')

    readImageNoConv('CDL_reraster_new.tif',lc_in,0,0,x_max,y_max)
    #selectStatistics(lc_in)
    crop_max = 17
    #crop_translate = np.ndarray([0,1],[1,5],[2,6],[3,23],[4,28],[5,36],[6,37],[7,41],[8,111],[9,121],[10,122],[11,123],[12,124],[13,141],[14,176],[15,190],[16,195])
    dummy_list = [1.,5.,6.,23.,28.,36.,37.,41.,111.,121.,122.,123.,124.,141.,176.,190.,195.]
    set0 = set([1.,5.,6.,23.,41.])
    set1 = set([28.,36.,37.,176.])
    set2 = set([121.,122.,123.,124.])
    set3 = set([111.,141.,190.,195.])
    set10 = set([28.,36.,37.,111.,121.,122.,123.,124.,141.,176.,190.,195.])
    string_list = ["Corn","Soybeans","Sunflower","Spring Wheat","Oats","Alfalfa","Other Hay","Sugarbeets","Open Water","Dev Open","Dev Low","Dev Med","Dev High","Deciduous","Pasture","Woody Wetlands","Herbacious Wetlands"]
    print(dummy_list)
    crop_translate = np.array(dummy_list)
    crops = np.ndarray(shape=(crop_max,x_max*y_max))
    for i in range(0,crop_max):
        crops[i,lc_in==crop_translate[i]] = 1
    stats = np.zeros(crop_max)
    for i in range(0,crop_max):
        stats[i] = np.count_nonzero(lc_in == crop_translate[i])
    print(stats)

    out_image = createImageRGB(x_max,y_max,2,2)
    #lc_select = selectArray(lc_in,select_set)
    #print lc_select
    lc_select0 = selectArray(lc_in,set0)
    writeToImage(out_image,1-lc_select0,1-lc_select0,1-lc_select0,x_max,y_max,1,0,0)
    lc_select1 = selectArray(lc_in,set1)
    writeToImage(out_image,1-lc_select1,1-lc_select1,1-lc_select1,x_max,y_max,1,1,0)
    lc_select2 = selectArray(lc_in,set2)
    writeToImage(out_image,1-lc_select2,1-lc_select2,1-lc_select2,x_max,y_max,1,0,1)
    lc_select3 = selectArray(lc_in,set3)
    writeToImage(out_image,1-lc_select3,1-lc_select3,1-lc_select3,x_max,y_max,1,1,1)
    out_image.save('selectOrig.tif')

    
    #out_image = createImageRGB(x_max,y_max,2,2)
    #counter = 0
    #for lc_select in [lc_select0,lc_select1,lc_select2,lc_select3]:
    #    lc_aggregate = lc_select
    #    for power in range(0,power_target):
    #        lc_aggregate = windowSum(lc_aggregate,x_max,y_max,2**(power+1))/4
            #print "2**power+1",2**power+1
    #    print int(counter/2), int(counter%2)
    #    writeToImage(out_image,1-lc_aggregate,1-lc_aggregate,1-lc_aggregate,x_max,y_max,2**power_target,int(counter%2),int(counter/2))
    #    counter += 1
    #out_image.save('selectAv.tif')

    crop_avs = np.ndarray(shape=(crop_max,(x_max-2**power_target + 1)*(y_max-2**power_target + 1)))

    for i in range(0,crop_max):
        crop_av_loc = crops[i,]
        for power in range(0,power_target):
            crop_av_loc = windowSum(crop_av_loc,x_max,y_max,2**(power+1))/4
        crop_avs[i,] = crop_av_loc

    #print "lc_aggregate.size",lc_aggregate.size

    b_out = thresholdArrayMan(a_in,a_thresh)
    f_out, f_error = windowFractalArray(b_out,x_max,y_max,power_start,power_target)
    print(("f_out",f_out))
    print(("f_error",f_error))
    print(("f_error.size/2",f_out.size/2))
    frac_array = fractality(f_out[0,],f_error)
    print(("frac_array.size", frac_array.size))
    print(frac_array)
    print(crop_avs)
    low_array = low(f_out[0,],f_error)
    high_array = high(f_out[0,],f_error)

    ndvi_aggregate = b_out
    for power in range(0,power_target):
        ndvi_aggregate = windowSum(ndvi_aggregate,x_max,y_max,2**(power+1))/4
        #print "2**power+1",2**power+1
    print(("ndvi_aggregate.size",ndvi_aggregate.size))

    weighted_sum = np.zeros(crop_max)
    low_sum = np.zeros(crop_max)
    high_sum = np.zeros(crop_max)
    ndvi_sum = np.zeros(crop_max)
    for i in range(0,crop_max):
        #printCorrelation (crop_avs[i,],frac_array)
        weighted_sum[i] = np.sum(crop_avs[i,]*frac_array,0)
        low_sum[i] = np.sum(crop_avs[i,]*low_array,0)
        high_sum[i] = np.sum(crop_avs[i,]*high_array,0)
        ndvi_sum[i] = np.sum(crop_avs[i,]*ndvi_aggregate,0)
        print((i, crop_translate[i],string_list[i],stats[i],weighted_sum[i],low_sum[i],high_sum[i],ndvi_sum[i]))
        print((weighted_sum[i]/stats[i],low_sum[i]/stats[i],high_sum[i]/stats[i],ndvi_sum[i]/stats[i]))
        print()
    with open('fractal_results.pickle','w') as f:
        pickle.dump([crop_translate,stats,weighted_sum,low_sum,high_sum,ndvi_sum],f)

    out_image = createImageRGB(x_max,y_max,1,3)
    #printCorrelation (lc_aggregate,frac_array)
    #writeToImage(out_image,lc_aggregate,lc_aggregate,lc_aggregate,x_max,y_max,2**power_target,0,1)
    writeToImage(out_image,1-b_out,1-b_out,1-b_out,x_max,y_max,1,0,0)
    writeToImage(out_image,1-high_array,1-low_array,blue(f_out[0,],f_error),x_max,y_max,2**power_target,0,1)
    lc_select10 = selectArray(lc_in,set10)
    for power in range(0,power_target):
        lc_select10 = windowSum(lc_select10,x_max,y_max,2**(power+1))/4
    writeToImage(out_image,1-lc_select10,1-lc_select10,1-lc_select10,x_max,y_max,2**power_target,0,2)

    print(("x_max", x_max))
    print(("y_max", y_max))
    out_image.save('classFractalOut.tif')

def fractalSpeedTest():
    x_max = 1024#729#1024
    y_max = 1024#int(round(0.5*np.sqrt(3)*x_max))#1024#729#1024
    power_start = 0
    threshold = 0.5
    
    a_in = np.empty(x_max*y_max)
    readImageNoConv('sierpinskiImage.tif',a_in,0,0,x_max,y_max)
    target_min = 3
    target_max = 8
    time_array = np.empty(target_max-target_min)
    start_time = time.time()
    for power_target in range(target_min,target_max):
        f_out, f_error = windowFractalArray(a_in,x_max,y_max,power_start,power_target)
        print(("f_out",f_out))
        print(("f_error",f_error))
        time_array[power_target-target_min] = time.time() - start_time
        print(("power_target",power_target,"time", time_array[power_target-target_min]))
        start_time = time.time()
    
    print(time_array)
    out_image = createImageRGB(x_max,y_max,1,1)
    writeToImage(out_image,1-high(f_out[0,],f_error),1-low(f_out[0,],f_error),blue(f_out[0,],f_error),x_max,y_max,2**power_target,0,0)
    out_image.save('sierpinskiFractalOut.tif')


def fractalSpeedTestBrute():
    x_max = 1024#729#1024
    y_max = 1024#int(round(0.5*np.sqrt(3)*x_max))#1024#729#1024
    power_start = 0
    threshold = 0.5
    
    a_in = np.empty(x_max*y_max)
    readImageNoConv('sierpinskiImage.tif',a_in,0,0,x_max,y_max)
    start_time = time.time()
    target_min = 3
    target_max = 8
    time_array = np.empty(target_max-target_min)
    for power_target in range(target_min,target_max):
        f_out, f_error = windowFractalArrayBrute(a_in,x_max,y_max,power_start,power_target)
        print(("f_out",f_out))
        print(("f_error",f_error))
        red = np.exp(-(f_out[0,]-0.5)**2/0.125) * np.exp(-2*f_error)
        green = np.exp(-(f_out[0,]-1.5)**2/0.125) * np.exp(-2*f_error)
        time_array[power_target-target_min] = time.time() - start_time
        print(("power_target",power_target,"time", time_array[power_target-target_min]))
        start_time = time.time()
    
    print(time_array)
    out_image = createImageRGB(x_max,y_max,1,1)
    writeToImage(out_image,red,green,blue(f_out[0,],f_error),x_max,y_max,2**power_target,0,0)
    out_image.save('sierpinskiFractalOutBrute.tif')


def testFractalWindowBrute():
    x_max = 128#1024#729#1024
    y_max = 128#1024#int(round(0.5*np.sqrt(3)*x_max))#1024#729#1024
    power_start = 0
    power_target = 5
    threshold = 0.5
    a_in = np.empty(x_max*y_max)
    #readImageNoConv('SierpinskiImage.tif',a_in,0,0,x_max,y_max)
    #print np.reshape(a_in,(x_max,-1))
    #readImageNoConv('CheckerboardImage.tif',a_in,0,0,x_max,y_max)
    #readImageNoConv('XImage.tif',a_in,0,0,x_max,y_max)
    #readImageNoConv('plusImage.tif',a_in,0,0,x_max,y_max)
    #readImageNoConv('SquareImage.tif',a_in,0,0,x_max,y_max)
    #readImageNoConv('WhiteImage.tif',a_in,0,0,x_max,y_max)
    #readImageNoConv('Menger2Image.tif',a_in,0,0,x_max,y_max)
    #readImageNoConv('r322_14b_red.tif',a_in,256,0,x_max,y_max)
    #readImageNoConv('r322_14b_nir.tif',b_in,256,0,x_max,y_max)
    #c_in = (b_in-a_in)/(b_in+a_in)
    
    readImageNoConv('sierpinskiImage.tif',a_in,0,0,x_max,y_max)
    f_out, f_error = windowFractalArrayBrute(a_in,x_max,y_max,power_start,power_target)
    print(("f_out",f_out))
    print(("f_error",f_error))
    red = np.exp(-(f_out[0,]-0.5)**2/0.125) * np.exp(-2*f_error)
    green = np.exp(-(f_out[0,]-1.5)**2/0.125) * np.exp(-2*f_error)
    out_image = createImageRGB(x_max,y_max,1,1)
    writeToImage(out_image,red,green,blue(f_out[0,],f_error),x_max,y_max,2**power_target,0,0)
    out_image.save('sierpinskiFractalOutBrute.tif')

def populateArraySakakawea():
    x_max = 3000
    y_max = 800
    power_target = 8
    b_in = np.empty((x_max+1)*(y_max+1))
    c_in = np.empty((x_max+1)*(y_max+1))
    readImageNoConv('Sakaka_RED2.tiff',b_in,2200,3300,x_max+1,y_max+1)
    readImageNoConv('Sakaka_NIR2.tiff',c_in,2200,3300,x_max+1,y_max+1)
    np.nan_to_num(b_in)
    np.nan_to_num(c_in)
    a_in = (c_in-b_in)/(c_in+b_in)
    print(a_in)
    print(b_in)
    print(c_in)
    thresh = 0
    b_out = thresholdArrayMan(a_in,thresh)
    return windowXOR(b_out,x_max+1,y_max+1,2)

def populateNDVISakakawea():
    x_max = 3000
    y_max = 800
    power_target = 8
    b_in = np.empty((x_max+1)*(y_max))
    c_in = np.empty((x_max+1)*(y_max))
    readImageNoConv('Sakaka_RED2.tiff',b_in,2200,3300,x_max,y_max)
    readImageNoConv('Sakaka_NIR2.tiff',c_in,2200,3300,x_max,y_max)
    np.nan_to_num(b_in)
    np.nan_to_num(c_in)
    a_in = (c_in-b_in)/(c_in+b_in)
    print(a_in)
    print(b_in)
    print(c_in)
    thresh = 0
    b_out = thresholdArrayMan(a_in,thresh)
    return b_out


def testSakakawea():
    x_max = 3000
    y_max = 800
    power_target = 5
    b_in = np.empty((x_max+1)*(y_max+1))
    c_in = np.empty((x_max+1)*(y_max+1))
    readImageNoConv('Sakaka_RED2.tiff',b_in,2200,3300,x_max+1,y_max+1)
    readImageNoConv('Sakaka_NIR2.tiff',c_in,2200,3300,x_max+1,y_max+1)
    np.nan_to_num(b_in)
    np.nan_to_num(c_in)
    a_in = (c_in-b_in)/(c_in+b_in)
    print(a_in)
    print(b_in)
    print(c_in)
    thresh = 0
    b_out = thresholdArrayMan(a_in,thresh)
    b_out = windowXOR(b_out,x_max+1,y_max+1,2)
    coast_image = createImageRGB(x_max,y_max,1,1)
    writeToImage(coast_image,1-b_out,1-b_out,1-b_out,x_max,y_max,1,0,0)
    coast_image.save('Sakaka_IN.tif')
   
    w_out = 1

    out_image = createImageRGB(x_max,y_max,1,power_target+1)
    #writeToImage(out_image,a_in,a_in,a_in,x_max,y_max,2**power_target,0,0)
    #writeToImage(out_image,shore_line,shore_line,shore_line,x_max,y_max,2**power_target,0,0)

    x_regress = np.empty(power_target)
    y_regress = np.empty(power_target)
    
    for result_index in range(0,power_target+1):
        #writeToImage(out_image,a_out,a_out,a_out,x_max,y_max,w_out,result_index,0)
        #writeToImage(out_image,1-b_out,1-b_out,1-b_out,x_max,y_max,w_out,0,result_index)
        print(("w_out", w_out))
        print(("b_out.size", b_out.size))
        print(("(x_max-w_out+1)*(y_max-w_out+1)",(x_max-w_out+1)*(y_max-w_out+1)))
        sum_b_out = windowCollectSum(b_out,x_max,y_max,w_out)
        print_b_out = windowCollectSquares(b_out,x_max,y_max,w_out)
        print(print_b_out)
        writeToImage(out_image,1-print_b_out,1-print_b_out,1-print_b_out,x_max,y_max,1,0,result_index)
        #alt_sum_b = windowCreateSum(c_in,x_max,y_max,w_out)
        numerator = np.log(sum_b_out)/np.log(2)
        denominator = np.log(x_max)/np.log(2)-result_index
        epsilon = 2**(-denominator)
        print((sum_b_out, numerator, denominator, numerator/denominator, epsilon))
        #if result_index > 0:
        x_regress[result_index-1] = denominator #epsilon #result_index
        y_regress[result_index-1] = numerator #numerator/denominator
        #a_in = a_out
        b_in = b_out
        w_out *= 2
        #a_out = windowSum(a_in,x_max,y_max,w_out)/4
        b_out = windowMax(b_in,x_max,y_max,w_out)
    
    '''print(x_regress, y_regress)
    slope, y0 = np.polyfit(x_regress,y_regress,1)
    print slope, y0
    fig = plt.figure()
    font = {'family' : 'normal','weight' : 'normal','size'   : 16}
    plt.rc('font', **font)
    plt.plot(x_regress,y_regress,'ro')
    plt.plot([x_regress[0],x_regress[x_regress.size-1]],[y0+slope*x_regress[0],y0+slope*x_regress[x_regress.size-1]])
    #plt.xticks([0,1,2,3,4,5,6])
    plt.xlabel('Binary Logarithm of Inverse Window Size')
    plt.ylabel('Binary Logarithm of Sum')
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)
    plt.show()
    fig.savefig('fig_fractal_regression_saka.png')'''

    out_image.save('Sakaka_OUT.tif')

# not tested, not yet usee
def testFractal3d():
    x_max = 1024#1000#1024#4096#729#1024
    y_max = 1024#1000#1024#4096#int(round(0.5*np.sqrt(3)*x_max))#1024#729#1024
    power_target = 6
    a_in = np.empty(x_max*y_max)
    #ab_in = np.empty(x_max*y_max)
    #ac_in = np.empty(x_max*y_max)
    
    #readImageNoConv('xTestImage.tif',a_in,0,0,x_max,y_max)
    readImageNoConv('SierpinskiImage.tif',a_in,0,0,x_max,y_max)
    #print np.reshape(a_in,(x_max,-1))
    
    
    #readImageNoConv('r322_14b_red.tif',ab_in,2048,3074,x_max,y_max)
    #readImageNoConv('r322_14b_nir.tif',ac_in,2048,3074,x_max,y_max)
    #readImageNoConv('r322_red_reraster.tif',ab_in,0,0,x_max,y_max)
    #readImageNoConv('r322_nir_reraster.tif',ac_in,0,0,x_max,y_max)
    
    #a_in = (ac_in-ab_in)/(ac_in+ab_in)
    
    print(a_in)
    
    f_out, f_error = windowFractal3d(a_in,x_max,y_max,power_target)

    '''ndvi_image = createImageRGB(x_max,y_max,1,1)
    a_rev = 1-a_in
    writeToImage(ndvi_image,a_rev,a_rev,a_rev,x_max,y_max,1,0,0)
    ndvi_image.save('ndvitest.tif')
    
    b_out = thresholdArrayMan(a_in,a_thresh)
    ndvi_thresh_image = createImageRGB(x_max,y_max,1,1)
    b_rev = 1-b_out
    writeToImage(ndvi_image,b_rev,b_rev,b_rev,x_max,y_max,1,0,0)
    ndvi_image.save('ndvi_thresh_test.tif')
    
    f_out, f_error = windowFractalArray(b_out,x_max,y_max,power_start,power_target)
    print "f_out",f_out
    print "f_out.size",f_out.size
    print "f_error",f_error
    out_image = createImageRGB(x_max,y_max,2,1)
    #writeToImage(out_image,1-high(f_out[0,],f_error),1-low(f_out[0,],f_error),blue(f_out[0,],f_error),x_max,y_max,2**power_target,0,0)
    a_rev = 1-a_in
    writeToImage(out_image,a_rev,a_rev,a_rev,x_max,y_max,1,0,0)
    writeToImage(out_image,1-high(f_out[0,],f_error),1-low(f_out[0,],f_error),blue(f_out[0,],f_error),x_max,y_max,2**power_target,1,0)
    
    out_image.save('ndviFractalOut.tif')
    print "x_max", x_max
    print "y_max", y_max'''

# BEGIN NICK METHODS

def testFractalWindow2(x_max, y_max, power_start, power_target, a_thresh):
    #x_max = 128#1024#4096#729#1024
    #y_max = 128#1024#4096#int(round(0.5*np.sqrt(3)*x_max))#1024#729#1024
    #power_start = 0
    #power_target = 5
    #a_thresh = 0.5
    a_in = np.empty(x_max*y_max)
    ab_in = np.empty(x_max*y_max)
    ac_in = np.empty(x_max*y_max)
    
    #readImageNoConv('r322_14b_red.tif',ab_in,2048,3074,x_max,y_max)
    #readImageNoConv('r322_14b_nir.tif',ac_in,2048,3074,x_max,y_max)
    readImageNoConv('r322_red_reraster.tif',ab_in,0,0,x_max,y_max)
    readImageNoConv('r322_nir_reraster.tif',ac_in,0,0,x_max,y_max)

    a_in = (ac_in-ab_in)/(ac_in+ab_in)
    print(a_in)
    ndvi_image = createImageRGB(x_max,y_max,1,1)
    a_rev = 1-a_in
    writeToImage(ndvi_image,a_rev,a_rev,a_rev,x_max,y_max,1,0,0)
    ndvi_image.save('ndvitest_new.tif')

    b_out = thresholdArrayMan(a_in,a_thresh)
    ndvi_thresh_image = createImageRGB(x_max,y_max,1,1)
    b_rev = 1-b_out
    writeToImage(ndvi_image,b_rev,b_rev,b_rev,x_max,y_max,1,0,0)
    ndvi_image.save('ndvi_thresh_test_new.tif')

    f_out, f_error = windowFractalArray(b_out,x_max,y_max,power_start,power_target)
    return f_out[0,], f_error, a_rev, a_in

zerofractal = np.vectorize(lambda x: float(x if x <= 1.0 else 0.0))
onefractal = np.vectorize(lambda x: x if (x > 1.0 and x <= 2.0) else 0)

def printFractalImage(f_out, f_error, x_max, y_max, power_target, a_rev, a_in, image_name):
    
    # for debugging, print entire array
    #np.set_printoptions(threshold=np.nan)

    print(("f_out",np.sort(f_out)))
    print(("f_out.size",f_out.size))
    print(("f_error",f_error))
    out_image = createImageRGB(x_max,y_max,2,1)
    #writeToImage(out_image,1-high(f_out[0,],f_error),1-low(f_out[0,],f_error),blue(f_out[0,],f_error),x_max,y_max,2**power_target,0,0)
    a_rev = 1-a_in
    writeToImage(out_image,a_rev,a_rev,a_rev,x_max,y_max,1,0,0)
    # This produces the plot (check what high and low are...)
    # f_error = residual; f_out = slope
    #writeToImage(out_image,1-sin_filter(f_out,f_error),1-sin_filter(f_out,f_error),1-sin_filter(f_out,f_error),x_max,y_max,2**power_target,1,0)
    writeToImage(out_image,1-high(f_out,f_error),1-high(f_out,f_error),1-high(f_out,f_error),x_max,y_max,2**power_target,1,0)

    out_image.save(image_name)
    print(("x_max", x_max))
    print(("y_max", y_max))

def printFractalImage2(f_out_1, f_out_2, f_error, x_max, y_max, power_target, image_name):
    
    # for debugging, print entire array
    #np.set_printoptions(threshold=np.nan)

    #print("f_out",np.sort(f_out))
    #print("f_out.size",f_out.size)
    #print("f_error",f_error)
    out_image = createImageRGB(x_max,y_max,2,1)

    writeToImage(out_image,1-sin_filter(f_out_1,f_error),1-sin_filter(f_out_1,f_error),1-sin_filter(f_out_1,f_error),x_max,y_max,2**power_target,0,0)
    writeToImage(out_image,1-sin_filter(f_out_2,f_error),1-sin_filter(f_out_2,f_error),1-sin_filter(f_out_2,f_error),x_max,y_max,2**power_target,1,0)
    #writeToImage(out_image,1-high(f_out,f_error),1-high(f_out,f_error),1-high(f_out,f_error),x_max,y_max,2**power_target,1,0)

    out_image.save(image_name)
    print(("x_max", x_max))
    print(("y_max", y_max))

def main():
    x_max = 1024#128#4096#729#1024
    y_max = 1024#128#4096#int(round(0.5*np.sqrt(3)*x_max))#1024#729#1024
    power_start = 0
    power_target = 5
    a_thresh = 0.5

    f_out, f_error, a_rev, a_in = testFractalWindow2(x_max, y_max, power_start, power_target, a_thresh)

    #lessthanone = f_out <= 1
    print("slopes less than 1:")
    #vals = f_out[np.where(f_out <= 2)]
    #vals = np.extract(f_out <= 1,f_out)
    #print(vals)

    #print("slopes less than one: ",lessthanone)

    zerodim = zerofractal(f_out)
    onedim = onefractal(f_out)
    onedimsin = onefractal(f_out)-1

    # for debugging, print entire array
    #np.set_printoptions(threshold=np.nan)

    #print("Sorted f_out: ", np.sort(f_out))
    #print("Fractals Zero to One: ", zerodim)
    #print("Fractals One to Two: ", onedim)

    #printFractalImage(zerodim, f_error, x_max, y_max, power_target, a_rev, a_in, 'fractalZeroToOne.tif')
    #printFractalImage(onedim, f_error, x_max, y_max, power_target, a_rev, a_in, 'fractalOneToTwoOrig.tif')
    printFractalImage2(zerodim, onedimsin, f_error, x_max, y_max, power_target, 'result.tif')
    #printFractalImage(onedimsin, f_error, x_max, y_max, power_target, a_rev, a_in, 'fractalOneToTwo.tif')

    #return f_out

main()
#testSakakawea()
#createTestPlot()
#testWindowRegressionArray()
#testFractal()
#testFractalWindow()
#testFractalWindowBrute()
#fractalSpeedTestBrute()
#testFractalColors()


#testFractalWindowClass()

#testFractal3d()

#Test addition
