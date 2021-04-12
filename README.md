# SlidingWindow  

## Spatial analyses for geographic images predominantly using a log(n) aggregation algorithm  

### Methods  
##### Sectral  
* NDVI  
* Regression  
* Pearson  
* Fractal  
* Fractal 3D  

##### Topographical  
* Slope  
* Aspect  
* Standard Curvature  

### Files  
* test.py:  
unit test file  
* image_generator.py:  
generate point values and entire images with analytical formulae  
* windowagg/agg_ops.py:  
Enum of available aggregation operations  
* windowagg/aggregation.py:  
agggregation methods, vectorized and brute force  
* windowagg/analyses.py:  
Enum of available analyses  
* windowagg/cluster.py:  
functions to combine analyses in interesting ways  
* windowagg/dem.py:  
spatial analyses for topographical calculations  
* windowagg/dem_data.py:  
class to hold, export, and import data required for topographical calculations  
* windowagg/helper.py:  
auxiliary functions  
* windowagg/rbg.py:  
spatial analyses formulae for spectral images  
* windowagg/sliding_window.py:  
high-level class to handle entire analysis operations  

### Rasterio
Rasterio, a dependency of this package, has special dependencies of its own that must be satisfied before using this code.  
See the rasterio [installation guide](https://rasterio.readthedocs.io/en/latest/installation.html) for more information.
