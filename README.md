# SlidingWindow  

## Spatial analyses for geographic images using Iterative Aggregation of Regression Terms algorithm 

### Files

* demo.py:  
Five example analyses, including generating window-based slope and curvatures for artificial data 
with or without noise, curvatures of an example landscape, a speed test, and a visualization of 
differences between computed and analytical curvatures. 
Defaults to artificial data with noise, but other choices can be selected at the beginning of the file.

* image_generator.py:  
generate artificial images, in some cases including their analytical devatives

* windowagg/agg_ops.py:  
Enum of available aggregation operations  

* windowagg/aggregation.py:  
agggregation methods, vectorized and brute force  

* windowagg/analyses.py:  
Enum of available analyses  

* windowagg/dem.py:  
spatial analyses for topographical calculations  

* windowagg/dem_data.py:  
class to hold, export, and import data required for topographical calculations  

* windowagg/helper.py:  
auxiliary functions  

* windowagg/sliding_window.py:  
high-level class to handle entire analysis operations  

### Rasterio
Rasterio, a dependency of this package, has special dependencies of its own that must be satisfied before using this code.  
See the rasterio [installation guide](https://rasterio.readthedocs.io/en/latest/installation.html) for more information.
