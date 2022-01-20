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

### Python Dependencies

* **Rasterio:** 
Reads and writes GeoTIFF and other GIS formats to organize and store gridded raster datasets and provid a Python API based on Numpy N-dimensional arrays.
Rasterio, has special dependencies of its own that must be satisfied before using this code.  
See the rasterio [installation guide](https://rasterio.readthedocs.io/en/latest/installation.html) for more information.

* **NumPy:**
Provides multidimensional array objects, and an assortment of routines for fast operations on arrays.

* **Affine:**
Matrices describing affine transformations of georeferenced raster datasets to map from image coordinates to world coordinates.

* **Matplotlib:**
A comprehensive library for creating static, animated, and interactive visualizations.
