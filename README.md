# SlidingWindow  

## Spatial analyses for geographic images predominantly using a log(n) aggregation algorithm  

### Methods  
##### Spectral  
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

### Example Usage
```Python

from windowagg.sliding_window import SlidingWindow
import windowagg.cluster as Cluster
from windowagg.analyses import Analyses
from image_generator import ImageGenerator

import numpy as np

# Create image generator object
# arg: Desired output path, data type
img_gen = ImageGenerator('generated_image_location/', np.uint8)
# Generate image
img_gen.gauss()

# Create sliding window object
# args: Path of image to be analyzed
sliding_window = SlidingWindow('test.tif')
# Create RBG analysis image
# args: image band 1, image band 2, number of aggregations
sliding_window.regression(1, 2, 5)
# Initial data for DEM analysis
sliding_window.initialize_dem()
# Aggregate DEM data
for i in range(5):
	sliding_window.aggregate_dem()
# Create DEM analysis image
sliding_window.dem_slope()
# Export DEM data so we don't have to compute it again
# args: Desired output path
sliding_window.export_dem('export_location.npz')
# Import DEM data we exported before
# args: Path to import from
sliding_window.import_dem('export_location.npz')

# Generate Clustered image
# args:
# 	Path of image to analyze
# 	analyses to perform
# 	number of aggregations on corresponding analyses
# 	bands for corresponding analyses
Cluster.gen_clustered_img(
	"test.tif",
	[Analyses.pearson, Analyses.pearson],
	[5, 5],
	[[1, 2], [2, 3]]
)

```
