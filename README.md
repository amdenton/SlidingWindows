# WindowAgg

## Window based aggregation methods for analyzing geographic images

### METHODS
#### RGB IR IMAGES
    ndvi
    binary
    aggregation
    regression
    Pearson
    fractal
    fractal 3D
#### DEM IMAGES
    window mean
    slope
    aspect
    standard curve
    profile curve
    planform curve


### INSTALL
pip install windowagg

### IMPORT
from windowagg.sliding_window import SlidingWindow

But first, you must download and install rasterio/GDAL Binaries
[rasterio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio)
[GDAL](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)
pip install /path/to/GDAL_binaries
pip install /path/to/rasterio_binaries
