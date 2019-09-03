# SlidingWindows

## Installing dependencies for WindowFractalCode.py
## For Git Bash on Windows

1. Create a virtual environment and activate it.
cd /path/to/repo
python -m venv env
source env/Scripts/activate

2. Upgrade pip.
pip install --upgrade pip

3. Download and install rasterio/GDAL Binaries
[rasterio](https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio)
[GDAL](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal)
pip install /path/to/GDAL_binaries
pip install /path/to/rasterio_binaries

3. Install the rest of the dependencies
pip install -r requirements.txt
