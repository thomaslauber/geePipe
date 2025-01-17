# geePipe
Google Earth Engine Pipeline for Spatio-Temporal models

Very basic python package so far. Before installing it, you should create a testing conda environment: 
```bash
conda create --name ee_test --clone ee
```
Then, you can install it by running:
```bash
pip install -e . --user
```

In order to use the functions so far, load the package after loading the `earthengine-api`:
```python
import ee 
import geePipe
```

You can then use the `reduceRegionsGDF` function to sample an image using either an ee.FeatureCollection or a GeoDataFrame. The output will be given as a GeoDataFrame. Tests have shown so far to take roughly 20-40 sec for 150,000 points depending on the number of cores on the computer. 
