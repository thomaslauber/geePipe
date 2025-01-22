# geePipe
Google Earth Engine Pipeline for Spatio-Temporal models

Very basic python package so far. Before installing it, you should create a testing conda environment: 
```bash
conda create --name ee_test --clone ee
conda activate ee_test
```
Then, you can install it by downloading the repo and running inside of it:
```bash
pip install -e . --user
```

In order to use the functions so far, load the package after loading the `earthengine-api`:
```python
import ee 
import geePipe
```

You can then use the `reduceRegionsGDF` function to sample an image using either an ee.FeatureCollection or a GeoDataFrame. The output will be given as a GeoDataFrame. Tests have shown so far to take roughly 20-40 sec for 150,000 points depending on the number of cores on the computer. 

Example script for testing: 
```python
import ee 
import geePipe

import time 

ee.Initialize()

composite = ee.Image('projects/crowtherlab/Composite/CrowtherLab_Composite_30ArcSec')
fc = ee.FeatureCollection('projects/crowtherlab/t3/WOSIS_2prcOutliersRemoved').limit(150000)
df = pd.read_csv('tests/data/uniqueCoords.csv')
gdf = GeoDataFrame(df, geometry=points_from_xy(df['longitude'], df['latitude']), crs='EPSG:4326')

# Sample composite using FeatureCollection
size = fc.size().getInfo()
print(f'Length of ee.FeatureCollection: {size}')
t1 = time.time()
sampled = composite.reduceRegions(collection=fc, reducer=ee.Reducer.first())
t = time.time() - t1
print(f'{t:.2f} seconds to sample {size} features')

# Sample composite using GDF
size = len(gdf)
print(f'Length of GDF: {size}')
t1 = time.time()
sampled = composite.reduceRegions(collection=gdf, reducer=ee.Reducer.first())
t = time.time() - t1
print(f'{t:.2f} seconds to sample {size} features')

# Create random, regular and fibonacci samples over the mask of the first band of an image
samplingTypes = ['random', 'regular', 'fibonacci']
for sT in samplingTypes:
  t1 = time.time()
  sample = generateSample(composite, 1000, sT)
  t = time.time() - t1
  sample.plot()
  plt.show()
  print(f'{t:.2f} seconds to generate {sT} sample of 1,000 points')
  

```

