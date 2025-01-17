""" Extensions to Google Earth Engine Classes. """

# Author: Thomas Lauber

import concurrent.futures
from typing import Optional
import warnings

import ee
import geemap
from geopandas import GeoDataFrame
import numpy as np
import pandas as pd


def add_custom_functions_to_eeImage():

    # A custom function to reduce an image to a GeoDataFrame
    def reduceRegionsGDF(
            self,
            collection: ee.FeatureCollection | GeoDataFrame,
            reducer: ee.Reducer,
            scale: Optional[float] = None,
            crs: Optional[str] = None,
            crsTransform: Optional[list] = None,
            tileScale: Optional[int] = 1,
            n_jobs: Optional[int] = None
    ) -> GeoDataFrame:
        """ Apply a reducer over the area of each feature in the GeoDataFrame.

        The reducer must have the same number of inputs as the input image has bands.

        Args:
        image: The image to reduce.
        collection: The features to reduce over. If the crs is not the same as the one of the image, the points will get reprojected.
        reducer: The reducer to apply.
        scale: A nominal scale in meters of the projection to work in.
        crs: The projection to work in. If unspecified, the projection of the image's first band will be used. If specified in addition to scale, rescaled to the specified scale.
        crsTransform : The list of CRS transform values. This is a row-major ordering of the 3x2 transform matrix. This option is mutually exclusive with 'scale', and replaces any transform already set on the projection.
        tileScale: A scaling factor used to reduce aggregation tile size; using a larger tileScale (e.g., 2 or 4) may enable computations that run out of memory with the default.
        n_jobs: The number of parallel workers to use. If unspecified, use all available CPUs + 4, with at most 32.
        """

        # If crs is unspecified, use the projection of the first band of the image
        if crs is None:
            proj = self.select(0).projection().getInfo()
            crs = proj['crs']
            crsTransform = proj['transform']
        # If crs is a ee.Projection object, get the crs string
        if isinstance(crs, ee.Projection):
            proj = crs.getInfo()
            crs = proj['crs']
            crsTransform = proj['transform']
        # Raise an error if both scale and crsTransform are specified
        if scale is not None and crsTransform is not None:
            raise ee.ee_exception.EEException('Image.reduceRegionsGDF: Cannot specify both crsTransform and scale.')
        # Get a list of the output names of the reducer
        output_names = reducer.getOutputs().getInfo()
        # Get a list of the input columns
        if isinstance(collection, ee.FeatureCollection):
            input_columns = collection.first().propertyNames().getInfo()
        elif isinstance(collection, GeoDataFrame):
            input_columns = collection.columns
        # Get the band names of the image
        band_names = self.bandNames().getInfo()
        # Get the size of the collection
        if isinstance(collection, ee.FeatureCollection):
            size = collection.size().getInfo()
        elif isinstance(collection, GeoDataFrame):
            size = len(collection)
        # Define the number of break points for the sub-setting
        nr_of_subsets = (size + 1000-1)//1000

        # If the input is a GeoDataFrame, convert it to a FeatureCollection and create subsets
        if isinstance(collection, GeoDataFrame):
            # Reproject the GeoDataFrame if the crs is not the same
            if crs != collection.crs.to_string():
                collection = collection.to_crs(crs)
            # Subset the GeoDataFrame
            collection = np.array_split(collection, nr_of_subsets)

        def _get_results(self, collection, index, reducer, scale, crs, crsTransform, tileScale):
            # If collection is a GeoDataFrame, cast the subset into an ee.FeatureCollection
            if isinstance(collection, list) and all(isinstance(item, GeoDataFrame) for item in collection):
                fc = geemap.geopandas_to_ee(collection[index])
            # If collection is an ee.FeatureCollection, get the subset
            elif isinstance(collection, ee.FeatureCollection):
                fc = collection.toList(count=1000, offset=1000*index)
            # Sample the image
            sampled = self.reduceRegions(
                collection=fc,
                reducer=reducer,
                scale=scale,
                crs=crs,
                crsTransform=crsTransform,
                tileScale=tileScale
            )
            # Convert the sampled data to a pandas dataframe
            sampled_gdf = ee.data.computeFeatures({
                'expression': sampled,
                'fileFormat': 'GEOPANDAS_GEODATAFRAME'
            })
            # In case there is no data sampled for the subset, fill the columns with NaN
            if len(band_names) == 1 and sampled_gdf.columns == input_columns:
                sampled_gdf[output_names] = np.nan
            return sampled_gdf
        # Get the results with multiple workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(_get_results, self, collection, index, reducer, scale, crs, crsTransform, tileScale)
                       for index in range(nr_of_subsets)]
        # Get the results in one GeoDataFrame
        results = GeoDataFrame()
        for future in futures:
            results = pd.concat([results, future.result()])#.drop(columns='geo')
        # results = GeoDataFrame(results, geometry=results['geometry'])
        return results
    ee.Image.reduceRegionsGDF = reduceRegionsGDF











