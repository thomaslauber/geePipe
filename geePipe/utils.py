""" Utils methods. """

# Author: Thomas Lauber

import concurrent.futures
from fractions import Fraction
from typing import Optional
import warnings

import ee
from geopandas import GeoDataFrame
import pandas as pd
import numpy as np
from tqdm import tqdm


def _raiseEEException(
        function,
        argument,
        expected,
        actual
) -> ee.ee_exception.EEException:
    """ Raise an EEException inside a function, giving the argument and its actual and expected values.
    """
    raise ee.ee_exception.EEException(
        f"{function}, argument '{argument}': Invalid type.\n"
        f"Expected type: {expected}.\n"
        f"Actual type: {actual}.")

def stratificationDict(
        image: ee.Image,
        scale: Optional[float] = None,
        crs: Optional[str] = None,
        crsTransform: Optional[list] = None,
        bestEffort: Optional[bool] = False,
        maxPixels: Optional[int] = 1e9,
        tileScale: Optional[int] = 1
) -> ee.Dictionary:
    """ Generate an area-weighted stratification dictionary based on the distinct values discovered within the first band of the image. Returns a dictionary with the distinct values as keys and the area fraction as values. Masked pixels will be dropped.

    Args:
        image: The image to reduce. Only unmasked pixels of first band will be used.
        scale: A nominal scale in meters of the projection to work in.
        crs: The projection to work in. If unspecified, the projection of the image's first band will be used. If specified in addition to scale, rescaled to the specified scale.
        crsTransform : The list of CRS transform values. This is a row-major ordering of the 3x2 transform matrix. This option is mutually exclusive with 'scale', and replaces any transform already set on the projection.
        bestEffort: If the image would contain too many pixels at the given scale, compute and use a larger scale which would allow the operation to succeed.
        maxPixels: The maximum number of pixels to reduce.
        tileScale: A scaling factor used to reduce aggregation tile size; using a larger tileScale (e.g., 2 or 4) may enable computations that run out of memory with the default.
    """
    # Get the first band of the image and add the pixel area as band
    image = image.select(0)
    image = image.addBands(ee.Image.pixelArea()).updateMask(image.mask())

    # Compute the frequency histogram
    hist = ee.Dictionary(image.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram().splitWeights(),
        geometry=ee.Geometry.BBox(-180, -90, 180,90),
        scale=scale,
        crs=crs,
        crsTransform=crsTransform,
        bestEffort=bestEffort,
        maxPixels=maxPixels,
        tileScale=tileScale).get('histogram'))

    keys, values = hist.keys(), hist.values()
    sum = values.reduce(ee.Reducer.sum())
    values = values.map(lambda v: ee.Number(v).divide(sum))
    stratDict = ee.Dictionary.fromLists(keys,values)
    return stratDict

def resample(
        collection: ee.FeatureCollection | GeoDataFrame,
        n: Optional[int] = None,
        replace: Optional[bool] = True,
        classBand: Optional[str] = None,
        classDict: Optional[dict | ee.Dictionary] = None,
        seed: Optional[int] = 0
        ) -> ee.FeatureCollection | GeoDataFrame:
    """ Resample a collection with optional stratification.

    Args:
        collection: The features to resample.
        n: The number of samples to generate. If left to None this is automatically set to the length of the collection, or to the sum of the number of samples per class, if classDict is specified. If replace is False it cannot be larger than the length of the collection.
        replace: Allow or disallow sampling of the same row more than once.
        classBand: The name of the column in the collection containing the classes to use for stratification. If no classDict is provided, the frequency of the classes in the collection will be used.
        classDict: The stratification dictionary containing either the sample counts or fractions per class to generate.
        seed: The seed to use for reproducibility.
    """
    # Get the classDict as a dictionary
    if isinstance(classDict, ee.Dictionary):
        classDict = classDict.getInfo()
    # Get the column names of the collection
    if isinstance(collection, GeoDataFrame):
        column_names = collection.columns
    elif isinstance(collection, ee.FeatureCollection):
        column_names = collection.first().propertyNames().getInfo()
    else:
        raise ValueError("resample, 'collection' must be either a GeoDataFrame or a FeatureCollection.")
    # Get the size of the collection
    if isinstance(collection, GeoDataFrame):
        size = len(collection)
    elif isinstance(collection, ee.FeatureCollection):
        size = collection.size().getInfo()

    # Check if classBand is in the collection
    if isinstance(classBand, str) and classBand not in column_names:
        raise ValueError(f"resample, the collection does not contain the column '{classBand}'.")
    # Check if classBand is defined when classDict is
    if classDict is not None and classBand is None:
        raise ValueError("resample, 'classBand' must be defined when 'classDict' is.")
    # Check if classBand is defined and classDict is not, generate classDict from collection
    if isinstance(classBand, str) and classDict is None:
        if isinstance(collection, GeoDataFrame):
            classDict = dict(collection[classBand].value_counts(normalize=True))
        else:
            classDict = collection.reduceColumns(ee.Reducer.frequencyHistogram(), [classBand]).get('histogram').getInfo()
            classDict = {k: v/sum(classDict.values()) for k, v in classDict.items()}
    # Check if classDict contains fractions or counts
    if isinstance(classDict, dict):
        if all(isinstance(v, (int, float)) and 0 < v <= 1 for v in classDict.values()):
            pass
        elif all(isinstance(v, (int)) for v in classDict.values()):
            n = sum(classDict.values())
            classDict = {k: v/n for k, v in classDict.items()}
        else:
            raise ValueError("resample, 'classDict' values must be either integers or fractions.")
        # Check if n is larger than the length of the collection
        if replace is False and n > size:
            raise ValueError("resample, 'n' cannot be larger than the length of the collection when replace is 'False'.")

    # Define a function to resample a ee.FeatureCollection
    def _resampleFC(collection, n, replace):
        if replace is True:
            # Get the unique IDs of the collection
            uids = collection.aggregate_array('system:index').getInfo()
            uids_toSample = pd.Series(uids).sample(n, replace=replace, random_state=seed).values
            singleSample = collection.filter(ee.Filter.inList('system:index', list(uids_toSample)))
            # Filter for duplicates, triplicates, etc.
            uniqueIDs, counts = np.unique(uids_toSample, return_counts=True)
            multipleSamples = []
            for i in range(2, max(counts)+1):
                multipleSamples.append(collection.filter(ee.Filter.inList('system:index', list(uniqueIDs[counts >= i]))))
            multipleSample = ee.FeatureCollection(multipleSamples).flatten()
            # Combine the single and multiple samples
            resampled = singleSample.merge(multipleSample)
            return resampled
        else:
            return collection.randomColumn('random', seed).limit(n, 'random', True)

    # Resample the collections stratified or not, with or without replacement
    # Case 1: Resample a GeoDataFrame
    if isinstance(collection, GeoDataFrame):
        if classDict is not None:
            # Check if the number of samples per class can be sampled without replacement
            if replace is False:
                for name, group in collection.groupby('class', group_keys=False):
                    min = int(round(classDict.get(name) * n))
                    if len(group) < min_samples:
                        raise ValueError(f"resample, class '{name}' has less samples than the minimum required for stratified sampling without replacement.")
            # Sample with replacement
            resampled = collection.groupby(classBand, group_keys=False).apply(lambda x: x.sample(n=int(round((classDict.get(x.name))*n)), replace=replacement, random_state=n))
        else:
            # Sample without replacement
            resampled = collection.sample(n=n, replace=replace, random_state=seed)

    # Case 2: Resample a ee.FeatureCollection
    elif isinstance(collection, ee.FeatureCollection):
        if classDict is not None:
            # Get the class, the number of points in the class, and the minium number required to allow stratified sampling without replacement
            classSizes = ee.List([[k, collection.filter(ee.Filter.eq(classBand, k)).size(), int(round(v*n))] for k, v in classDict.items()]).getInfo()
            # Check if the number of samples per class can be sampled without replacement
            if replace is False:
                if any(size < min for c, size, min in classSizes):
                    raise ValueError(f"resample, class '{c}' has less samples than the minimum required for stratified sampling without replacement.")
            # Sample with replacement
            return ee.FeatureCollection([_resampleFC(collection.filter(ee.Filter.eq(classBand, c)), n, replace) for c, n, _ in classSizes]).flatten()
        else:
            # Sample without replacement
            return _resampleFC(collection, n, replace)

