""" Extensions to Google Earth Engine Classes. """

# Author: Thomas Lauber

import ee
import geemap


def reduceRegionsGDF(self, gdf: GeoDataFrame, subset_size=1000,
               scale=None, crs=None, crsTransform=None, n_jobs=None):
    """ Sample an image at the points of interest.

    This function samples an image at the points of interest (GeoDataFrame). It is a wrapper
    around the `ee.Image.reduceRegions` function. It takes care of the projection of the points
    of interest and the image. It also splits the points of interest into subsets to avoid
    memory issues.

    Parameters
    ----------
    image : ee.Image
        The image to sample.
    gdf : geopandas.GeoDataFrame
        The points of interest. If the crs is not the same as the one of the image, the points
        will get reprojected.
    subset_size : int, optional
        The number of points of interest to sample at once. The default is 1000.
    scale: int, optional
        A nominal scale in meters of the projection to work in.
    crs : str, optional
        The projection to work in. If unspecified, the projection of the image's first band will be used.
        If specified in addition to scale, rescaled to the specified scale.
    crsTransform : list, optional
        The list of CRS transform values. This is a row-major ordering of the 3x2 transform matrix.
        This option is mutually exclusive with 'scale', and replaces any transform already set on the projection.
    n_jobs : int, optional
        The number of parallel workers to use. If None, uses all available CPUs + 4, with at most 32.
    """

    # If no projections are specified, get them from the image
    if crs is None and crsTransform is None:
        if crs and crsTransform in self.propertyNames().getInfo():
            crs, crsTransform = self.get('crs'), self.get('crsTransform')
        else:
            proj = self.select(0).projection().getInfo()
            crs, crsTransform = proj['crs'], proj['transform']
    # Check if the points have the same crs, if not reproject the points
    if crs != gdf.crs.to_string():
        gdf = gdf.to_crs(crs)
    # Define break points for the subsetting
    nr_of_subsets = len(gdf)//subset_size + 1
    # Subset the geopandas dataframe
    gdf_subsets = np.array_split(gdf, nr_of_subsets)
    # Get all the individual results per sublist
    def _get_results(image, gdf_subset, scale, crs, crsTransform):
        # Cast the geopandas df into a ee.FeatureCollection
        fc = geemap.geopandas_to_ee(gdf_subset)
        # Sample the image
        sampled_data = image.reduceRegions(
            collection=fc,
            reducer=ee.Reducer.first(),
            crs=crs,
            crsTransform=crsTransform,
            scale=scale
        )
        # Convert the sampled data to a pandas dataframe
        sampled_data_df = ee.data.computeFeatures({
            'expression': sampled_data,
            'fileFormat': 'PANDAS_DATAFRAME'
        })
        if not any(sampled_data_df.columns == 'first'):
            sampled_data_df['first'] = np.nan
        # Add the geometry column
        sampled_data_df['geometry'] = gdf_subset['geometry'].values
        return sampled_data_df
    # Run the sampling function with multiple workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(_get_results, self, gdf_subset, scale, crs, crsTransform)
                   for gdf_subset in gdf_subsets]
    # Get the results in one dataframe
    results = pd.DataFrame()
    for future in futures:
        results = pd.concat([results, future.result()]).drop(columns='geo')
    results = GeoDataFrame(results, geometry=results['geometry'])
    return results









