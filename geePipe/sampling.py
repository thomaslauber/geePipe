""" Extensions to Google Earth Engine Classes. """

# Author: Thomas Lauber
### !! could still implement approximate distances

import concurrent.futures
from typing import Optional
import warnings

import ee
import geemap
from geopandas import GeoDataFrame, points_from_xy
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .utils import raiseEEException
from .extensions import add_custom_functions_to_eeImage
add_custom_functions_to_eeImage()


def _random_coordinates(
        n: int,
        seed: Optional[int] = 0
) -> np.ndarray:
    """ Generates random coordinates. This function is based on the R function geosphere::randomCoordinates.

    Args:
        n: The number of point locations to generate.
        seed: The seed to use for reproducibility. The default is 0. To make it truly random, set it to None.
    """
    # Sanity check
    if n < 1:
        raise ValueError('Number of point locations to create should be >= 1.')
    n = int(np.round(n))
    # Make this example reproducible
    if seed is not None:
        rng = np.random.RandomState(int(seed))
    # Get random numbers
    z = rng.uniform(size=n) * 2 - 1
    t = rng.uniform(size=n) * 2 * np.pi
    # Get the uniformly distributed points
    r = np.sqrt(1-z**2)
    x = r * np.cos(t)
    y = r * np.sin(t)
    # Get the angles
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    # Get the coordinates
    lat = theta * 180 / np.pi - 90
    lon = phi * 180 / np.pi
    return np.column_stack((lon, lat))

def _regular_coordinates(
        n: int
) -> np.ndarray:
    """ Generates regular coordinates. This function is based on the R function geosphere::regularCoordinates.
    The number of generated points increases as: N(n)=5n^2+n

    Args:
        n: The number of point locations to generate.
        seed: The seed to use for reproducibility. The default is 0. To make it truly random, set it to None.
    """
    # Sanity check
    if n < 1:
        raise ValueError('Number of point locations to create should be >= 1.')
    n = int(np.round(n))
    # Subdivision angle
    beta = 0.5 * np.pi / n
    # Line segment length
    A = 2 * np.sin(beta/2)
    # Endcap
    points = np.array([[0, 0, 1], [0, 0, -1]])
    # Rings
    Z = np.array([np.cos(n_i * beta) for n_i in range(1, n+1)])
    R = np.array([np.sin(n_i * beta) for n_i in range(1, n+1)])
    M = np.round(R * 2 * np.pi / A)
    for i in range(n):
        Alpha = np.arange(0, int(M[i]))/M[i] * 2 * np.pi
        X = np.cos(Alpha) * R[i]
        Y = np.sin(Alpha) * R[i]
        points = np.vstack((points, np.column_stack((X, Y, np.repeat(Z[i], len(X))))))
        if i != range(n)[-1]:
            points = np.vstack((points, np.column_stack((X, Y, np.repeat(-Z[i], len(X))))))
    # Create lon & lat
    r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
    theta = np.arccos(points[:, 2] / r)
    phi = np.arctan2(points[:, 1], points[:, 0])
    lat = np.clip(theta * 180 / np.pi - 90, -89.99, 89.99)
    lon = phi * 180 / np.pi
    return np.column_stack((lon, lat))

def _fibonacci_coordinates(
        n: int
) -> np.ndarray:
    """ Generates coordinates using a Fibonacci lattice. This function is based on the work of Martin Roberts (https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/#more-3069)
    and Alvaro Gonzalez, 2010. Measurement of Areas on a Sphere Using Fibonacci and Latitude-Longitude Lattices. Mathematical Geosciences 42(1), p. 49-64.

    Args:
        n: The number of point locations to generate.
    """
    # Sanity check
    if n < 1:
        raise ValueError('Number of point locations to create should be >= 1.')
    n = int(np.round(n))

    if n >= 600000:
      epsilon = 214
    elif n >= 400000:
      epsilon = 75
    elif n >= 11000:
      epsilon = 27
    elif n >= 890:
      epsilon = 10
    elif n >= 177:
      epsilon = 3.33
    elif n >= 24:
      epsilon = 1.33
    else:
      epsilon = 0.33

    goldenRatio = (1 + 5**0.5)/2
    i = np.arange(0, n)
    theta = 2 * np.pi * i / goldenRatio
    phi = np.arccos(1 - 2*(i+epsilon)/(n-1+2*epsilon))
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    lat = np.arcsin(z) * 180 / np.pi
    lon = np.arctan2(y, x) * 180 / np.pi
    return np.column_stack((lon, lat))

def _approximateN(
        n: int,
        mask: ee.Image,
        samplingType: str,
        crs: None
) -> int:
    """ Generating regular coordinates across a mask can only approximate the sample size. Therefore, the angle/distance between the points needs to be adjusted iteratively, until the number of generated point locations is roughly n.
        To do so, this function .. many of the grid points are located inside the mask
        .. A polynomial function is fit to approximate the size of the
        grid that will generate n sample points across the mask.

        Args:
            n: The number of point locations to generate.
            mask: An image mask that will be filled with points.
            samplingType: The type of sampling to use. Either 'regular' or 'fibonacci'.
            crs: The projection to work in. If unspecified, the projection of the image's first band will be used.
        """
    # Sanity check
    if n is None or n < 1:
        raise ValueError('Number of point locations to create should be >= 1.')
    n = int(np.round(n))
    if samplingType not in ['regular', 'fibonacci']:
        raise ee.ee_exception.EEException("_approximateN: argument 'samplingType' needs to be one of 'regular' or 'fibonacci'.")
    if crs is None:
        crs = image.select(0).projection().getInfo()['crs']

    # Define polynomial functions
    def _polynomial_function(x, a, b, c):
        return a * x**2 + b * x + c
    def _polynomial_function_reversed(y, a, b, c):
        coeff_c = c - y
        d = (b**2) - (4 * a * coeff_c)
        x1, x2 = (-b + (d**0.5)) / (2 * a), (-b - (d**0.5)) / (2 * a)
        if a < 0:
            return sorted(list([x1, x2]), reverse=False)[0]
        else:
            return sorted(list([x1, x2]), reverse=True)[0]

    # Define the lower limit for the number of points needed
    """
    For regular sampling, the number of generated points increases as: n=5N^2+N
    For fibonacci sampling, the number of generated points is n=N. 
    """
    if samplingType == 'regular':
        lower_limit = np.floor(np.max(np.roots([5, 1, -n])))     # solving quadratic equation: 5n^2 + n - N(n) = 0
    if samplingType == 'fibonacci':
        lower_limit = n

    # Get an estimate for the upper limit of the points needed, we increase by a factor of 10
    n_upper = 0
    upper_limit = lower_limit
    while n_upper < n:
        # Set upper limits as 10 times the lower limit (1st try) or 10 times the previous upper limit
        if samplingType == 'regular':
            upper_n = 5 * upper_limit**2 + upper_limit
            upper_limit = np.ceil(np.max(np.roots([5, 1, -upper_n*10])))
            points = _regular_coordinates(upper_limit)
        if samplingType == 'fibonacci':
            upper_limit = upper_limit*10
            points = _fibonacci_coordinates(upper_limit)
        # Check how many points are inside the mask
        gdf = GeoDataFrame(geometry=points_from_xy(points[:, 0], points[:, 1]), crs=crs).iloc[:, -1:]
        within_mask = mask.reduceRegionsGDF(gdf, ee.Reducer.first())['first'].fillna(0).astype(bool)
        n_upper = within_mask.sum()

    # Now interpolate between the lower and upper limit
    x, y = np.linspace(lower_limit, upper_limit, 5, dtype=int), []
    for x_i in x:
        if samplingType == 'regular':
            points = _regular_coordinates(x_i)
        if samplingType == 'fibonacci':
            points = _fibonacci_coordinates(x_i)
        gdf = GeoDataFrame(geometry=points_from_xy(points[:, 0], points[:, 1]), crs=crs).iloc[:, -1:]
        within_mask = mask.reduceRegionsGDF(gdf, ee.Reducer.first())['first'].fillna(0).astype(bool)
        y.append(within_mask.sum())
    params, _ = curve_fit(_polynomial_function, np.asarray(x), np.asarray(y))
    a, b, c = params
    N = _polynomial_function_reversed(n, a, b, c)
    return N

# def _approximateN(d):
#     """ Generating coordinates on a Fibonacci lattice can only be done by defining the number of samples.
#         Therefore, the distance between points needs to be adjusted iteratively. To do so, this function
#         generates 20 different grids and evaluates the nearest neighbour distance across a maximum of
#         10,000 points. A polynomial function is fit to approximate the number of samples that can be
#         generated with a minimum nearest neighbor distance ~> d. """
#     def nndist(points):
#         points[:, [0, 1]] = points[:, [1, 0]]
#         locs = np.deg2rad(points)
#         tree = BallTree(locs, metric="haversine")
#         dists, ilocs = tree.query(locs, k=2)
#         distToItself, nndist = np.array(list(zip(*dists)))
#         return nndist * 6367
#
#     def _linear_function(x, a, b):
#         return a * x + b
#
#     def _linear_function_reversed(y, a, b):
#         return (y - b) / a
#
#     # Approximate N needed to get roughly d spaced points
#     x, y = np.logspace(1, 6, 10, dtype=int), []
#     for x_i in x:
#         points = _fibonacci_coordinates(x_i)
#         y.append(np.min(nndist(points)))
#     # Fit a curve to the simulated distances
#     params, _ = curve_fit(_linear_function, np.log10(x), np.log10(y))
#     N = 10 ** _linear_function_reversed(np.log10(d), params[0], params[1])
#
#     # Approximate again at a smaller scale
#     x, y = np.linspace(int(N*0.8), int(N*1.2), 20, dtype=int), []
#     for x_i in x:
#         points = _fibonacci_coordinates(x_i)
#         y.append(np.min(nndist(points)))
#     # Fit a curve to the simulated distances
#     params, _ = curve_fit(_linear_function, np.log10(x), np.log10(y))
#     N = 10 ** _linear_function_reversed(np.log10(d), params[0], params[1])
#     return int(np.floor(N))

# A custom function to generate point locations across an image's mask
def generateSample(
        image: ee.Image,
        numPoints: int,
        # region: Optional[ee.Geometry] = None,
        samplingType: Optional[str] = "regular",
        seed: Optional[int] = 0,
        crs: Optional[str] = None,
        verbose: Optional[int] = 0
) -> GeoDataFrame:
    """ This function generates points on a sphere.

    This function generates point locations on a sphere by filling the mask of an ee.Image's first band using random or regular sampling methods.
    In case of regular sampling methods, the function will try to fill the mask as closely as possible to the requested number of points.

    Args:
        image: The image, whose mask will be filled with points.
        numPoints: The number of point locations to generate.
        region: The region to sample from. If unspecified, the input image's whole footprint is used.
        samplingType: The type of sampling to use. Either 'random', 'regular' or 'fibonacci'. The default is 'regular'.
        seed: The seed to use for reproducibility. The default is 0. To make it truly random, set it to None.
        crs: The projection to work in. If unspecified, the projection of the image's first band will be used.
        verbose: Controls the verbosity. The default is 0. Set to 1 to print messages.
    """
    # Checks
    if not isinstance(image, ee.Image):
        raiseEEException('generateSample', 'image', 'Image', type(image))
    if not isinstance(numPoints, int):
        raiseEEException('generateSample', 'numPoints', 'integer', type(numPoints))
    # if region is None:
    #     region = image.geometry()
    # if not isinstance(region, ee.Geometry):
    #     raiseEEException('generateSample', 'region', 'Geometry', type(region))
    # region = region.getInfo()
    if samplingType not in ['random', 'regular', 'fibonacci']:
        raise ee.ee_exception.EEException("generateSample: argument 'samplingType' needs to be one of 'random', 'regular' or 'fibonacci'.")
    # Cast the input
    mask = image.select(0).mask()
    n = numPoints
    if verbose != 0:
        verbose = 1
    if verbose == 1:
        print('Generating {n} {samplingType}ly sampled points across the mask.'.format(
            samplingType=samplingType, n='~'+str(n) if samplingType == 'regular' else n))
    if crs is None:
        crs = mask.projection().getInfo()['crs']

    # Generate a sample of n points using the specified sampling type
    if samplingType == "random":
        try_nr = 1
        ppoints = np.empty((0, 2))
        while try_nr < int(1e4):
            seed += 1
            points = _random_coordinates(n*2, seed)
            gdf = GeoDataFrame(points, geometry=points_from_xy(points[:, 0], points[:, 1]), crs=crs).iloc[:, -1:]
            within_mask = mask.reduceRegionsGDF(gdf, ee.Reducer.first())['first'].fillna(0).astype(bool).values
            ppoints = np.concatenate([ppoints, points[within_mask]])
            if len(ppoints) >= n:
                ppoints = ppoints[0:n]
                return GeoDataFrame(geometry=points_from_xy(ppoints[:, 0], ppoints[:, 1]), crs=crs)
            try_nr += 1
    else:
        # Approximate how the mask can be filled with regularly sampled points
        N = _approximateN(n, mask, samplingType, crs)
        # Try the floor and the ceiling of N
        if samplingType == "regular":
            points_floor, points_ceil = _regular_coordinates(np.floor(N)), _regular_coordinates(np.ceil(N))
        if samplingType == "fibonacci":
            points_floor, points_ceil = _fibonacci_coordinates(np.floor(N)), _fibonacci_coordinates(np.ceil(N))
        points_floor_gdf = GeoDataFrame(geometry=points_from_xy(points_floor[:, 0], points_floor[:, 1]),
                                        crs=crs).iloc[:, -1:]
        points_ceil_gdf = GeoDataFrame(geometry=points_from_xy(points_ceil[:, 0], points_ceil[:, 1]),
                                       crs=crs).iloc[:, -1:]
        within_mask_floor = mask.reduceRegionsGDF(points_floor_gdf, ee.Reducer.first())['first'].fillna(0).astype(bool)
        within_mask_ceil = mask.reduceRegionsGDF(points_ceil_gdf, ee.Reducer.first())['first'].fillna(0).astype(bool)
        ppoints_floor, ppoints_ceil = points_floor[within_mask_floor], points_ceil[within_mask_ceil]
        absDiff_floor, absDiff_ceil = abs(n - len(ppoints_floor)), abs(n - len(ppoints_ceil))
        if absDiff_floor <= absDiff_ceil:
            return GeoDataFrame(geometry=points_from_xy(ppoints_floor[:, 0], ppoints_floor[:, 1]), crs=crs).iloc[:, -1:]
        else:
            return GeoDataFrame(geometry=points_from_xy(ppoints_ceil[:, 0], ppoints_ceil[:, 1]), crs=crs).iloc[:, -1:]
