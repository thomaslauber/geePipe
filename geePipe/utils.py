""" Utils methods. """

# Author: Thomas Lauber

from typing import Optional
import warnings

import ee
from geopandas import GeoDataFrame
import numpy as np


def raiseEEException(function, argument, expected, actual):
    raise ee.ee_exception.EEException(
        f"{function}, argument '{argument}': Invalid type.\n"
        f"Expected type: {expected}.\n"
        f"Actual type: {actual}.")




