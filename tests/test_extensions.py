"""Tests for `extensions` module."""

import pytest
from geopandas import GeoDataFrame
from shapely.geometry import Point
from unittest.mock import patch, MagicMock

import ee

from geePipe import extensions
from tests import fake_ee


@pytest.fixture
def sample_geodataframe():
    """Create a minimal GeoDataFrame for testing."""
    data = {
        "id": [1, 2],
        "geometry": [Point(0, 0), Point(1, 1)],
    }
    gdf = GeoDataFrame(data, crs="EPSG:4326")
    return gdf

@patch.object(ee, "FeatureCollection", fake_ee.FeatureCollection)
@patch.object(ee, "Image", fake_ee.Image)
def test_reduceRegionsGDF_wGDF(eeImage(), sample_geodataframe):
    """Test the reduceRegionsGDF function with pre-existing GDF objects."""


    # Configure the mocks
    mock_image.bandNames().getInfo.return_value = ["band1", "band2"]
    mock_image.select(0).projection().getInfo.return_value = {
        "crs": "EPSG:4326",
        "transform": [0.1, 0, 0, 0, -0.1, 0],
    }
    mock_reducer.getOutputs().getInfo.return_value = ["mean", "stdDev"]
    mock_image.reduceRegions.return_value.getInfo.return_value = [
        {"id": 1, "mean": 10, "stdDev": 1},
        {"id": 2, "mean": 20, "stdDev": 2},
    ]

    # Add the custom function to ee.Image
    mypackage.add_custom_functions_to_eeImage()

    # Call the function
    result = mock_image.reduceRegionsGDF(
        collection=sample_geodataframe,
        reducer=mock_reducer,
        scale=30,
        crs="EPSG:4326",
    )

    # Verify the output is a GeoDataFrame
    assert isinstance(result, GeoDataFrame)

    # Check the column names
    expected_columns = ["id", "mean", "stdDev", "geometry"]
    assert all(col in result.columns for col in expected_columns)

    # Check the values
    assert result.loc[result["id"] == 1, "mean"].values[0] == 10
    assert result.loc[result["id"] == 2, "stdDev"].values[0] == 2

    # Verify the projection
    assert result.crs.to_string() == "EPSG:4326"
