from src.step2_preprocessing.preprocessing import XArrayScaler
from xarray.testing import assert_allclose


def test_default_scaling(lake_data):
    scaler = XArrayScaler()
    scaled_xarray = scaler.fit_transform(lake_data)

    inversed = scaler.inverse_transform(scaled_xarray)

    assert scaled_xarray.shape == lake_data.shape
    assert_allclose(lake_data, inversed)


def test_single_series_scaling(lake_data):
    scaler = XArrayScaler()
    subset = lake_data.sel(variable="runoff")
    scaled_xarray = scaler.fit_transform(subset)

    assert scaled_xarray.max() < subset.max()
