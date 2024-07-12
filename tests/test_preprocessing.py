from xarray.testing import assert_allclose
import xarray as xr

from src.step2_preprocessing.preprocessing import (
    XArrayStandardScaler,
    CreateMonthDummies,
    SeasonalFeatures,
)
from src.utils import flatten_array


def test_default_scaling(lake_data):
    scaler = XArrayStandardScaler()
    scaled_xarray = scaler.fit_transform(lake_data)

    inversed = scaler.inverse_transform(scaled_xarray)

    assert scaled_xarray.shape == lake_data.shape
    assert_allclose(lake_data, inversed)


def test_single_series_scaling(lake_data):
    scaler = XArrayStandardScaler()
    subset = lake_data.sel(variable="runoff")
    scaled_xarray = scaler.fit_transform(subset)

    assert scaled_xarray.max() < subset.max()


def test_month_dummies(lake_data):
    enc = CreateMonthDummies()
    enc.fit(lake_data)
    months = enc.transform(lake_data)

    assert months.shape[1] == 11
    assert isinstance(months, xr.DataArray)


def test_flatten_df(lake_data):
    flat_data = flatten_array(lake_data)

    assert flat_data.shape[0] == lake_data.shape[0]
    assert flat_data.shape[1] == 4 * lake_data.shape[1]  # 4 lakes, then each variable


def test_seasonal_features(lake_data):

    seasoner = SeasonalFeatures()

    seasonal_features = seasoner.fit_transform(
        lake_data.sel(variable=["evap", "rnbs"])
    )  # doesn't matter, since it uses the indexes

    assert seasonal_features.shape[1] == 2
    assert isinstance(seasonal_features, xr.DataArray)
