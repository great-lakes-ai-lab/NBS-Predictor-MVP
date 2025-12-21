from functools import partial

import numpy as np
import pytest
import xarray as xr
from skfda.representation.basis import BSplineBasis
from xarray.testing import assert_allclose

from src.preprocessing.preprocessing import (
    BasisFunctionTransformer,
    CreateMonthDummies,
    SeasonalFeatures,
    WindowTransformer,
    XArrayStandardScaler,
    time_window_generator,
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


@pytest.mark.parametrize("reindex", [True, False])
def test_time_windows(lake_data, reindex):
    windows = time_window_generator(
        lake_data, train_size=48, test_size=12, reindex_domain=reindex
    )
    for train, test in windows:
        assert train.shape[0] == 48 and test.shape[0] == 12
        if reindex:
            assert np.all(train.coords[train.dims[0]] == np.arange(0, 48))


def test_window_transformer(lake_data):
    train_window = 48
    test_window = 12
    transformer = WindowTransformer(train_size=train_window, test_size=test_window)
    train_out, test_out = transformer.fit_transform(lake_data)

    for data in (train_out, test_out):
        assert isinstance(data, xr.DataArray)

    assert train_out.shape[1] == train_window
    assert test_out.shape[1] == test_window


def test_functional_basis(lake_data):
    # windowedData
    lake_data = lake_data.dropna("Date")
    train_windows, test_windows = WindowTransformer().fit_transform(lake_data)

    train_basis = BasisFunctionTransformer(
        default_basis=partial(BSplineBasis, n_basis=5)
    )

    train_data = train_basis.fit_transform(train_windows)
    inverted_data = train_basis.inverse_transform(train_data)

    assert isinstance(train_data, xr.DataArray)
    assert inverted_data.dims == train_windows.dims
