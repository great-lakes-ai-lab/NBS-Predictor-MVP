import pytest
import xarray as xr

from src.step1_data_loading.data_loading import load_data, series_map


@pytest.mark.parametrize("series", series_map.keys())
def test_single_series(series):
    df = load_data(series)
    assert isinstance(df, xr.DataArray)


def test_multi_series():
    series_list = list(series_map.keys())
    covars = load_data(series_list)

    assert list(covars.data_vars) == series_list
    assert isinstance(covars, xr.Dataset)
    assert list(covars.dims)[:2] == ["Date", "lake"]
