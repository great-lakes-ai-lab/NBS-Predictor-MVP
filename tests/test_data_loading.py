import pytest
import xarray as xr

from src.step1_data_loading.data_loading import load_data, read_series


@pytest.mark.parametrize("series", ["rnbs", "precip", "runoff", "evap", "water_level"])
def test_single_series(series):
    df = read_series(series)
    assert isinstance(df, xr.DataArray)


def test_multi_series():
    series_list = ["rnbs", "precip", "runoff", "evap", "water_level"]
    covars = load_data(series_list)

    assert list(covars.indexes["variable"]) == series_list
    assert isinstance(covars, xr.DataArray)
    assert covars.dims == ("Date", "lake", "variable")
