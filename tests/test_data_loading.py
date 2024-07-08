import pytest
import xarray as xr

from src.step1_data_loading.data_loading import load_data, input_map, forecast_map


@pytest.mark.parametrize("series", input_map.keys())
def test_single_series(series):
    df = load_data(series)
    assert isinstance(df, xr.DataArray)


@pytest.mark.parametrize(
    "input_mapping", [input_map, forecast_map], ids=["inputs", "forecasts"]
)
def test_multi_series(input_mapping, request):
    id = request.node.callspec.id.split("-")
    series_list = list(input_mapping.keys())
    covars = load_data(series_list, data_type=id[0])

    assert list(covars.data_vars) == series_list
    assert isinstance(covars, xr.Dataset)
    assert list(covars["precip"].dims)[:2] == ["Date", "lake"]
