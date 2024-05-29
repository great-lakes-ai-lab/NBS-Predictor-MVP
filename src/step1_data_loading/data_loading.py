import pandas as pd
import xarray as xr
from typing import List, Union

from resources.ciglr import (
    runoff_path,
    evap_lake_path,
    pcp_lake_path,
    water_level_path,
    rnbs_path,
)
from functools import partial, reduce

default_reader = partial(pd.read_csv, index_col="Date", date_format="%Y%m%d")

# map the reader function to a specific series name. When requested,
series_map = {
    "rnbs": (default_reader, rnbs_path),
    "precip": (default_reader, pcp_lake_path),
    "evap": (default_reader, evap_lake_path),
    "runoff": (
        partial(pd.read_csv, date_format="%Y%m", index_col="Date"),
        runoff_path,
    ),
    "water_level": (default_reader, water_level_path),
}
column_order = ["sup", "mic_hur", "eri", "ont"]


def convert_lake_df_to_xarray(lake_df, **kwargs):
    """
    Small helper function to turn the default CSV file format to a Data array with proper dimensions, namely
    date for the rows and lake for the columns
    Args:
        lake_df: The dataframe with date as the rows and lakes as the columns
        **kwargs: Keyword options to be passed to xr.DataArray - primarily used to set the name of the series
    Returns:

    """
    return xr.DataArray(
        lake_df,
        **kwargs,
        dims=["Date", "lake"],
        coords={"Date": lake_df.index, "lake": column_order}
    )


def read_series(series):
    read_fn, path = series_map[series]
    df = read_fn(path)[column_order]
    return xr.DataArray(df, dims=["Date", "lake"], coords={"Date": df.index, "lake": df.columns}, name=series)


def load_data(series: Union[str, List[str]]):
    """
    Load a data series based on name. Requires that raw files be available in the DATA_DIR from constants.py
    Args:
        series: the name of the series. Valid names include "rnbs", "precip", "evap", "runoff", "water_level"
    Returns:
        An xarray DataArray containing each series OR a pandas dataframe if only one series is requested.

    """

    if isinstance(series, str):
        return read_series(series)
    else:
        data = map(lambda j: read_series(j), series)
        dataset = (
            reduce(lambda a, x: xr.merge([a, x]), data)
            .to_array()
            .transpose("Date", "lake", "variable")
        )
        return dataset
