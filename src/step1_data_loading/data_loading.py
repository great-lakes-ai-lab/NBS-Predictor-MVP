import datetime as dt
import re
from functools import partial, reduce
from typing import List, Union

import pandas as pd
import xarray as xr

from src.constants import (
    runoff_path,
    rnbs_path,
    pcp_lake_path,
    evap_lake_path,
    water_level_path,
    temp_path,
)

__all__ = [
    "load_data",
]

default_reader = partial(pd.read_csv, index_col="Date", date_format="%Y%m%d")


column_order = ["sup", "mic_hur", "eri", "ont"]


def default_parser(df) -> xr.DataArray:
    """
    Default parser for turning a particular pandas dataframe into an Xarray DataArray.
    Args:
        df:

    Returns:

    """
    return xr.DataArray(
        df,
        dims=["Date", "lake"],
        coords={"Date": df.index, "lake": df.columns},
    )


class FileReader(object):

    def __init__(self, reader=default_reader, parser=default_parser, **metadata):
        """
        Helper class to connect a particular CSV reader and an Xarray formatter. There are default
        options in use for files with 4 series, one for each lake. This allows for a custom
        formatter to be used as well. This is a callable, so once instantiated, it can be used
        like a normal function.

        Args:
            reader: A function for reading in data.
            parser: A function parsing the data into XArray format
            **metadata: Any keyword arguments to append as attributes to the output XArray
        """
        super().__init__()
        self._reader = reader
        self._parser = parser
        self.metadata = metadata or {}

    def __call__(self, path) -> xr.DataArray:
        arr: xr.DataArray = self._parser(self._reader(path))
        return arr.assign_attrs(**self.metadata)


def read_type_separated_csv(path):
    """
    Read CSVs which have the format of {Type}{Lake} for each column. These consist of multiple columns for each lake.
    For example, each column in temperature is BasinSuperior, BasinErie, WaterMichigan, etc.
    Args:
        path: Path of the file to read in

    Returns:
        a pandas dataframe

    """
    df = pd.read_csv(path)

    date_index = pd.Index(
        list(
            map(
                lambda date_args: dt.datetime(*date_args, 1),
                zip(df["year"], df["month"]),
            )
        ),
        name="Date",
    )
    melted_data = (
        df.set_index(date_index)
        .drop(["year", "month"], axis=1)
        .reset_index()
        .melt(value_name="temperature", id_vars="Date")
    )
    new_cols = pd.DataFrame(
        map(lambda x: re.findall("[A-Z][^A-Z]*", x), melted_data["variable"]),
        index=melted_data.index,
        columns=["type", "lake"],
    )
    array_data = melted_data.merge(new_cols, left_index=True, right_index=True).drop(
        ["variable"], axis=1
    )
    return array_data


def parse_read_separated_csv(df):

    lake_arrays = []
    grps = []
    for (grp,), arr in df.groupby(["type"]):
        grps.append(grp)
        pivot_df = arr.pivot(columns="lake", values="temperature", index="Date")
        array = (
            pivot_df.assign(mic_hur=pivot_df["Michigan"] + pivot_df["Huron"])
            .drop(["Michigan", "Huron"], axis=1)
            .rename({"Erie": "eri", "Ontario": "ont", "Superior": "sup"}, axis=1)
        )[column_order]

        lake_x_array = xr.DataArray(
            array,
            coords={"Date": array.index, "lake": column_order},
            dims=["Date", "lake"],
            name=grp,
        )
        lake_arrays.append(lake_x_array)

    full_set = xr.concat(lake_arrays, pd.Index(grps, name="type"))
    return full_set


# map the reader function to a specific series name. When requested,
series_map = {
    "rnbs": (FileReader(source="CIGLR"), rnbs_path),
    "precip": (FileReader(source="CIGLR"), pcp_lake_path),
    "evap": (FileReader(source="CIGLR"), evap_lake_path),
    "runoff": (
        FileReader(reader=partial(pd.read_csv, date_format="%Y%m", index_col="Date")),
        runoff_path,
    ),
    "water_level": (FileReader(source="CIGLR"), water_level_path),
    "temp": (
        FileReader(
            reader=read_type_separated_csv,
            parser=parse_read_separated_csv,
            source="CFSR",
            units="K",
        ),
        temp_path,
    ),
}


def read_series(series):
    read_fn, path = series_map[series]
    return read_fn(path).rename(series)


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
        dataset = reduce(lambda a, x: xr.merge([a, x]), data).transpose(
            "Date", "lake", ...
        )
        return dataset
