import datetime as dt
import re
from functools import partial, reduce
from typing import List, Union

import pandas as pd
import xarray as xr

from src.constants import (
    DATA_DIR,
)

# historical
runoff_hist_path = DATA_DIR / "historical" / "runoff_glerl_mic_hur_combined.csv"
rnbs_hist_path = DATA_DIR / "historical" / "rnbs_glcc.csv"
precip_hist_path = DATA_DIR / "historical" / "pcp_glerl_lakes_mic_hur_combined.csv"
evap_hist_path = DATA_DIR / "historical" / "evap_glerl_lakes_mic_hur_combined.csv"
water_level_hist_path = DATA_DIR / "historical" / "wl_glcc.csv"


# CFSR
lhfx_cfsr_path = DATA_DIR / "CSFR" / "CFSR_LHFX_Basin_Avgs.csv"
temp_cfsr_path = DATA_DIR / "CFSR" / "CFSR_TMP_Basin_Avgs.csv"
evap_cfsr_path = DATA_DIR / "CFSR" / "CFSR_EVAP_Basin_Avgs.csv"
precip_cfsr_path = DATA_DIR / "CFSR" / "CFSR_APCP_Basin_Avgs.csv"


__all__ = [
    "load_data",
]

column_order = ["sup", "mic_hur", "eri", "ont"]


def read_historical_files(path):
    # FIXME: these methods depend heavily on the column order being correct
    return pd.read_csv(path, index_col="Date", date_format="%Y%m%d")[column_order]


def parse_historical_files(df) -> xr.DataArray:
    """
    Default parser for turning a particular pandas dataframe into an Xarray DataArray.
    Args:
        df: The output of a reader function (assumed, but not required to be the default reader)

    Returns:
        An Xarray dataset.

    """
    return xr.DataArray(
        df,
        dims=["Date", "lake"],
        coords={"Date": df.index, "lake": df.columns},
    )


class FileReader(object):

    def __init__(
        self,
        reader=read_historical_files,
        parser=parse_historical_files,
        series_name=None,
        **metadata
    ):
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

    def __call__(self, path, series_name=None) -> xr.DataArray:
        arr: xr.DataArray = self._parser(self._reader(path))
        if series_name:
            arr = arr.rename(series_name)
        else:
            arr = arr.rename("value")
        return arr.assign_attrs(**self.metadata)


def read_cfsr_files(path):
    """
    Read CSVs which have the format of {Type}{Lake} for each column. These consist of multiple columns for each lake.
    For example, each column in temperature is BasinSuperior, BasinErie, WaterMichigan, etc.
    Args:
        path: Path of the file to read in

    Returns:
        A pandas dataframe, modified to be in a "long" format, i.e. Date -> lake -> value_type
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
        .melt(value_name="value", id_vars="Date")
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


def parse_cfsr_data(df):
    """
    Parse the output of the CFSR file reader function into an xarray.DataArray.
    Args:
        df:

    Returns:

    """

    lake_arrays = []
    grps = []
    for grp, arr in df.groupby("type"):
        grps.append(grp)
        pivot_df = arr.pivot(columns="lake", values="value", index="Date")
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
    "rnbs_hist": (FileReader(source="historical"), rnbs_hist_path),
    "precip_hist": (FileReader(source="historical"), precip_hist_path),
    "precip_cfsr": (
        FileReader(
            reader=read_cfsr_files,
            parser=parse_cfsr_data,
            source="CFSR",
        ),
        evap_cfsr_path,
    ),
    "evap_hist": (
        FileReader(source="historical"),
        evap_hist_path,
    ),
    "evap_cfsr": (
        FileReader(
            reader=read_cfsr_files,
            parser=parse_cfsr_data,
            source="CFSR",
        ),
        evap_cfsr_path,
    ),
    "runoff_hist": (
        FileReader(reader=partial(pd.read_csv, date_format="%Y%m", index_col="Date")),
        runoff_hist_path,
    ),
    "water_level": (FileReader(source="historical"), water_level_hist_path),
    "temp_cfsr": (
        FileReader(
            reader=read_cfsr_files,
            parser=parse_cfsr_data,
            source="CFSR",
            units="K",
        ),
        temp_cfsr_path,
    ),
}


def read_series(series):
    read_fn, path = series_map[series]
    return read_fn(path, series_name=series)


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
        data = map(read_series, series)
        dataset = reduce(lambda a, x: xr.merge([a, x]), data).transpose(
            "Date", "lake", ...
        )
        return dataset
