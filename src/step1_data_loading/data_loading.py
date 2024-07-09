import datetime as dt
import re
from functools import partial
from typing import List, Union

import pandas as pd
import xarray as xr

from src.constants import (
    DATA_DIR,
)

# glcc
runoff_hist_path = DATA_DIR / "GLCC" / "runoff_glerl_mic_hur_combined.csv"
rnbs_hist_path = DATA_DIR / "GLCC" / "rnbs_glcc.csv"
precip_hist_path = DATA_DIR / "GLCC" / "pcp_glerl_lakes_mic_hur_combined.csv"
evap_hist_path = DATA_DIR / "GLCC" / "evap_glerl_lakes_mic_hur_combined.csv"
water_level_hist_path = DATA_DIR / "GLCC" / "wl_glcc.csv"


# CFSR
lhfx_cfsr_path = DATA_DIR / "CFSR" / "CFSR_LHFX_Basin_Avgs.csv"
temp_cfsr_path = DATA_DIR / "CFSR" / "CFSR_TMP_Basin_Avgs.csv"
evap_cfsr_path = DATA_DIR / "CFSR" / "CFSR_EVAP_Basin_Avgs.csv"
precip_cfsr_path = DATA_DIR / "CFSR" / "CFSR_APCP_Basin_Avgs.csv"


# CFS
temp_cfs_path = DATA_DIR / "CFS" / "CFS_TMP_Basin_Avgs.csv"
precip_cfs_path = DATA_DIR / "CFS" / "CFS_APCP_Basin_Avgs.csv"
evap_cfs_path = DATA_DIR / "CFS" / "CFS_EVAP_Basin_Avgs.csv"


# L2SWBM


# Only interact with the data through the load_data
__all__ = ["load_data", "input_map", "forecast_map"]

column_order = ["sup", "mic_hur", "eri", "ont"]
name_remap = {
    "Erie": "eri",
    "Ontario": "ont",
    "Superior": "sup",
    "Huron": "hur",
    "Michigan": "mic",
}  # mic_hur is done while creating the column


def read_historical_files(path, reader_args=None) -> xr.DataArray:
    """
    Read in glcc files. These have a simple format. For a given series, there are 5 columns: date,
    and the 4 great lakes. Once the file is read in, ensure that the columns are in the correct order.
    It is also assumed that the columns will have the following names: "sup", "mic_hur", "eri", "ont".

    Args:
        path: The path to the file
        reader_args: Any arguments that are passed to pd.read_csv. If none are provided,
        default options are assumed.

    Returns:
        An Xarray DataArray with the glcc data, with "Date" as the leading dimension and "lake" as the
        second.

    """

    reader_args = reader_args or {"index_col": "Date", "date_format": "%Y%m%d"}
    df = pd.read_csv(path, **reader_args)[column_order]
    return xr.DataArray(
        df,
        dims=["Date", "lake"],
        coords={"Date": df.index, "lake": df.columns},
    )


def read_cfsr_files(path, reader_args=None) -> xr.DataArray:
    """
    Read CSVs which have the format of {Type}{Lake} for each column. These consist of multiple columns for each lake.
    For example, each column in temperature is BasinSuperior, BasinErie, WaterMichigan, etc.
    Args:
        path: Path of the file to read in

    Returns:
        An xarray.DataArray with dimensions 1 and 2 of Date and lake (respectively) followed by any other
        dimensions, though generally this is "type", i.e. "Land", "Water", and "Basin".
    """
    reader_args = reader_args or {}
    df = pd.read_csv(path, **reader_args)

    # create a date index from two columns - year and month
    date_index = pd.Index(
        list(
            map(
                lambda date_args: dt.datetime(*date_args, 1),
                zip(df["year"], df["month"]),
            )
        ),
        name="Date",
    )

    # with the new date index, convert to a "long" format
    melted_data = (
        df.set_index(date_index)
        .drop(["year", "month"], axis=1)
        .reset_index()
        .melt(value_name="value", id_vars="Date")
    )

    # Current columns are BasinErie, for example: find these, split, and create this varaible as two variables
    new_cols = pd.DataFrame(
        map(lambda x: re.findall("[A-Z][^A-Z]*", x), melted_data["variable"]),
        index=melted_data.index,
        columns=["type", "lake"],
    )
    long_df = melted_data.merge(new_cols, left_index=True, right_index=True).drop(
        ["variable"], axis=1
    )

    # with the melted data in the DF (Date -> lake -> ...) convert to an Xarray
    # loop over the groups ("Basin", "Land", "Lake") and format to match the format of Type I files.
    lake_arrays = []
    grps = []
    for grp, arr in long_df.groupby("type"):
        grps.append(grp)
        pivot_long_df = arr.pivot(columns="lake", values="value", index="Date")

        # rename the columns to match the shortened names in all other arrays (maintaining order as well)
        array = (
            pivot_long_df.assign(
                mic_hur=pivot_long_df["Michigan"] + pivot_long_df["Huron"]
            )
            .drop(["Michigan", "Huron"], axis=1)
            .rename(name_remap, axis=1)
        )

        lake_x_array = xr.DataArray(
            array,
            coords={"Date": array.index, "lake": column_order},
            dims=["Date", "lake"],
            name=grp,
        )
        lake_arrays.append(lake_x_array)

    full_set = xr.concat(lake_arrays, pd.Index(grps, name="type"))
    return full_set


def read_cfs_file(path) -> xr.DataArray:

    input_csv = pd.read_csv(path)
    cfsrun = pd.to_datetime(input_csv["cfsrun"], format="%Y%m%d%H")
    forecast_date = pd.to_datetime(
        input_csv.pop("year").astype(str) + input_csv.pop("month").astype(str),
        format="%Y%m",
    )

    input_csv["cfsrun"] = cfsrun
    input_csv["forecast_date"] = forecast_date

    melted_vars = input_csv.melt(id_vars=["cfsrun", "forecast_date"])
    new_cols = pd.DataFrame(
        map(lambda x: re.findall("[A-Z][^A-Z]*", x), melted_vars.pop("variable")),
        index=melted_vars.index,
        columns=["type", "lake"],
    )

    melted_vars[["type", "lake"]] = new_cols
    melted_vars = melted_vars[
        ~melted_vars.set_index(
            ["cfsrun", "forecast_date", "type", "lake"]
        ).index.duplicated()
    ]

    # remove duplicated rows
    arrays = []
    melted_vars["forecast_step"] = (
        melted_vars["forecast_date"].dt.to_period("M")
        - melted_vars["cfsrun"].dt.to_period("M")
    ).apply(lambda x: x.n)

    types = []
    for lake_data_type, grp in melted_vars.groupby("type"):
        arr = xr.concat(
            [
                xr.DataArray(
                    df.pivot(columns="lake", index="cfsrun", values="value").rename(
                        name_remap, axis=1
                    )
                )
                for step, df in grp.groupby("forecast_step")
            ],
            pd.Index(range(10), name="forecast_step"),
        )
        arrays.append(arr)
        types.append(lake_data_type)
    forecast_vals = xr.concat(arrays, dim=pd.Index(types, name="type")).rename(
        {"cfsrun": "Date"}
    )

    # need to collapse michigan/huron measurements together into a single lake
    mich_hur = (
        forecast_vals.sel(lake=["mic", "hur"])
        .mean(dim="lake")
        .expand_dims(dim={"lake": ["mic_hur"]})
    )

    cur_forecasts = forecast_vals.sel(lake=["eri", "sup", "ont"])
    forecast_array = (
        xr.concat([mich_hur, cur_forecasts], dim="lake")
        .sel(lake=column_order)
        .transpose("Date", "lake", ...)
    )
    return forecast_array


class FileReader(object):

    def __init__(
        self,
        path,
        series_name=None,
        reader: callable = read_historical_files,
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
        self.metadata = metadata or {}
        self.path = path
        self.series_name = series_name

    def __call__(self) -> xr.DataArray:
        arr: xr.DataArray = self._reader(self.path).rename(self.series_name)
        return arr.assign_attrs(**self.metadata)


def expand_dims(fn, var_name="Thiessen"):
    """
    Decorator to take the output of a FileReader and add a new dimensions (type) to a DataArray. This allows merging
    with FileReaders that return a type ("Basin", "Water", "Land") with FileReaders that do not, i.e. only return
    Date and Lake.

    Args:
        fn: A filereader function
        var_name: The variable name of the newly "type" dimension.

    Returns:
        The DataArray from "fn" with a new "type" dimension appended on the right.

    """

    def inner(*args, **kwargs):
        return fn(*args, **kwargs).expand_dims(dim={"type": [var_name]}, axis=-1)

    return inner


# Map a series name to a reader function and filepath. The filepaths are dynamic but based on
# the source directory.
forecast_map = {
    "precip": FileReader(
        precip_cfs_path, reader=read_cfs_file, source="CFS", type="forecast"
    ),
    "evap": FileReader(
        evap_cfs_path, reader=read_cfs_file, source="CFS", type="forecast"
    ),
    "temp": FileReader(
        temp_cfs_path,
        reader=read_cfs_file,
        source="CFS",
        type="forecast",
    ),
}

input_map = {
    "rnbs": FileReader(rnbs_hist_path, source="glcc", series_name="rnbs_hist"),
    "precip": [
        expand_dims(
            FileReader(precip_hist_path, source="glcc", series_name="precip_hist")
        ),
        FileReader(
            precip_cfsr_path,
            reader=read_cfsr_files,
            source="CFSR",
            series_name="precip_reanalysis",
        ),
    ],
    "evap": [
        expand_dims(FileReader(evap_hist_path, source="glcc", series_name="evap_hist")),
        FileReader(
            evap_cfsr_path,
            reader=read_cfsr_files,
            source="CFSR",
            series_name="evap_reanalysis",
        ),
    ],
    "runoff": FileReader(
        runoff_hist_path,
        reader=partial(
            read_historical_files,
            reader_args={"date_format": "%Y%m", "index_col": "Date"},
        ),
    ),
    "water_level": FileReader(
        water_level_hist_path, source="glcc", series_name="water_level"
    ),
    "temp": FileReader(
        temp_cfsr_path,
        reader=read_cfsr_files,
        source="CFSR",
        units="K",
        series_name="temp",
    ),
    "lhfx": FileReader(
        lhfx_cfsr_path,
        reader=read_cfsr_files,
        source="CFSR",
        units="K",
        series_name="lhfx",
    ),
}


def load_data(series: Union[str, List[str]], data_type="inputs"):
    """
    Load a data series based on name. Requires that raw files be available in the DATA_DIR from constants.py
    Args:
        series: the name of the series. Valid names include "rnbs", "precip", "evap", "runoff", "water_level"
        data_type: Type of values to get. Options include "inputs" and "forecasts".
    Returns:
        An xarray DataArray containing each series OR a pandas dataframe if only one series is requested.

    """

    assert data_type in ["inputs", "forecasts"]
    series_mapping = input_map if data_type == "inputs" else forecast_map

    # If a list of series is passed in, recursively call the loading function
    if isinstance(series, List):
        return xr.merge(
            [load_data(s, data_type=data_type).rename(s) for s in series]
        ).transpose("Date", "lake", ...)
    else:
        read_fn = series_mapping[series]
        if isinstance(read_fn, list):
            inputs = [read_fn() for read_fn in series_mapping[series]]
            return xr.concat(inputs, dim="type").rename(series)
        else:
            return read_fn().rename(series)
