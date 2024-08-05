import datetime as dt
import re
from functools import partial
from typing import List, Union
from collections.abc import Iterable

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
cfs_dir = DATA_DIR / "CFS"

temp_cfs_path = cfs_dir.glob("*/CFS_TMP_Basin_Avgs.csv")
precip_cfs_path = cfs_dir.glob("*/CFS_APCP_Basin_Avgs.csv")
evap_cfs_path = cfs_dir.glob("*/CFS_EVAP_Basin_Avgs.csv")


# L2SWBM


# Only interact with the data through the load_data
__all__ = ["load_data", "input_map", "forecast_map"]

lake_order = ["sup", "mic_hur", "eri", "ont"]
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
    df = pd.read_csv(path, **reader_args)[lake_order]
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

    df = df.drop(["year", "month"], axis=1).set_index(date_index)
    new_columns = pd.MultiIndex.from_tuples(
        [tuple(re.findall("[A-Z][^A-Z]*", x)) for x in df.columns],
        names=["type", "lake"],
    )
    df.columns = new_columns
    df = df.rename(columns=name_remap, level=1)

    # with the new date index, convert to a "long" format
    output_array = (
        df.melt(value_name="value", ignore_index=False)
        .set_index(["lake", "type"], append=True)
        .to_xarray()
        .to_array()
        .squeeze()
        .drop("variable")
    )

    mic_hur = (
        output_array.sel(lake=["mic", "hur"])
        .sum(dim="lake")
        .expand_dims(dim={"lake": ["mic_hur"]})
    )

    # Current columns are BasinErie, for example: find these, split, and create this varaible as two variables
    # with the melted data in the DF (Date -> lake -> ...) convert to an Xarray
    # loop over the groups ("Basin", "Land", "Lake") and format to match the format of Type I files.
    cur_forecasts = output_array.sel(lake=["eri", "sup", "ont"])
    forecast_array = (
        xr.concat([mic_hur, cur_forecasts], dim="lake")
        .sel(lake=lake_order)
        .transpose("Date", "lake", "type")
    )
    return forecast_array


def read_cfs_file(path) -> xr.DataArray:
    """
    Reads a CFS (Climate Forecast System) CSV file and converts it into an Xarray DataArray.

    The CSV file is expected to contain columns for year, month, cfsrun, and various lake measurements.
    The function processes the data to create a multi-dimensional DataArray with dimensions for Date,
    months ahead, lake, and type.

    Args:
        path: The path to the CSV file to be read.

    Returns:
        An Xarray DataArray with dimensions Date, months_ahead, lake, and type.
    """

    input_csv = pd.read_csv(path)
    forecast_date = pd.to_datetime(
        input_csv.pop("year").astype(str) + input_csv.pop("month").astype(str),
        format="%Y%m",
    )

    input_csv = input_csv.assign(
        cfsrun=pd.to_datetime(input_csv["cfsrun"], format="%Y%m%d%H"),
        forecast_date=forecast_date,
    ).set_index(["cfsrun", "forecast_date"])

    new_columns = [tuple(re.findall("[A-Z][^A-Z]*", x)) for x in input_csv.columns]
    input_csv.columns = pd.MultiIndex.from_tuples(new_columns, names=["type", "lake"])
    input_csv = input_csv.rename(columns=name_remap)

    input_csv["months_ahead"] = (
        input_csv.sort_values(["cfsrun", "forecast_date"]).groupby("cfsrun").cumcount()
    )

    # convert from an explicit forecast date to a simple integer for how many months ahead we are predicting
    input_csv = input_csv.reset_index(level=1, drop=True).set_index(
        "months_ahead", append=True
    )
    input_csv.index.names = ["Date", "months_ahead"]

    output_array = (
        input_csv.melt(ignore_index=False)
        .set_index(["lake", "type"], append=True)
        .to_xarray()
        .to_array()
        .squeeze()
        .drop("variable")
    )

    # need to collapse michigan/huron measurements together into a single lake
    mich_hur = (
        output_array.sel(lake=["mic", "hur"])
        .mean(dim="lake")
        .expand_dims(dim={"lake": ["mic_hur"]})
    )

    cur_forecasts = output_array.sel(lake=["eri", "sup", "ont"])
    forecast_array = (
        xr.concat([mich_hur, cur_forecasts], dim="lake")
        .sel(lake=lake_order)
        .transpose("Date", "months_ahead", "lake", "type")
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

        # If a list of files is passed in, assume that we want to concatenate across dates
        # otherwise, just read in the file and return the Xarray
        # In both cases, assign the metadata to the Xarray
        if isinstance(self.path, Iterable):
            arrs = [self._reader(path).rename(self.series_name) for path in self.path]
            return (
                xr.concat(arrs, dim="Date")
                .assign_attrs(**self.metadata)
                .sortby("Date", ascending=True)
            )
        else:
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
