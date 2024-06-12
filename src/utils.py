import datetime as dt
import logging
from functools import partial, reduce
from typing import Union

import numpy as np
from jax import numpy as jnp
import pandas as pd
import torch
import xarray as xr
from dateutil.relativedelta import relativedelta
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def convert_year_to_date(year: Union[str, int, dt.date]):
    if isinstance(year, str):
        try:
            return dt.datetime.strptime(year, "%Y-%m-%d")
        except ValueError:
            return dt.datetime(int(year), 1, 1)
    elif isinstance(year, int):
        return dt.datetime(int(year), 1, 1)
    else:
        return year


def percentile(q):
    return lambda x: x.quantile(q)


class create_rnbs_snapshot(object):
    def __init__(
        self,
        rnbs_data: Union[pd.Series, xr.DataArray],
        split_date: Union[int, str, dt.date, dt.datetime],
        num_years_forward=0,
        covariates: Union[NDArray, torch.Tensor, xr.DataArray, None] = None,
        sequential_validation=True,
        validation_proportion=0.5,
        validation_steps=12,
    ):
        """

        :param rnbs_data: Residual net basin supply - expecting an indexed series
        :param split_date: The date on which to split between training/validation (set 1) and test set (set 2). Note that this is inclusive.
        :param num_years_forward: The number of years after the split date to include in the test set.
        :param covariates: X covariates to fit on the model. If these are not provided, then a sequential integer array is created.
        :param sequential_validation: Boolean indicating whether the training set should be split seequentially into training and validation divisions or if it should be randomized.
        :param validation_proportion: The proportion of the training set to hold out for validation.
        """
        self.split_date = split_date
        if isinstance(split_date, int):
            # if just a year is given, set it to January 1
            split_date = dt.datetime(split_date, 1, 1)
        elif isinstance(split_date, dt.date):
            split_date = dt.datetime(split_date.year, split_date.month, split_date.day)
        elif isinstance(split_date, str):
            split_date = convert_year_to_date(split_date)
        self.prediction_year = split_date

        months_forward = 12 * num_years_forward
        final_date = split_date + relativedelta(months=months_forward)
        try:
            y_index = rnbs_data.index
        except AttributeError:  # xarray instead of dataframe
            y_index = rnbs_data.indexes["Date"]

        y_subset = rnbs_data[(y_index <= final_date)]

        if covariates is not None:
            assert covariates.shape[0] == len(rnbs_data)
            self.covariates = covariates[y_index <= final_date]
        else:
            covariates = np.arange(0, y_subset.shape[0])
            self.covariates = xr.DataArray(
                covariates,
                dims=["Date"],
                coords={"Date": y_index[(y_index <= final_date)]},
            )

        self.test_index = y_index[y_index <= final_date]

        train_y = y_subset.loc[:split_date]
        self.test_y = y_subset.loc[:final_date]

        train_x = self.covariates.loc[:split_date]
        train_index = train_y.indexes["Date"]
        train_n = len(train_x)

        # divide the training set into a validation and training set
        if sequential_validation and validation_steps > 0:
            self.val_index = (
                train_index[-validation_steps:]
                if validation_steps > 0
                else train_index[:0]
            )
        else:
            self.val_index = np.sort(
                np.random.permutation(train_x.indexes["Date"])[
                    : int(validation_proportion * train_n)
                ]
            )
        self.val_x = self.covariates.loc[self.val_index]
        self.val_y = train_y.loc[self.val_index]

        self.train_index = train_index.difference(self.val_index)

        # subset the training set to NOT include the validation set
        self.train_x, self.train_y = (
            train_x.loc[self.train_index],
            train_y.loc[self.train_index],
        )

        self.test_x = self.covariates.loc[self.test_index]

        self._x_transformer = None
        self._y_transformer = None
        self._is_transformed = False

    def __iter__(self):
        for item in [self.train_x, self.train_y, self.test_x, self.test_y, self.index]:
            yield item

    @property
    def is_transformed(self):
        return self._is_transformed

    @property
    def x_transformer(self):
        return self._x_transformer

    @property
    def y_transformer(self):
        return self._y_transformer

    def apply_transformer(self, x_transformer=None, y_transformer=None):
        if x_transformer:
            self.train_x = x_transformer.fit_transform(self.train_x)
            self.val_x = x_transformer.transform(self.val_x)
            self.test_x = x_transformer.transform(self.test_x)

        if y_transformer:
            self.train_y = y_transformer.fit_transform(self.train_y)
            self.val_y = y_transformer.transform(self.val_y)
            self.test_y = y_transformer.transform(self.test_y)

        self._x_transformer = x_transformer
        self._y_transformer = y_transformer

        return self

    def cuda(self):
        self.train_x = self.train_x.cuda()
        self.train_y = self.train_y.cuda()
        self.val_x = self.val_x.cuda()
        self.val_y = self.val_y.cuda()
        self.test_x = self.test_x.cuda()
        self.test_y = self.test_y.cuda()
        return self

    def cpu(self):
        self.train_x = self.train_x.cpu()
        self.train_y = self.train_y.cpu()
        self.val_x = self.val_x.cpu()
        self.val_y = self.val_y.cpu()
        self.test_x = self.test_x.cpu()
        self.test_y = self.test_y.cpu()
        return self


def filter_model_results(lake_df: pd.DataFrame):
    # get one month, 3 month, 6 month, and 12 month prediction
    return (
        lake_df[lake_df["group"] == "test"]
        .iloc[[0, 2, 5, -1]]
        .assign(months_ahead=[1, 3, 6, 12])
    )


def acf(x, max_lag=20):
    return pd.DataFrame(
        [
            (i, np.corrcoef(x[:-i], x[i:])[0, 1] if i > 0 else 1)
            for i in range(max_lag + 1)
        ],
        columns=["index", "rho"],
    ).set_index("index")


def lag_series(x: pd.Series, lags=(1)):
    """Create a lagged vector from a given series. Currently only works on named series.
    Args:
        x (pd.Series): A pandas series
        lags (tuple, optional): The number of lags to return. Defaults to (1).
    """

    def lag_closure(a, lag, j=None):
        if not j:
            return pd.concat([a, x.shift(lag).rename(f"l_{lag}")], axis=1)
        else:
            return pd.concat([a, x[j].shift(lag).rename(f"{j}_{lag}")], axis=1)

    if len(x.shape) > 1:
        lagged_var = pd.concat(
            [
                reduce(partial(lag_closure, j=col_name), lags, x).iloc[:, -len(lags) :]
                for col_name in x.columns
            ],
            axis=1,
        )
    else:
        lagged_var = reduce(lag_closure, lags, x)
    return lagged_var


def lag_vector(x: Union[pd.Series, np.array], lags=(1)):

    fn_lookup = {pd.Series: lag_series, np.ndarray: lag_array, xr.DataArray: lag_xarray}
    return fn_lookup.get(x.__class__, lag_array)(x, lags)


def lag_array(x: jnp.array, lags=(1)):
    lag_vals = [
        jnp.concatenate(
            [jnp.repeat(np.nan, 4).reshape(-1, 4).repeat(i, axis=0), x[:-i]], axis=0
        )
        for i in lags
    ]
    values = jnp.stack(lag_vals, axis=len(x.shape))
    return values


def lag_xarray(x: xr.DataArray, lags=(1)):
    lag_vect = [x.shift(Date=j).rename(f"lag_{j}") for j in lags]

    lagged_data = xr.DataArray(
        lag_vect,
        dims=["lags", *x.dims],
        coords={"lags": list(lags), **x.coords},
    ).transpose("Date", ...)

    return lagged_data


def setup_logger(lake, year):
    def lake_filter(record: logging.LogRecord):
        record.lake = lake
        record.date = year.strftime("%Y-%m-%d")
        return record

    lake_logger = logging.getLogger("models")
    lake_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s -- Lake: %(lake)s, Date: %(date)s -- %(message)s"
    )
    handler.setFormatter(formatter)
    handler.addFilter(lake_filter)
    lake_logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    return logger


class ContextFilter(logging.Filter):
    def __init__(self, filter_name, extra):
        super(ContextFilter, self).__init__(filter_name)
        self.lake = extra

    def filter(self, record):
        record.lake = self.lake
        return True
