import datetime as dt

import pandas as pd
import pytest

from src.step2_preprocessing.preprocessing import XArrayStandardScaler
from src.utils import create_rnbs_snapshot, acf, lag_array


@pytest.mark.parametrize(
    "split_date",
    [1990, dt.date(2001, 10, 1), dt.datetime(2004, 9, 1)],
    ids=["year_int", "date", "datetime"],
)
def test_snapshot(lake_data, split_date):
    local_snapshot = create_rnbs_snapshot(
        lake_data.sel(lake="sup", variable="rnbs"),
        split_date=split_date,
        num_years_forward=1,
        sequential_validation=True,
    )
    assert local_snapshot.test_index.max() > local_snapshot.train_index.max()
    assert local_snapshot.test_index.min() == local_snapshot.train_index.min()


@pytest.mark.skip(reason="Deprecated")
def test_snapshot_scaling(lake_data):
    local_snapshot = create_rnbs_snapshot(
        lake_data.sel(variable="rnbs"),
        split_date=dt.datetime(1999, 12, 1),
        covariates=lake_data.sel(variable=["runoff", "precip"]),
    )
    x_scaler = XArrayStandardScaler()
    y_scaler = XArrayStandardScaler()

    old_max = local_snapshot.train_x.max()

    local_snapshot.apply_transformer(x_transformer=x_scaler, y_transformer=y_scaler)

    new_max = local_snapshot.train_x.max()

    assert old_max > new_max


def test_multi_lake_snapshot(lake_data):
    snapshot = create_rnbs_snapshot(
        rnbs_data=lake_data.sel(variable="rnbs"),
        covariates=lake_data.sel(variable=["precip", "runoff", "evap"]),
        split_date=2000,
    )
    assert snapshot


def test_snapshot_divides_covars(lake_data):
    # just use the other lakes for our "covariates"
    local_snapshot = create_rnbs_snapshot(
        rnbs_data=lake_data.sel(variable="rnbs"),
        split_date=dt.date(2005, 1, 1),
        covariates=lake_data.sel(variable="precip"),
    )
    assert local_snapshot.train_x.shape[1] == 4
    assert (
        local_snapshot.train_x.shape[0] == len(local_snapshot.train_y)
        and local_snapshot.train_x.shape[1] == 4
    )
    assert (
        local_snapshot.val_x.shape[0] == len(local_snapshot.val_y)
        and local_snapshot.val_x.shape[1] == 4
    )
    assert (
        local_snapshot.test_x.shape[0] == len(local_snapshot.test_y)
        and local_snapshot.test_x.shape[1] == 4
    )


@pytest.mark.parametrize("lag", [10, 20, 50])
def test_acf(lake_data, lag):
    rnbs_12 = (
        lake_data.sel(lake="sup", variable="rnbs").rolling(Date=12).sum().dropna("Date")
    )
    acf_values = acf(rnbs_12, max_lag=lag)

    assert isinstance(acf_values, pd.DataFrame)
    assert (acf_values["rho"].abs() <= 1).all()
    assert len(acf_values) == lag + 1


@pytest.mark.parametrize("lags", [(1, 2), (1, 5, 10)], ids=["1,2", "1,5,10"])
def test_lag_var(lake_data, lags):
    rnbs_vect = lake_data.sel(lake="sup", variable="rnbs")
    lag_return = lag_array(rnbs_vect, lags=lags)
    assert lag_return.shape == (len(rnbs_vect), len(lags))


def test_lag_array(lake_data):
    my_data = lake_data.sel(variable="rnbs")[:10]
    lagged_data = lag_array(my_data, lags=(1, 2, 3))

    assert lagged_data.shape == (10, 4, 3)
