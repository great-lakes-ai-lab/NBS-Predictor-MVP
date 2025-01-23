import os

import pytest

from src.data_loading.data_loading import load_data
from src.utils import create_rnbs_snapshot


@pytest.fixture
def lake_data():
    data = (
        load_data(["rnbs", "runoff", "precip", "evap"])
        .sel(type="Thiessen")
        .drop("type")
    )
    return data.to_array().transpose("Date", "lake", ...)


@pytest.fixture
def snapshot(lake_data):

    data_subset = lake_data.dropna("Date")
    snapshot = create_rnbs_snapshot(
        data_subset.sel(variable="rnbs"),
        split_date="2000-01-01",
        sequential_validation=True,
        validation_steps=12,
        covariates=data_subset.sel(variable=["runoff", "precip", "evap"]),
    )
    return snapshot


# skip_tests = os.environ.get("SKIP_FITS", "true").lower() == "true"
skip_tests = False
