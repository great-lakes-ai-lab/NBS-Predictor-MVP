import pytest

from src.step1_data_loading.data_loading import load_data
from src.utils import create_rnbs_snapshot


@pytest.fixture
def lake_data():
    return (
        load_data(["rnbs", "runoff", "precip", "evap"])
        .to_array()
        .transpose("Date", "lake", ...)
    )


@pytest.fixture
def snapshot(lake_data):

    data_subset = lake_data.dropna("Date")
    snapshot = create_rnbs_snapshot(
        data_subset.sel(variable="rnbs"),
        split_date="1980-01-01",
        sequential_validation=True,
        validation_proportion=0.05,
        covariates=data_subset.sel(variable=["runoff", "precip", "evap"]),
    )
    return snapshot
