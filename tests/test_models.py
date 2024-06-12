import numpy as np
import pytest
import xarray as xr

from src.step3_modeling.ensemble import DefaultEnsemble
from src.step3_modeling.metrics import summarize
from src.step3_modeling.modeling import ModelBase
from src.step3_modeling.multivariate import LakeMVT
from src.step3_modeling.var_models import VAR

from tests.conftest import skip_tests

modelList = {
    "DefaultEnsemble": DefaultEnsemble(),
    "MVN": LakeMVT(num_warmup=0, num_samples=3, num_chains=1),
    "VAR": VAR(num_warmup=0, num_samples=3, num_chains=1, lags={"y": 2}),
    "VARCovars": VAR(
        num_warmup=0, num_samples=3, num_chains=1, lags={"y": 2, "precip": 2}
    ),
    "VARCovarsLag1": VAR(
        num_warmup=0, num_samples=3, num_chains=1, lags={"y": 1, "precip": 1, "evap": 1}
    ),
}


@pytest.mark.skipif(skip_tests, reason="Skip model fits")
@pytest.mark.parametrize("model", modelList.values(), ids=modelList.keys())
def test_model_fit(model: ModelBase, snapshot):
    model.fit(y=snapshot.train_y, X=snapshot.train_x)

    results = model.predict(X=snapshot.test_x, y=snapshot.test_y, forecast_steps=24)

    # dim should be forecast length (24), lakes (4) and output_values (4),
    # which are mean, lower, upper, and std
    assert results.shape == (24, 4, 4)
    assert isinstance(results, xr.DataArray)
    assert (snapshot.test_index[-24:] == results.indexes["Date"]).all()
    # ensure that this is random and not just conditioning on the last value
    assert (results[-1, :, 0] != snapshot.test_y[-1, :]).all()


def test_forecaster_output_format(snapshot):
    new_labels = snapshot.train_index[-12:]
    forecasts = np.random.uniform(size=12 * 4 * 4).reshape(12, 4, 4)
    forecast_array = ModelBase.output_forecast_results(forecasts, new_labels)

    assert isinstance(forecast_array, xr.DataArray)
    assert forecast_array.dims == (new_labels.name, "lake", "value")


def test_baseline(snapshot):
    model = DefaultEnsemble()
    model.fit(y=snapshot.train_y, X=snapshot.train_x)
    results = model.predict(X=snapshot.test_x, y=snapshot.test_y, forecast_steps=24)

    result_df = (
        xr.merge([results, snapshot.test_y.rename("true")])
        .dropna("Date")
        .to_dataframe(dim_order=["Date", "lake", "value"])
        .drop("variable", axis=1)
        .reset_index()
        .pivot(index=["Date", "lake", "true"], columns="value", values="forecasts")
        .reset_index(level=-1)
    )

    summary_result = result_df.groupby("lake").apply(summarize)
    assert summary_result["rmse"].max() < 65 and summary_result["rmse"].min() > 40
