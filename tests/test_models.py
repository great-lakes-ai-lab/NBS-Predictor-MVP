import numpy as np
import pytest
import xarray as xr
from sklearn.gaussian_process import kernels as k
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, SplineTransformer

from src.preprocessing.preprocessing import XArrayAdapter, XArrayStandardScaler
from src.modeling.ensemble import (
    BaggedXArrayRegressor,
    DefaultEnsemble,
    RandomForest,
    BoostedRegressor,
)
from src.modeling.gaussian_process import (
    LaggedSklearnGP,
    MultitaskGP,
    SklearnGPModel,
)
from src.modeling.lm import LinearModel
from src.modeling.metrics import summarize
from src.modeling.modeling import ModelBase
from src.modeling.multivariate import LakeMVT
from src.modeling.nn import BayesNN
from src.modeling.var_models import NARX, VAR
from src.postprocessing.postprocessing import output_forecast_results
from src.utils import flatten_array
from tests.conftest import skip_tests

modelList = {
    "DefaultEnsemble": DefaultEnsemble(),
    "MVN": LakeMVT(num_warmup=0, num_samples=3, num_chains=1),
    "VAR": VAR(num_warmup=0, num_samples=3, num_chains=1, lags={"y": 2}),
    "NARX": NARX(num_warmup=0, num_samples=3, num_chains=1, lags={"y": 2, "precip": 2}),
    "VARX": VAR(
        num_warmup=0,
        num_samples=3,
        num_chains=1,
        lags={"y": 1, "precip": 1, "evap": 1},
    ),
    "GP": Pipeline(
        steps=[
            ("scale", XArrayStandardScaler()),
            ("flatten", FunctionTransformer(flatten_array)),
            ("gp", SklearnGPModel(kernel=1.0 * k.Matern())),
        ]
    ),
    "SklearnRegressor": Pipeline(
        [
            ("flatten", FunctionTransformer(flatten_array)),
            ("nnet", BaggedXArrayRegressor()),
        ]
    ),
    "BayesNN": Pipeline(
        [
            ("flatten", FunctionTransformer(flatten_array)),
            ("nnet", BayesNN(num_warmup=0, num_samples=3, num_chains=1)),
        ]
    ),
    "LaggedSklearnGP": Pipeline(
        [("flatten", FunctionTransformer(flatten_array)), ("gp", LaggedSklearnGP())]
    ),
    "MultiTaskGP": Pipeline(
        [("flatten", FunctionTransformer(flatten_array)), ("gp", MultitaskGP())]
    ),
    "LinearModel": Pipeline(
        [("flatten", FunctionTransformer(flatten_array)), ("lm", LinearModel())]
    ),
    "SplineModel": Pipeline(
        [
            ("flatten", FunctionTransformer(flatten_array)),
            ("splines", XArrayAdapter(SplineTransformer(knots="quantile"))),
            ("lm", LinearModel()),
        ]
    ),
    "RandomForest": Pipeline(
        [
            ("flatten", FunctionTransformer(flatten_array)),
            ("rf", RandomForest()),
        ]
    ),
    "BoostedTrees": Pipeline(
        [
            ("flatten", FunctionTransformer(flatten_array)),
            ("gb", BoostedRegressor()),
        ]
    ),
}


@pytest.fixture
def preprocessor():
    """
    Using simple scaling, test that the models all work with a scikit-learn interface and pipeline objects
    Returns:

    """
    return Pipeline([("scaler", XArrayStandardScaler())])


@pytest.mark.skipif(skip_tests, reason="Skip kernel fits")
@pytest.mark.parametrize("model", modelList.values(), ids=modelList.keys())
def test_model_fit(model: ModelBase, snapshot, preprocessor):
    y_scaler = XArrayStandardScaler()
    train_y = y_scaler.fit_transform(snapshot.train_y)
    test_y = y_scaler.transform(snapshot.test_y)
    full_pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    full_pipeline.fit(y=train_y, X=snapshot.train_x)

    results = full_pipeline.predict(X=snapshot.test_x, y=test_y, forecast_steps=24)

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
    forecast_array = output_forecast_results(forecasts, new_labels)

    assert isinstance(forecast_array, xr.DataArray)
    assert forecast_array.dims == (new_labels.name, "lake", "value")


@pytest.mark.skip("Baseline may not apply when changing data")
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
    assert summary_result["rmse"].max() < 70 and summary_result["rmse"].min() > 40
