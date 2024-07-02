import numpy as np
import xarray as xr
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

from src.step3_modeling.modeling import ModelBase
from src.step4_postprocessing.postprocessing import output_forecast_results

__all__ = [
    # Classes
    "DefaultEnsemble",
    "BaggedXArrayRegressor",
]


class DefaultEnsemble(ModelBase):

    @property
    def name(self):
        return "DefaultEnsemble"

    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha
        self.month_df = None
        self.lakes = ["sup", "mic_hur", "eri", "ont"]

    def fit(self, X, y, *args, **kwargs):
        quantiles = (
            y.groupby("Date.month")
            .quantile(
                q=[self.alpha / 2, 1 - self.alpha / 2],
            )
            .assign_coords({"quantile": ["lower", "upper"]})
            .rename({"quantile": "variable"})
        )
        means = y.groupby("Date.month").mean().expand_dims(variable=["mean"])
        std = y.groupby("Date.month").std().expand_dims(variable=["std"])

        self.month_df = xr.concat([means, quantiles, std], dim="variable")
        return self

    def predict(self, X, y=None, forecast_steps=12, *args, **kwargs) -> xr.DataArray:
        # convert to DatetimeIndex type just in case for .month syntax

        forecast_index = X.indexes["Date"][-forecast_steps:]
        forecasts = (
            self.month_df.sel(month=forecast_index.month)
            .rename(month="Date")
            .assign_coords(Date=forecast_index)
            .transpose("Date", "lake", "variable")
        )

        formatted_output = output_forecast_results(forecasts, forecast_index)
        return formatted_output

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass


class BaggedXArrayRegressor(ModelBase):
    """
    Take any given Sklearn regressor model and wrap it into a BaggedRegressor to
    get a simple empirical estimate of standard deviation and quantile of predictions
    """

    def __init__(self, sklearn_regressor=None, **bagging_kwargs):
        super().__init__()
        self.regressor = sklearn_regressor or LinearRegression()
        bagging_kwargs = bagging_kwargs or {"n_estimators": 250, "n_jobs": -1}
        self.model = BaggingRegressor(estimator=self.regressor, **bagging_kwargs)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)

    def predict(
        self, X, y, forecast_steps=12, alpha=0.05, *args, **kwargs
    ) -> xr.DataArray:
        predictions = np.stack([m.predict(X) for m in self.model.estimators_], axis=-1)

        output_array = np.stack(
            [
                predictions.mean(axis=-1),
                *np.quantile(predictions, q=[alpha / 2, 1 - alpha / 2], axis=-1),
                predictions.std(axis=-1),
            ],
            axis=-1,
        )[-forecast_steps:]

        return output_forecast_results(
            output_array, forecast_labels=X.indexes["Date"][-forecast_steps:]
        )

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass
