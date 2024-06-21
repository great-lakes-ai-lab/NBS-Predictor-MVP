from functools import reduce

import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

from src.step3_modeling.modeling import ModelBase
from src.step4_postprocessing import output_forecast_results
from src.utils import lag_array, flatten_array

__all__ = ["SklearnGPModel"]


class SklearnGPModel(ModelBase, ABC):
    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass

    def __init__(self, kernel, *args, **kwargs):
        """
        Args:
            kernel:
        """
        super().__init__()
        self.model = GaussianProcessRegressor(kernel, *args, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        self.model.fit(X, y, *args, **kwargs)

    def predict(
        self,
        X: xr.DataArray,
        y: xr.DataArray = None,
        forecast_steps=12,
        alpha=0.05,
        *args,
        **kwargs
    ) -> xr.DataArray:
        mean, sd = self.model.predict(X, return_std=True)
        mean, sd = mean[-forecast_steps:], sd[-forecast_steps:]

        z_scores = norm.ppf([alpha / 2, 1 - alpha / 2])
        lower, upper = mean + z_scores[0] * sd, mean + z_scores[1] * sd

        outputs = np.stack([mean, lower, upper, sd], axis=-1)

        formatted_predictions = output_forecast_results(
            outputs, X.indexes["Date"][-forecast_steps:]
        )
        return formatted_predictions


class LaggedGPModel(ModelBase):

    def __init__(self, kernel=kernels.Matern(), lags=None, *args, **kwargs):
        super().__init__()
        self.model = GaussianProcessRegressor(kernel)
        self.lags = lags or {"y": 3}

    def fit(self, X, y, *args, **kwargs):
        max_lag = self.lags["y"]
        lagged_y = flatten_array(lag_array(y, lags=range(1, max_lag + 1)))

        fit_covars = xr.concat([X[max_lag:], lagged_y[max_lag:]], dim="variable")
        fit_y = y[max_lag:]

        self.model.fit(fit_covars, fit_y)

    def predict(self, X, y, forecast_steps=12, *args, **kwargs) -> xr.DataArray:
        max_lag = self.lags["y"]
        lagged_y = lag_array(y, lags=range(1, max_lag + 1))

        start_val = lagged_y[[-forecast_steps]].values
        covars = X[-forecast_steps:].values

        lake_preds = []
        for i in range(forecast_steps):
            xc = covars[i].reshape(1, -1)
            start_val_flat = flatten_array(start_val)

            rec = np.concatenate([xc, start_val_flat], axis=-1)
            mean, sd = self.model.predict(rec, return_std=True)

            lower, upper = mean - sd * 1.96, mean + sd * 1.96

            full_pred = np.stack([mean, lower, upper, sd], axis=-1)
            lake_preds.append(full_pred)

            start_val = np.concatenate(
                [mean.reshape(-1, 4, 1), start_val[:, :, :-1]], axis=-1
            )

        results = output_forecast_results(
            np.concatenate(lake_preds, axis=0),
            forecast_labels=X.indexes["Date"][-forecast_steps:],
        )
        return results

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass
