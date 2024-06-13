from abc import ABC

import numpy as np
import xarray as xr
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from src.step3_modeling.modeling import ModelBase
from src.step4_postprocessing import output_forecast_results

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
