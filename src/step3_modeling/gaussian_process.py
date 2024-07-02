from abc import ABC
from functools import reduce

import numpy as np
import numpyro
import xarray as xr
from jax import jit, vmap
from jax import numpy as jnp
from numpyro import distributions as dist
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from src.constants import lake_order
from src.step3_modeling.modeling import ModelBase, NumpyroModel
from src.step4_postprocessing import output_forecast_results
from src.utils import lag_array, flatten_array

__all__ = ["SklearnGPModel"]


def dist_fn(x, y):
    return jnp.sum((x - y) ** 2)


def rbf_kernel(X, Z, var, length, noise, jitter=1.0e-6, include_noise=True):

    # broadcast so that each element in X is subtracted from the entirety of Z
    distance_calc = jit(vmap(vmap(dist_fn, in_axes=(None, 0)), in_axes=(0, None)))
    dist_mat = distance_calc(X, Z)
    k = var * jnp.exp(-(dist_mat**2) / (2 * length**2))
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k


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
        **kwargs,
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


class NumpyroGPModel(NumpyroModel):

    # TODO: Lag variables need to be implemented with GP

    def __init__(self, lags=None, num_chains=4, num_samples=1000, num_warmup=1000):
        super().__init__(lags, num_chains, num_samples, num_warmup)
        self.lags = lags or {"y": 3}
        self.train_x = None
        self.train_y = None

    @property
    def coords(self):
        return {"lake": lake_order}

    @property
    def dims(self):
        return {
            "length": ["lake"],
        }

    def fit(self, X: xr.DataArray, y: xr.DataArray, rng_key=None, *args, **kwargs):
        self.train_x = X
        self.train_y = y
        super().fit(X, y, rng_key, *args, **kwargs)

    def predict(self, X, y=None, rng_key=None, forecast_steps=12, *args, **kwargs):
        y_index = y.indexes["Date"]
        if rng_key is None:
            rng_key = self.get_rng_key()

        # Using future, chop the last `num_steps_forward` values off and treat them as unknown. This allows
        # the covariates to align with the test set
        forecast_marginal = self.predictive_fn(
            rng_key,
            y=jnp.array(y),
            y_index=y_index,
            covariates=X,
            lags=self.lags,
            future=forecast_steps,
        )["y_forecast"]

        mean = jnp.mean(forecast_marginal, axis=0)
        std = jnp.std(forecast_marginal, axis=0)
        low, high = hpdi(forecast_marginal)

        forecasts = jnp.stack([mean, low, high, std], axis=2)

        results = output_forecast_results(forecasts, y_index[-forecast_steps:])
        return results

        return super().predict(X, y, rng_key, forecast_steps, *args, **kwargs)

    @staticmethod
    def model(y, y_index, lags, covariates, future=0):

        # TODO: broken

        # set up covars and X values in one; make four distinct gaussian processes, but they
        # each use the features of the other lakes as well as their own
        ar_lag = lags.get("y")
        max_lag = reduce(max, lags.values())

        if future > 0:
            fit_y = y[:-future]
        else:
            fit_y = y

        # To set up previous values of Y, instead use a numeric index

        lagged_covars = [
            x[max_lag:]
            for x in lag_array(
                covariates, lags={k: v for k, v in lags.items() if k != "y"}
            )
        ]
        lagged_y = lag_array(fit_y, range(1, ar_lag + 1))[max_lag:]
        flattened_y = flatten_array(lagged_y)

        input_vars = jnp.concatenate([*lagged_covars, flattened_y])

        with numpyro.plate("lake", size=fit_y.shape[1]):
            length = numpyro.sample("length", dist.HalfNormal(6))
            var = numpyro.sample("var", dist.HalfNormal(4))
            noise = numpyro.sample("noise", dist.HalfNormal(2))

        obs = []
        with numpyro.handlers.condition(data={"y": fit_y[max_lag:]}):
            for j in np.arange(fit_y.shape[1]):
                covar = rbf_kernel(
                    input_vars,
                    input_vars,
                    length[j],
                    var[j],
                    noise[j],
                    include_noise=True,
                )

                obs.append(
                    numpyro.sample(
                        f"lake_{j}",
                        dist.MultivariateNormal(0, covariance_matrix=covar),
                        obs=fit_y[max_lag:, j],
                    )
                )

        y = numpyro.deterministic("y", all_fit_vals)

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass
