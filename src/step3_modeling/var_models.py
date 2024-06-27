import calendar
from functools import reduce
from typing import Union

import numpy as np
import numpyro
import xarray as xr
import jax
from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro.contrib.control_flow import scan
from numpyro.infer.reparam import LocScaleReparam
from statsmodels.tsa.api import VAR as StatsVAR

from src.step3_modeling.modeling import ModelBase, NumpyroModel
from src.step4_postprocessing.postprocessing import output_forecast_results
from src.utils import lag_array, flatten_array

__all__ = [
    # Classes
    "VAR",
    "StatsModelVAR",
]


class VAR(NumpyroModel):

    @property
    def name(self):
        return "VAR"

    def __init__(
        self,
        lags: Union[None, dict] = None,
        num_chains=4,
        num_samples=1000,
        num_warmup=1000,
        lakes=("sup", "mic_hur", "eri", "ont"),
    ):
        super().__init__()
        if lags is None:
            self.lags = {"y": 3, "precip_hist": 6}
        else:
            self.lags = lags
        self.num_chains = num_chains
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.lakes = list(lakes)
        self.predictive_fn = None  # updated during model fitting
        self.trace = None
        self.mcmc = None

    @property
    def coords(self):
        return {
            "series": self.lakes,
            "lakes": self.lakes,
            "month": list(calendar.month_abbr)[1:],
            **{f"{k}_lags": list(range(1, self.lags[k] + 1)) for k in self.lags.keys()},
        }

    @property
    def dims(self):
        return {
            "intercept": ["month", "lakes"],
            "sigma": ["lakes"],
            "corr": ["series", "lakes"],
            "theta": ["lakes"],
            **{
                f"{k}_alpha": ["series", "lakes", f"{k}_lags"] for k in self.lags.keys()
            },
        }

    def load(cls, path):
        pass

    def save(self, path):
        pass

    @staticmethod
    @numpyro.handlers.reparam(config={"intercept": LocScaleReparam(0)})
    def model(y, y_index, lags, covariates, future=0):
        """
        Autoregressive process.
        Args:
            y: the time series to fit
            covariates: An XArray of covariates for use in the model. Leading index should be date, second index should be lake, and the third index is the actual covariate values.
            lags: A dictionary indicating which covariates have which lags
            future: How many periods to run into the future.

        Returns:
            None - samples

        """
        global_mu = numpyro.sample("global_mu", dist.Normal(0, 1))
        nu = numpyro.sample("nu", dist.HalfCauchy(2.0))

        ar_lag = lags.get("y")
        max_lag = reduce(max, lags.values())

        lagged_covars = [
            lag_array(jnp.array(covariates.sel(variable=covar)), np.arange(0, lag))[
                max_lag:
            ]
            for covar, lag in lags.items()
            if covar != "y"
        ]

        covar_alphas = [
            numpyro.sample(
                f"{cov}_alpha",
                dist.Normal(0, 0.2),
                sample_shape=(4, 4, lag),  # lag at 0
            )
            for cov, lag in lags.items()
        ]

        theta = numpyro.sample("theta", dist.HalfNormal(5), sample_shape=(4,))

        with numpyro.plate("lakes", size=4):
            with numpyro.plate("months", size=12):
                intercept = numpyro.sample("intercept", dist.Laplace(global_mu, 1))

        # t_nu = numpyro.sample("t_nu", dist.HalfNormal(10))
        l_omega = numpyro.sample("corr", dist.LKJCholesky(4, concentration=0.5))
        sigma = jnp.sqrt(theta)
        L_Omega = sigma[..., None] * l_omega

        def transition_fn(carry, covars):
            prev_y = carry
            month_t = covars[0]

            lagged_series = [prev_y, *[df.T for df in covars[1:]]]

            m = jnp.zeros((4,))

            # Loop over coefficients for Precip and AR
            for i in range(len(lags.items())):
                alphas = covar_alphas[i]
                dataset = lagged_series[i]
                for j in jnp.arange(alphas.shape[-1]):
                    m += jnp.matmul(alphas[:, :, j], dataset[j, :])

            m_t = intercept[month_t, :] + m
            y_t = numpyro.sample(
                "y", dist.MultivariateStudentT(df=nu, loc=m_t, scale_tril=L_Omega)
            )

            if ar_lag > 1:
                new_vals = jnp.append(prev_y[1:], y_t.reshape(1, -1), axis=0)
            else:
                new_vals = y_t.reshape(1, -1)
            return new_vals, y_t

        prev = y[:max_lag][-ar_lag:]
        months = jnp.array(y_index.month - 1)
        # need to subtract one because indexing starts at 0.
        initial_values = prev

        covars = (months[max_lag:], *lagged_covars)

        if future > 0:
            y_fit = y[:-future]
        else:
            y_fit = y

        with numpyro.handlers.condition(data={"y": y_fit[max_lag:]}):
            _, ys = scan(transition_fn, initial_values, covars)

        if future > 0:
            numpyro.deterministic("y_forecast", ys[-future:])


class StatsModelVAR(ModelBase):

    def __init__(
        self, lag_order=None, lag_selection_criterion="bic", lag_selection_order=18
    ):
        self.lag_order = lag_order
        self.lag_selection_criterion = lag_selection_criterion
        self.selection_lag_order = lag_selection_order
        self.model = None

    def fit(self, X, y, *args, **kwargs):
        model = StatsVAR(
            endog=np.array(y),
            exog=np.array(X),
        )

        if self.lag_order is None:
            order_value = model.select_order(self.selection_lag_order).aic
            self.lag_order = order_value

        self.model = model.fit(self.lag_order)

    def predict(self, X, y, forecast_steps=12, *args, **kwargs) -> xr.DataArray:
        exog_past, exog_future = X[:-forecast_steps], X[-forecast_steps:]
        y_past = y.loc[exog_past.indexes["Date"]]

        forecast_dates = exog_future.indexes["Date"]
        mean, lower, upper = self.model.forecast_interval(
            y_past, steps=forecast_steps, exog_future=exog_future
        )
        sd = (upper - mean) / 1.96

        # order MUST be mean, lower, upper, sd
        arrays = np.stack([mean, lower, upper, sd], axis=-1)
        results = output_forecast_results(
            arrays, forecast_labels=forecast_dates, lakes=y_past.indexes["lake"]
        )

        return results

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass


class NARX(NumpyroModel):

    def __init__(self, lags=None, num_chains=4, num_samples=1000, num_warmup=1000):
        super().__init__(lags, num_chains, num_samples, num_warmup)
        self.lags = lags or {"y": 3, "evap": 2, "precip": 2}

    @property
    def is_fitted(self):
        return super().is_fitted

    @property
    def coords(self):
        pass

    @property
    def dims(self):
        pass

    @staticmethod
    def model(y, y_index, lags, covariates, future=0):
        """
        Autoregressive process.
        Args:
            y: the time series to fit
            covariates: An XArray of covariates for use in the model. Leading index should be date, second index should be lake, and the third index is the actual covariate values.
            lags: A dictionary indicating which covariates have which lags
            future: How many periods to run into the future.

        Returns:
            None - samples

        """
        nu = numpyro.sample("nu", dist.HalfCauchy(2.0))

        ar_lag = lags.get("y")
        max_lag = reduce(max, lags.values())

        theta = numpyro.sample("theta", dist.HalfNormal(5), sample_shape=(4,))

        l_omega = numpyro.sample("corr", dist.LKJCholesky(4, concentration=0.5))
        sigma = jnp.sqrt(theta)
        L_Omega = sigma[..., None] * l_omega

        input_dim = reduce(lambda a, x: a + 4 * x, lags.values(), 0)
        h1 = 50
        output_dim = 4

        # first layer of the neural network
        w1 = numpyro.sample(
            "w1", dist.Normal(jnp.zeros((input_dim, h1)), jnp.ones((input_dim, h1)))
        )
        b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(h1), jnp.ones(h1)))

        w2 = numpyro.sample(
            "w2",
            dist.Normal(jnp.zeros((h1, output_dim)), jnp.ones((h1, output_dim))),
        )
        b2 = numpyro.sample(
            "b2", dist.Normal(jnp.zeros(output_dim), jnp.ones(output_dim))
        )

        lagged_covars = [
            flatten_array(
                lag_array(jnp.array(covariates.sel(variable=covar)), np.arange(0, lag))[
                    max_lag:
                ]
            )
            for covar, lag in lags.items()
            if covar != "y"
        ]
        covars = jnp.concatenate(lagged_covars, axis=-1)

        def transition_fn(carry, covars):
            prev_y = carry

            input_vals = jnp.concatenate([carry.reshape(-1), covars], axis=-1)
            z1 = jax.nn.relu(jnp.matmul(input_vals, w1) + b1)
            z2 = jnp.matmul(z1, w2) + b2

            y_t = numpyro.sample(
                "y", dist.MultivariateStudentT(df=nu, loc=z2, scale_tril=L_Omega)
            )

            if ar_lag > 1:
                new_vals = jnp.append(prev_y[1:], y_t.reshape(1, -1), axis=0)
            else:
                new_vals = y_t.reshape(1, -1)
            return new_vals, y_t

        initial_values = jnp.array(y[:max_lag][-ar_lag:])

        if future > 0:
            y_fit = jnp.array(y[:-future])
        else:
            y_fit = jnp.array(y)

        with numpyro.handlers.condition(data={"y": y_fit[max_lag:]}):
            _, ys = scan(transition_fn, initial_values, covars)

        if future > 0:
            numpyro.deterministic("y_forecast", ys[-future:])
