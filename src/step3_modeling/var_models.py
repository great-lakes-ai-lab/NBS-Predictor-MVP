import numpy as np
from typing import Union
from functools import reduce

import calendar
import numpyro
from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro.contrib.control_flow import scan

from numpyro.infer.reparam import LocScaleReparam

from src.step3_modeling.modeling import NumpyroModel
from src.utils import lag_vector

__all__ = [
    # Classes
    "VAR",
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
            self.lags = {"y": 3, "precip": 6}
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
    def model(y, y_index, lags, covariates=None, future=0):
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

        covar_alphas = [
            numpyro.sample(
                f"{cov}_alpha", dist.Laplace(0, 0.2), sample_shape=(4, 4, lag)
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

        ar_lag = lags.get("y")

        def transition_fn(carry, covars):
            prev_y = carry
            month_t = covars[0]

            lagged_series = [prev_y, *[df.T for df in covars[1:]]]

            m = jnp.zeros((4,))

            # Loop over coefficients for Precip and AR
            for i in range(len(lags.items())):
                alphas = covar_alphas[i]
                dataset = lagged_series[i]
                # if alphas.shape[-1] == 1:
                #     m += jnp.matmul(alphas, dataset)
                # else:
                # switch to reduce
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

        prev = y[:ar_lag]
        months = jnp.array(y_index.month - 1)
        # need to subtract one because indexing starts at 0.
        initial_values = prev

        max_lag = reduce(max, lags.values())

        lagged_covars = [
            lag_vector(
                jnp.array(covariates.sel(variable=covar)), np.arange(1, lag + 1)
            )[max_lag:]
            for covar, lag in lags.items()
            if covar != "y"
        ]

        covars = (months[max_lag:], *lagged_covars)

        if future > 0:
            y_fit = y[:-future]
        else:
            y_fit = y

        with numpyro.handlers.condition(data={"y": y_fit[max_lag:]}):
            _, ys = scan(transition_fn, initial_values, covars)

        if future > 0:
            numpyro.deterministic("y_forecast", ys[-future:])
