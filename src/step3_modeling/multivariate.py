import calendar

import numpyro
from jax import numpy as jnp
from numpyro import distributions as dist

from step3_modeling.modeling import NumpyroModel

__all__ = [
    # Classes
    "LakeMVT",
]


class LakeMVT(NumpyroModel):

    @property
    def name(self) -> str:
        return "LakeMVT"

    @property
    def coords(self):
        return {
            "lakes": self.lakes,
            "lakes_cov": self.lakes,
            "month": list(calendar.month_abbr)[1:],
        }

    @property
    def dims(self):
        return {
            "intercept": ["month", "lakes"],
            "corr": ["lakes", "lakes_cov"],
            "sigma": ["lakes"],
        }

    @classmethod
    def load(cls, path):
        pass

    def save(self, path):
        pass

    @staticmethod
    def model(y, y_index, lags, covariates, future=0):
        global_bias = numpyro.sample("global_mu", dist.Normal(0, 3))

        theta = numpyro.sample("theta", dist.HalfNormal(5), sample_shape=(4,))

        with numpyro.plate("lakes", size=4):
            with numpyro.plate("months", size=12):
                intercept = numpyro.sample("intercept", dist.Laplace(global_bias, 3))

        l_omega = numpyro.sample("corr", dist.LKJCholesky(4, concentration=0.5))
        sigma = jnp.sqrt(theta)
        L_Omega = sigma[..., None] * l_omega

        t_nu = numpyro.sample("t_nu", dist.HalfNormal(10))

        # separate out the conditional from the forecasting. Wonky to match other forecasting methods
        months = (y_index.month - 1).values
        mu = intercept[months]
        N = y.shape[0]

        with numpyro.plate("obs", N):
            if future > 0:
                y_t = numpyro.sample(
                    "y",
                    dist.MultivariateStudentT(t_nu, loc=mu, scale_tril=L_Omega),
                )
            else:
                y_t = numpyro.sample(
                    "y",
                    dist.MultivariateStudentT(t_nu, loc=mu, scale_tril=L_Omega),
                    obs=y,
                )

        if future > 0:
            numpyro.deterministic("y_forecast", y_t[-future:])
