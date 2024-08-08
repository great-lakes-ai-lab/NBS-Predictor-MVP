import gpytorch
import jax
import numpy as np
import numpyro
import torch
import xarray as xr
from abc import ABC
from jax import numpy as jnp
from numpyro import distributions as dist
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

from src.modeling.modeling import ModelBase, NumpyroModel
from src.postprocessing import output_forecast_results
from src.utils import flatten_array, lag_array

import logging

__all__ = ["SklearnGPModel", "LaggedSklearnGP", "MultitaskGP"]

logger = logging.getLogger(__name__)


# Kernel definitions are taken from https://www.cs.toronto.edu/~duvenaud/cookbook/
def rbf_kernel(X, Z, lengthscale, variance, noise=None, jitter=1e-6):
    # See https://num.pyro.ai/en/latest/examples/gp.html
    diffs = jnp.sqrt(((X[:, None] - Z) ** 2).sum(axis=-1))
    output = variance * jnp.exp(-0.5 * diffs**2 / lengthscale**2)

    if noise is not None:
        output += (noise + jitter) * jnp.eye(X.shape[0])
    return output


def matern_kernel(X, Z, lengthscale=1.0, nu=1.0):
    raise NotImplemented


class NumpyroLagGP(NumpyroModel):

    @property
    def coords(self):
        pass

    @property
    def dims(self):
        pass

    @staticmethod
    def model(y, y_index, lags, covariates, future=0):
        lengthscale = numpyro.sample("lengthscale", dist.Gamma(2, 1))
        noise = numpyro.sample("noise", dist.HalfNormal(1))
        variance = numpyro.sample("variance", dist.HalfNormal(1))

        y_lag = lags["y"]
        lagged_y = lag_array(y, range(1, y_lag + 1))

        if future > 0:
            fit_y, fit_covars, fit_lag_y = (
                y[y_lag:-future],
                covariates[y_lag:-future],
                lagged_y[y_lag:-future],
            )

            fit_covars = jnp.concatenate(
                [fit_covars, fit_lag_y], axis=1
            )  # make a new set of covariates
            test_y, test_x, initial = y[-future:], covariates[-future:], fit_y[-y_lag:]
        else:
            fit_y, fit_covars = y[y_lag:], covariates[y_lag:]
            test_y, test_x, initial = None, None, None

        k = rbf_kernel(fit_covars, fit_covars, lengthscale, variance, noise)

        train_estimates = numpyro.sample(
            "y",
            dist.MultivariateNormal(
                loc=jnp.zeros(covariates.shape[0]), covariance_matrix=k
            ),
            obs=fit_y,
        )

        # do GP prediction for a given set of hyperparameters. this makes use of the well-known
        # formula for Gaussian process predictions
        def transition(carry, covariates):

            X, Y, X_test = covariates

            new_covars = jnp.concatenate([X_test, carry], axis=1)

            k_pp = rbf_kernel(new_covars, new_covars, variance, lengthscale, noise)
            k_pX = rbf_kernel(new_covars, X, variance, lengthscale)
            k_XX = rbf_kernel(X, X, variance, lengthscale, noise)

            # since K_xx is symmetric positive-definite, we can use the more efficient and
            # stable Cholesky decomposition instead of matrix inversion
            K_xx_cho = jax.scipy.linalg.cho_factor(k_XX)
            K = k_pp - jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, k_pX.T))
            mean = jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, Y))

            # we return both the mean function and a sample from the posterior predictive for the
            # given set of hyperparameters
            return (mean, mean + sigma_noise)

        if future > 0:
            with numpyro.handlers.condition(data={"y": test_y}):
                _, ys = scan(transition_fn, test_y, covars)
            jax.lax.scan()

    @property
    def name(self):
        return "NumpyroGP"

    def __init__(self, *args, **kwargs):
        # See https://num.pyro.ai/en/latest/examples/gp.html - this example is adapted for an auto-regressive model
        super().__init__(*args, **kwargs)

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass


class SklearnGPModel(ModelBase, ABC):
    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass

    def __init__(self, kernel=1.0 * kernels.Matern(), *args, **kwargs):
        """
        Args:
            kernel:
        """
        super().__init__()
        self.model = GaussianProcessRegressor(kernel, *args, **kwargs)

    @property
    def name(self):
        return "SklearnGP"

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


class LaggedSklearnGP(ModelBase):

    def __init__(self, kernel=1.0 * kernels.Matern(), lags=None, *args, **kwargs):
        """
        Gaussian Process from sklearn. Assumes identical variance across lakes, though not means.

        Args:
            kernel: Gaussian Process kernel
        """
        super().__init__()
        self.model = GaussianProcessRegressor(kernel, *args, **kwargs)
        self.lags = lags or {"y": 3}

    @property
    def name(self):
        return "LaggedGPModel"

    def fit(self, X, y, *args, **kwargs):
        assert len(X.shape) == 2
        lagged_y_vars = self._lag_y_var(y)
        lag_y_align, X_align = xr.align(lagged_y_vars, X)

        train_df = xr.concat([X_align, flatten_array(lag_y_align)], dim="variable")
        train_y, _ = xr.align(y, train_df)

        self.model.fit(train_df, train_y, *args, **kwargs)

    def _lag_y_var(self, y):
        return lag_array(y, range(1, self.lags["y"] + 1)).dropna("Date")

    def predict(
        self,
        X: xr.DataArray,
        y: xr.DataArray = None,
        forecast_steps=12,
        alpha=0.05,
        *args,
        **kwargs,
    ) -> xr.DataArray:
        assert y is not None, "y cannot be None in order to use lagged variables"
        X_covars = X[-forecast_steps:]
        lagged_y = self._lag_y_var(y).dropna("Date")

        align_X, align_y = xr.align(X_covars, lagged_y)
        initial_y = align_y[[0]].values

        predictions = []

        # TODO: make this more functional / simplify
        for x_cov in align_X.transpose("Date", ...):
            flattened_y = flatten_array(initial_y)[0]
            pred_input = np.concatenate([x_cov.values, flattened_y]).reshape(1, -1)
            new_mean, new_sd = self.model.predict(pred_input, return_std=True)

            pred = np.stack([new_mean, new_sd], axis=2)
            predictions.append(pred)

            # prepend the new mean - the first value is the most recent lag
            initial_y = np.concatenate(
                [new_mean.reshape(-1, 4, 1), initial_y[:, :, 1:]],  # date, lake, lag
                axis=2,
            )

        # This transpose has mean first based on the for loop above
        mean, sd = np.concatenate(predictions, axis=0).transpose(2, 0, 1)
        z_scores = norm.ppf([alpha / 2, 1 - alpha / 2])
        lower, upper = mean + z_scores[0] * sd, mean + z_scores[1] * sd

        outputs = np.stack([mean, lower, upper, sd], axis=-1)

        formatted_predictions = output_forecast_results(
            outputs, X.indexes["Date"][-forecast_steps:]
        )
        return formatted_predictions

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass


class GPyTorchKernel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks=4, rank=4):
        super(GPyTorchKernel, self).__init__(train_x, train_y, likelihood)
        self.rank = rank
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel() * gpytorch.kernels.RQKernel(),
            num_tasks=num_tasks,
            rank=self.rank,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class MultitaskGP(ModelBase):

    def __init__(self, epochs=50, optimizer_params=None, **kernel_args):
        super().__init__()

        self.kernel = None
        self.likelihood = None
        self.mll = None
        self.optim = None
        self.epochs = epochs
        self.optimizer_params = optimizer_params or {"lr": 0.1}
        self.kernel_args = kernel_args

    @property
    def name(self):
        return "MultiTaskGP"

    def fit(self, X, y, *args, **kwargs):

        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=y.shape[1]
        )
        self.kernel = GPyTorchKernel(
            X, y, self.likelihood, num_tasks=y.shape[1], **self.kernel_args
        )

        # Find optimal model hyperparameters
        self.kernel.train()
        self.likelihood.train()

        # Use the adam optimizer
        self.optim = torch.optim.Adam(
            self.kernel.parameters(), **self.optimizer_params
        )  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.kernel
        )

        for i in range(self.epochs):
            self.optim.zero_grad()
            output = self.kernel(X)
            loss = -self.mll(output, y)
            loss.backward()
            if i % 50 == 0:
                logger.info(
                    "Iter %d/%d - Loss: %.3f" % (i + 1, self.epochs, loss.item())
                )
            self.optim.step()

    def predict(self, X, y=None, forecast_steps=12, *args, **kwargs) -> xr.DataArray:
        self.kernel.eval()
        self.likelihood.eval()

        X_tensor = torch.tensor(X.values, dtype=torch.float32)

        with torch.no_grad():
            preds = self.likelihood(self.kernel(X_tensor))

        mean, (low, high), std = (
            preds.mean,
            preds.confidence_region(),
            preds.variance ** (1 / 2),
        )

        output = torch.stack([mean, low, high, std], dim=-1).detach().numpy()
        return output_forecast_results(
            output[-forecast_steps:],
            forecast_labels=X.indexes["Date"][-forecast_steps:],
        )

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass
