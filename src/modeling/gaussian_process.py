import logging
from abc import ABC
from functools import partial

import arviz as az
import gpytorch
import jax
import numpy as np
import numpyro
import torch
import xarray as xr
from jax import numpy as jnp
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor, kernels

from src.modeling.modeling import ModelBase, NumpyroModel
from src.postprocessing import output_forecast_results
from src.utils import flatten_array, lag_array


__all__ = [
    "SklearnGPModel",
    "LaggedSklearnGP",
    "MultitaskGP",
    "HierarchicalARGP",
    "rbf_kernel",
    "matern_kernel",
    "rq_kernel",
]

logger = logging.getLogger(__name__)


# Kernel definitions are taken from https://www.cs.toronto.edu/~duvenaud/cookbook/
def rbf_kernel(X, Z, lengthscale, variance, noise=None, jitter=1e-6):
    # See https://num.pyro.ai/en/latest/examples/gp.html
    diffs = jnp.sqrt(((X[:, None] - Z) ** 2).sum(axis=-1))
    output = variance * jnp.exp(-0.5 * diffs**2 / lengthscale**2)

    if noise is not None:
        output += (noise + jitter) * jnp.eye(X.shape[0])
    return output


def matern_kernel(
    X, Z, var, length, noise, nu=3 / 2, jitter=1.0e-6, include_noise=True
):
    r = jnp.linalg.norm(X[:, None] - Z, axis=-1)
    if nu == 1 / 2:
        k = var * jnp.exp(-jnp.power(r / (jnp.sqrt(2) * length), 2))
    elif nu == 3 / 2:
        k = var * (1 + jnp.sqrt(3) * r / length) * jnp.exp(-jnp.sqrt(3) * r / length)
    elif nu == 5 / 2:
        k = (
            var
            * (1 + jnp.sqrt(5) * r / length + 5 * r**2 / (3 * length**2))
            * jnp.exp(-jnp.sqrt(5) * r / length)
        )
    else:
        raise ValueError("nu must be 1/2 or 3/2")

    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k


def rq_kernel(X, Z, alpha, length, noise, jitter=1.0e-6, include_noise=True):
    r = jnp.linalg.norm(X[:, None] - Z, axis=-1)
    k = (1 + r**2 / (2 * alpha * length**2)) ** (-alpha)
    if include_noise:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k


def general_gp_predict(
    self, rng_key, X, Y, X_test, var, length, noise, use_cholesky=True
):
    """
    unused prediction method, but kept for reference
    """
    # compute kernels between train and test data, etc.
    k_pp = self.kernel_fn(X_test, X_test, var, length, noise, include_noise=True)
    k_pX = self.kernel_fn(X_test, X, var, length, noise, include_noise=False)
    k_XX = self.kernel_fn(X, X, var, length, noise, include_noise=True)

    # this is pretty much straight copied from the numpyro tutorial on GPs
    # since K_xx is symmetric positive-definite, we can use the more efficient and
    # stable Cholesky decomposition instead of matrix inversion
    if use_cholesky:
        K_xx_cho = jax.scipy.linalg.cho_factor(k_XX)
        K = k_pp - jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, k_pX.T))
        mean = jnp.matmul(k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, Y))
    else:
        K_xx_inv = jnp.linalg.inv(k_XX)
        K = k_pp - jnp.matmul(k_pX, jnp.matmul(K_xx_inv, jnp.transpose(k_pX)))
        mean = jnp.matmul(k_pX, jnp.matmul(K_xx_inv, Y))

    sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), 0.0)) * jax.random.normal(
        rng_key, X_test.shape[:1]
    )

    # we return both the mean function and a sample from the posterior predictive for the
    # given set of hyperparameters
    return mean, mean + sigma_noise


class HierarchicalARGP(NumpyroModel):

    def __init__(
        self,
        lags=None,
        num_chains=4,
        num_samples=1000,
        num_warmup=1000,
        kernel_fn=None,
        lakes=("sup", "mic_hur", "eri", "ont"),
        ar_lag=3,
    ):
        super().__init__(lags, num_chains, num_samples, num_warmup)
        self.train_x = None
        self.train_y = None
        self.lakes = list(lakes)
        self.ar_lag = 3

        if kernel_fn is None:
            self.kernel_fn = partial(matern_kernel, nu=3 / 2)
        else:
            self.kernel_fn = kernel_fn

    @property
    def coords(self):
        return {
            "lakes": self.lakes,
        }

    @property
    def dims(self):
        return {
            "kernel_var_l": ["lakes"],
            "kernel_length_l": ["lakes"],
            "kernel_noise_l": ["lakes"],
        }

    @property
    def name(self):
        return "HierarchicalARGP"

    @staticmethod
    def model(X, Y, ar_lag=3, kernel_fn=matern_kernel):
        # set uninformative log-normal priors on our three kernel hyperparameters
        var = numpyro.sample("kernel_var", dist.HalfNormal(1))
        noise = numpyro.sample("kernel_noise", dist.HalfNormal(3.0))
        length = numpyro.sample("kernel_length", dist.LogNormal(0, 2.0))

        n_lakes = Y.shape[1]
        with numpyro.plate("lake_plate", n_lakes):
            var_l = numpyro.sample("kernel_var_l", dist.HalfNormal(var))
            noise_l = numpyro.sample("kernel_noise_l", dist.HalfNormal(noise))
            length_l = numpyro.sample("kernel_length_l", dist.LogNormal(length, 2.0))

        lags = [jnp.roll(Y, i, axis=0)[ar_lag:] for i in jnp.arange(1, ar_lag + 1)]

        fit_X = jnp.concatenate([X[ar_lag:], *lags], 1)

        # compute kernel
        cov_matrices = [
            kernel_fn(fit_X, fit_X, var_l[i], length_l[i], noise_l[i])
            for i in range(n_lakes)
        ]

        # make note that this doens't do anything for the paper
        # explain

        # NOTE: this slowed down the model a ton and didn't yield and better estimates
        # https://num.pyro.ai/en/stable/distributions.html#numpyro.distributions.continuous.LKJCholesky
        # theta = numpyro.sample("in_lake_errors", dist.HalfNormal(jnp.ones(n_lakes)))
        # sigma = jnp.sqrt(theta)
        # L_Omega = numpyro.sample("L_Omega", dist.LKJCholesky(n_lakes, concentration=1.0))

        # we can also use a faster formula `L_Omega = sigma[..., None] * L_omega`
        # L_Omega = jnp.matmul(jnp.diag(sigma), L_Omega)
        # L_Omega = sigma[..., None] * L_Omega
        # interLake_sigma = numpyro.sample("interLake_sigma", dist.MultivariateNormal(jnp.zeros(n_lakes), scale_tril=L_Omega))

        y_lakes = [
            numpyro.sample(
                f"y{i}",
                dist.MultivariateNormal(
                    loc=jnp.zeros(fit_X.shape[0]), covariance_matrix=k
                ),
                obs=Y[ar_lag:, i],
            )
            for i, k in enumerate(cov_matrices)
        ]

        numpyro.deterministic("y", jnp.stack(y_lakes, axis=-1))

    def fit(self, X, y, rng_key=None, *args, **kwargs):

        # need to include the training and testing parts for forecasting
        self.train_x = jnp.array(X)
        self.train_y = jnp.array(y)

        sampling_kernel = NUTS(self.model)
        mcmc = MCMC(
            sampling_kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
        )

        rng_key = self.get_rng_key() if rng_key is None else rng_key

        mcmc.run(rng_key, self.train_x, self.train_y)
        self.trace = az.from_numpyro(mcmc, coords=self.coords, dims=self.dims)

        self.posterior_samples = mcmc.get_samples()

        def _predict(rng_key, X, Y, X_test, var, length, noise):
            """
            must define a static method here so that jax.vmap can be used, along with compilation
            """
            # compute kernels between train and test data, etc.
            sample_keys = jax.random.split(rng_key, X_test.shape[0])
            mean_y = []
            for i in range(0, X_test.shape[0]):
                lags = jnp.stack(
                    [
                        jnp.roll(Y, i, axis=0)[self.ar_lag :]
                        for i in jnp.arange(1, self.ar_lag + 1)
                    ],
                    axis=1,
                )
                xi_test = jnp.concatenate([X_test[i], Y[-self.ar_lag:]]).reshape(1, -1)
                fitX = jnp.concatenate([X[self.ar_lag :], lags], 1)
                k_XX = self.kernel_fn(
                    fitX, fitX, var, length, noise, include_noise=True
                )

                k_pp = self.kernel_fn(
                    xi_test, xi_test, var, length, noise, include_noise=True
                )
                k_pX = self.kernel_fn(
                    xi_test, fitX, var, length, noise, include_noise=False
                )

                K_xx_cho = jax.scipy.linalg.cho_factor(k_XX)
                K = k_pp - jnp.matmul(
                    k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, k_pX.T)
                )
                mean = jnp.matmul(
                    k_pX, jax.scipy.linalg.cho_solve(K_xx_cho, Y[self.ar_lag :])
                )

                sigma_noise = jnp.sqrt(jnp.clip(jnp.diag(K), 0.0)) * jax.random.normal(
                    sample_keys[i], xi_test.shape[0]
                )
                pred_mean, pred_value = mean, mean + sigma_noise
                mean_y.append(pred_mean)
                Y = jnp.append(Y, pred_value)
                X = jnp.concatenate([X, X_test[i].reshape(1, -1)])

            return jnp.concat(mean_y), Y[-X_test.shape[0] :]

        self._prediction_fn = jax.jit(_predict)
            

    def predict(
        self,
        X_test,
        forecast_steps=12,
        rng_key=None,
        X=None,
        y=None,
        sample_subset=None,
        device=None,
        alpha=0.05,
        *args,
        **kwargs,
    ):
        """Adaptation of the prediction function that forecasts multiple steps ahead,
        assumping that each forecasted value is used as an input for the next forecasted value.

        Note that this function is meant to be applied over a set of parameters, in particular
        the results of simulations for var, length, and noise.

        Args:
            rng_key (): JAX random key
            X (array): Covariate features
            Y (array): The training target variable - required for lags.
            X_test (array): Covariates features for the test set.
            var (float): Variance parameter for the kernel
            length (float): Length parameter for the kernel
            noise (float): Noise parameter for the kernel
            ar_lag (int, optional): Numer of autoregressive lags. Defaults to 3.

        Returns:
            mean, forecast: A tuple of the forecasted mean and forecasted true values.
        """

        if device is None:
            device = jax.devices("cpu")[0]

        post_sims = self.num_chains * self.num_samples
        if sample_subset is None:
            sample_subset = post_sims

        if X is None:
            X = self.train_x
        if y is None:
            y = self.train_y
        else:
            y = jnp.array(y[:-forecast_steps])
        
        if rng_key is None:
            rng_key = self.get_rng_key()
        
        date_labels = X_test.indexes["Date"][-forecast_steps:]
        X_test_array = jnp.array(X_test)

        # The X_test values provided should be for each time step and START at basically 1 month ahead
        # We are also assuming that the y values, if provided, are longer than the forecast steps, so
        # we remove the last "forecast_steps" values and iterate over the covariates, starting at
        # the first row of X_test.
        param_dict = {k: v[:sample_subset] for k, v in self.posterior_samples.items()}
        vmap_args = (
            jax.random.split(rng_key, (sample_subset, 4)),
            X,
            y,
            X_test_array[:forecast_steps],
            param_dict["kernel_var_l"],
            param_dict["kernel_length_l"],
            param_dict["kernel_noise_l"],
        )

        # iteration proceeds from outer to inner, SO
        # 1) Iterate first over the columns of RNG_keys, the Y values,
        #    and var, length, and noise for each lake
        # 2) Then, iterate over each "row", in this case a single sample for vraiance, length, noise,
        #    and the RNG key for the forecast. At each point, we forecast "forecast_steps" ahead.
        # if device is None:
        #     device = jax.devices("cpu")[0]
        forecast_fn = jax.vmap(
            jax.jit(self._prediction_fn, device=device), in_axes=(0, None, None, None, 0, 0, 0)
        )
        forecast_fn = jax.vmap(forecast_fn, in_axes=(1, None, 1, None, 1, 1, 1))

        # note that here, we're ignoring the output of the mean and going just for the forecasted values
        _, preds = forecast_fn(*vmap_args)
        preds = preds.transpose(2, 0, 1)

        # forecast -> lake -> value
        pred_outputs = jnp.stack([
            preds.mean(axis=-1),
            jnp.quantile(preds, alpha/2, axis=-1),
            jnp.quantile(preds, 1-alpha/2, axis=-1),
            preds.std(axis=-1)
        ], axis=2)

        return output_forecast_results(pred_outputs, forecast_labels=date_labels, lakes=self.lakes)


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
