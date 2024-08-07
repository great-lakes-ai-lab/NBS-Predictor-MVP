from abc import ABC, abstractmethod

import arviz as az
import numpy as np
import xarray as xr
from jax import numpy as jnp
from jax.random import PRNGKey
from numpyro.diagnostics import hpdi
from numpyro.infer import MCMC, NUTS, Predictive
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.postprocessing import output_forecast_results

__all__ = [
    # Classes
    "ModelBase",
    "NumpyroModel",
    # Functions
    "split_data",
    "train_model",
    "evaluate_model",
]


class ModelBase(ABC):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def fit(self, X, y, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, y=None, forecast_steps=12, *args, **kwargs) -> xr.DataArray:
        """
        Abstract method for running the predictions
        Args:
            X:
            y: the input series
            forecast_steps: how many time steps from the end of the input series are actually forecast steps. This
            is implemented so that we can consider the covariates

            *args: other arguments
            **kwargs: keyword arguments

        Returns: A nx4x3 XArray DataArray where n is the number of periods forward that you are forecasting,
        for 4 lakes, as well as a low and high estimate value
        """
        pass

    @abstractmethod
    def save(self, path):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        pass


class NumpyroModel(ModelBase):

    def __init__(
        self,
        lags=None,
        num_chains=4,
        num_samples=1000,
        num_warmup=1000,
    ):
        super().__init__()
        self.lags = lags or {}
        self.num_chains = num_chains
        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.predictive_fn = None  # updated during kernel fitting
        self.trace = None
        self.mcmc = None
        self.lakes = None
        self._is_fitted = False

    @property
    def is_fitted(self):
        return self._is_fitted

    @property
    @abstractmethod
    def coords(self):
        pass

    @property
    @abstractmethod
    def dims(self):
        pass

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass

    def fit(self, X: xr.DataArray, y: xr.DataArray, rng_key=None, *args, **kwargs):
        """
        Args:
            y: the time series to predict forward
            y_index: The datetime index for the series
            X: Extra X variables that can be passed in as predictors at fitting time
        Returns:
            fitted kernel object
        """
        self.lakes, y_index = y.indexes["lake"], y.indexes["Date"]

        if rng_key is None:
            rng_key = self.get_rng_key()

        y = jnp.array(y)
        kernel = NUTS(self.model)

        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
        )
        mcmc.run(rng_key, y=y, y_index=y_index, lags=self.lags, covariates=X)
        samples = mcmc.get_samples()

        self.predictive_fn = Predictive(
            self.model, samples, return_sites=["y", "y_forecast"]
        )

        self.trace = az.from_numpyro(mcmc, coords=self.coords, dims=self.dims)
        self._is_fitted = True

    @staticmethod
    @abstractmethod
    def model(y, y_index, lags, covariates, future=0):
        """
        Model method. Must be static to work with Numpyro MCMC.
        Args:
            y:
            y_index:
            lags:
            covariates:
            future:

        Returns:

        """
        pass

    def predict(self, X, y=None, forecast_steps=12, rng_key=None, *args, **kwargs):
        """

        Args:
            X:
            y: The time series to predict forward
            forecast_steps:

        Returns:

        """
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

    def get_rng_key(self):
        return PRNGKey(np.random.randint(1e3))


def split_data(data, target_column):
    """
    Split data into train and test sets.

    Args:
    - data (pd.DataFrame): Input data
    - target_column (str): Name of the target column

    Returns:
    - tuple: (X_train, X_test, y_train, y_test)
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(model: ModelBase, X_train, y_train, *args, **kwargs):
    """
    Train a Random Forest Classifier.

    Args:
    - X_train (pd.DataFrame): Training features
    - y_train (pd.Series): Training target

    Returns:
    - RandomForestClassifier: Trained kernel
    """
    model.fit(X=X_train, y=y_train, *args, **kwargs)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a kernel using accuracy score.

    Args:
    - kernel: Trained kernel
    - X_test (pd.DataFrame): Test features
    - y_test (pd.Series): Test target

    Returns:
    - float: Accuracy score
    """
    y_pred = model.predict(None, X_test)
    return accuracy_score(y_test, y_pred)
