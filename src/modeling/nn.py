import jax.nn
import numpyro
from jax import numpy as jnp
from numpyro import distributions as dist

from src.constants import lake_order
from src.modeling.modeling import NumpyroModel


class BayesNN(NumpyroModel):

    def __init__(self, lags=None, num_chains=4, num_samples=1000, num_warmup=1000):
        super().__init__(lags, num_chains, num_samples, num_warmup)

    @property
    def name(self):
        return "BayesNN"

    @property
    def coords(self):
        return {"lakes": lake_order}

    @property
    def dims(self):
        return {"sigma": ["lakes"]}
        pass

    @staticmethod
    def model(y, y_index, lags, covariates, future=0):

        input_dim = covariates.shape[1]
        h1 = 16
        output_dim = 4

        nu = numpyro.sample("nu", dist.HalfNormal(4))

        sigma = numpyro.sample("theta", dist.HalfNormal(5), sample_shape=(4,))

        # l_omega = numpyro.sample("corr", dist.LKJCholesky(4, concentration=0.5))
        # sigma = jnp.sqrt(theta)
        # L_Omega = sigma[..., None] * l_omega

        # The weights and bias for the hidden layer
        w1 = numpyro.sample(
            "w1",
            dist.Normal(
                jnp.zeros((input_dim, h1)),
                1.0 * jnp.ones((input_dim, h1)),
            ),
        )
        b1 = numpyro.sample("b1", dist.Normal(jnp.zeros(h1), jnp.ones(h1)))

        # The weights and bias for the output layer
        w2 = numpyro.sample(
            "w2",
            dist.Normal(
                jnp.zeros((h1, output_dim)),
                1.0 * jnp.ones((h1, output_dim)),
            ),
        )
        b2 = numpyro.sample(
            "b2", dist.Normal(jnp.zeros(output_dim), jnp.ones(output_dim))
        )

        # The network architecture
        z1 = jax.nn.relu(jnp.matmul(jnp.array(covariates), w1) + b1)
        z2 = jnp.matmul(z1, w2) + b2

        if future > 0:
            ys = numpyro.sample("y", dist.StudentT(nu, z2, sigma))
            y_forecast = numpyro.deterministic("y_forecast", ys[-future:])
        else:
            ys = numpyro.sample("y", dist.StudentT(nu, z2, sigma), obs=y)
