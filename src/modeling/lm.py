import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xarray.core.dataarray import DataArray

from modeling.modeling import ModelBase
from postprocessing.postprocessing import output_forecast_results


def working_hotelling(alpha, n):
    return np.sqrt(2 * stats.distributions.f.cdf(alpha, 2, n - 2))


class LinearModel(ModelBase):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.model = LinearRegression()
        self.mse = None
        self.mean_x = None
        self.n = None
        self.train_x = None
        self.total_x = None

    @property
    def name(self):
        return "LinearModel"

    def fit(self, X, y):
        self.model.fit(X, y)
        self.mean_x = X.mean(axis=0)
        self.n = X.shape[0]
        self.tran_x = X

        self.total_x = ((X - self.mean_x) ** 2).sum(axis=1).sum()

        train_predictions = self.model.predict(X)
        self.mse = mean_squared_error(y, train_predictions, multioutput="raw_values")

    def predict(
        self, X, y=None, forecast_steps=12, alpha=0.05, *args, **kwargs
    ) -> DataArray:
        predictions = self.model.predict(X)
        prediction_x_diff = ((X - self.mean_x) ** 2).sum(axis=1)

        broadcasted_vector = np.broadcast_to(self.mse, (X.shape[0], self.mse.shape[0]))
        x_diffs = np.repeat(
            (1 + 1 / self.n + prediction_x_diff / self.total_x).values.reshape(-1, 1),
            4,
            axis=1,
        )
        stddev_est = np.sqrt(broadcasted_vector * x_diffs)

        wh_num = working_hotelling(1 - alpha, self.n)

        lower, upper = (
            predictions - wh_num * stddev_est,
            predictions + wh_num * stddev_est,
        )

        outputs = np.stack([predictions, lower, upper, stddev_est], axis=2)[
            -forecast_steps:
        ]
        date_labels = X.indexes["Date"][-forecast_steps:]

        return output_forecast_results(outputs, date_labels)

    def save(self):
        pass

    @classmethod
    def load(self):
        pass
