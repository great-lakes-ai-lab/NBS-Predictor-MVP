import xarray as xr

from src.step3_modeling.modeling import ModelBase
from src.step4_postprocessing.postprocessing import output_forecast_results

__all__ = [
    # Classes
    "DefaultEnsemble",
]


class DefaultEnsemble(ModelBase):

    @property
    def name(self):
        return "DefaultEnsemble"

    def __init__(self, alpha=0.05):
        super().__init__()
        self.alpha = alpha
        self.month_df = None
        self.lakes = ["sup", "mic_hur", "eri", "ont"]

    def fit(self, X, y, *args, **kwargs):
        quantiles = (
            y.groupby("Date.month")
            .quantile(
                q=[self.alpha / 2, 1 - self.alpha / 2],
            )
            .assign_coords({"quantile": ["lower", "upper"]})
            .rename({"quantile": "variable"})
        )
        means = y.groupby("Date.month").mean().expand_dims(variable=["mean"])
        std = y.groupby("Date.month").std().expand_dims(variable=["std"])

        self.month_df = xr.concat([means, quantiles, std], dim="variable")
        return self

    def predict(self, X, y, forecast_steps=12, *args, **kwargs) -> xr.DataArray:
        # convert to DatetimeIndex type just in case for .month syntax

        forecast_index = y.indexes["Date"][-forecast_steps:]
        forecasts = (
            self.month_df.sel(month=forecast_index.month)
            .rename(month="Date")
            .assign_coords(Date=forecast_index)
            .transpose("Date", "lake", "variable")
        )

        formatted_output = output_forecast_results(forecasts, forecast_index)
        return formatted_output

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass
