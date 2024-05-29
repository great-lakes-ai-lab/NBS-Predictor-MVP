import xarray as xr

from src.step3_modeling.modeling import ModelBase


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
        means = y.groupby("Date.month").mean().rename("mean")
        quantiles = (
            y.groupby("Date.month")
            .quantile(q=[self.alpha / 2, 1 - self.alpha / 2])
            .rename(quantile="variable")
        )
        std = y.groupby("Date.month").std().rename("std")

        self.month_df = xr.concat(
            [means, quantiles, std], dim="variable"
        ).assign_coords(variable=["mean", "lower", "uppder", "std"])
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

        formatted_output = self.output_forecast_results(forecasts, forecast_index)
        return formatted_output

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass
