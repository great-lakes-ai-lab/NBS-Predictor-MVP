import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import ecdf

__all__ = [
    "convert_to_labels",
    "save_model",
    "output_forecast_results",
    "ExceedanceProbClosure",
    "PostprocessingPipeline",
]


class PostprocessingPipeline(object):

    def __init__(self, steps):
        self.steps = steps

    def transform(self, y, *args, **kwargs):
        predictions = y
        for name, transformer in self.steps:
            predictions = transformer.transform(predictions, *args, **kwargs)

        return predictions


def convert_to_labels(probabilities, threshold=0.5):
    """
    Convert predicted probabilities to class labels based on a threshold.

    Args:
    - probabilities (np.ndarray): Predicted probabilities
    - threshold (float): Threshold for classifying as positive

    Returns:
    - np.ndarray: Predicted class labels
    """
    return np.where(probabilities > threshold, 1, 0)


def save_model(model, file_path):
    """
    Save a trained model to a file.

    Args:
    - model: Trained model
    - file_path (str): Path to save the model
    """
    import joblib

    joblib.dump(model, file_path)


class ExceedanceProbClosure(object):

    def __init__(
        self,
        hist_rnbs: xr.DataArray,
        as_labels=False,
        exceedance_thresholds=(0.25, 0.75),
    ):
        super().__init__()
        self.quantile_fns = {
            str(label): ecdf(arr.squeeze()) for label, arr in hist_rnbs.groupby("lake")
        }
        self.as_labels = as_labels
        self.exceedance_thresholds = exceedance_thresholds

    def transform(self, X, y=None, *args, **kwargs):
        return self.__call__(X)

    def __call__(self, x: xr.DataArray):

        exceedances = xr.concat(
            [
                xr.DataArray(
                    1 - self.quantile_fns[str(arr.lake.values)].cdf.evaluate(arr),
                    dims=arr.dims,
                    coords=arr.coords,
                )
                for arr in x.transpose("lake", ...)
            ],
            dim="lake",
        ).transpose("Date", ...)

        if self.as_labels:
            # replace with labels if requested
            exceedances = xr.where(
                exceedances > self.exceedance_thresholds[0],
                "Wet",
                xr.where(exceedances > self.exceedance_thresholds[1], "Dry", "Normal"),
            )
        return exceedances


def output_forecast_results(
    array,
    forecast_labels: pd.DatetimeIndex,
    lakes=("sup", "mic_hur", "eri", "ont"),
    **kwargs
):
    assert len(array.shape) == 3, "Dims must be forecast -> lake -> value"
    if isinstance(lakes, tuple):
        lakes = list(lakes)

    results = xr.DataArray(
        array,
        coords={
            "Date": forecast_labels,
            "lake": lakes,
            "value": ["mean", "lower", "upper", "std"],
        },
        dims=["Date", "lake", "value"],
        name="forecasts",
        attrs=kwargs,
    )
    return results
