from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from properscoring import crps_gaussian
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)

__all__ = [
    # Functions
    "calculate_tail_summary",
    "calc_exceedance_prob",
    "crps",
    "summarize",
]


def calculate_tail_summary(df, thresholds=np.arange(0.1, 1, 0.05), tail_alpha=0.10):
    df = df[["true", "std", "pred", "lower", "upper"]].dropna()
    truth, std, pred, low, high = df[["true", "std", "pred", "lower", "upper"]].values.T

    quantiles = df["true"].quantile(thresholds)
    tail_low, tail_high = df["true"].quantile([tail_alpha / 2, 1 - tail_alpha / 2]).T
    quantiles.index = [f"q{i * 100:0.0f}" for i in np.arange(0.1, 1, 0.05)]
    tail_index = np.where((truth <= tail_low) | (truth >= tail_high))

    # ep = calc_exceedance_prob(pred)

    stats = pd.Series(
        {
            "mse": mean_squared_error(truth[tail_index], pred[tail_index]),
            "mse_tail": mean_squared_error(truth, pred),
            "rmse": mean_squared_error(truth, pred) ** (1 / 2),
            "rmse_tail": mean_squared_error(truth[tail_index], pred[tail_index])
            ** (1 / 2),
            "mae": mean_absolute_error(truth, pred),
            "mae_tail": mean_absolute_error(truth[tail_index], pred[tail_index]),
            "coverage": np.mean((low <= truth) & (truth <= high)),
            "coverage_tail": np.mean(
                (low[tail_index] <= truth[tail_index])
                & (truth[tail_index] <= high[tail_index])
            ),
            "crps": np.mean(crps_gaussian(truth, pred, std)),
            "r2": r2_score(truth, pred),
            "N": df.shape[0],
        }
    )

    return stats


def calc_exceedance_prob(values: Union[pd.Series, NDArray]):
    try:
        sorted_vect = values.sort_values(ascending=False)
    except ValueError:
        # must be a series
        sorted_vect = pd.Series(values).sort_values(ascending=False)
    ep_df = pd.DataFrame(
        {"values": sorted_vect, "rank": range(len(sorted_vect))},
        index=sorted_vect.index,
    )
    ep = (ep_df.shape[0] - ep_df["rank"] + 1) / (ep_df.shape[0] + 1)
    return ep.rename("ep").sort_index()


def crps(
    y_true: Union[pd.Series, NDArray],
    y_pred: Union[pd.Series, NDArray],
    stddev: Union[pd.Series, NDArray],
):
    """
    :param y_true: True values
    :param y_pred: Predicted values
    :return: The MSE of the predicted CDF vs. the empirical CDF
    """

    # Vectorized normal
    return crps_gaussian(y_true, mu=y_pred, sig=stddev)


def summarize(df, crps_func=crps_gaussian):
    return pd.Series(
        {
            "rmse": root_mean_squared_error(df["mean"], df["true"]),
            "variance": np.var(df["mean"] - df["true"]),
            "bias": np.mean(df["mean"] - df["true"]),
            "coverage": np.mean(
                (df["true"] >= df["lower"]) & (df["true"] <= df["upper"])
            ),
            "interval_len": np.mean(df["upper"] - df["lower"]),
            "crps": crps_func(df["true"], df["mean"], df["std"]).mean(),
            "N": df.shape[0],
        }
    )
