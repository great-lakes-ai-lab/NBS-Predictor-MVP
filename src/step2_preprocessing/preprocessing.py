from typing import Union
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xarray as xr
import pandas as pd


class XArrayScaler(object):

    def __init__(self):
        self.series_dim = None
        self.is_fitted = False
        self.means = None
        self.std = None
        self.dims = None
        self.coords = None

    def fit(self, X: xr.DataArray):
        assert X.dims[0] == "Date"
        self.series_dim = X.dims[-1]  # iterate over the last dim

        self.means = X.mean("Date")
        self.std = X.std("Date")

        self.is_fitted = True

    def transform(self, X: xr.DataArray):
        return (X - self.means) / self.std

    def fit_transform(self, X: xr.DataArray):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: xr.DataArray):
        return X * self.std + self.means


def handle_missing_values(data):
    """
    Handle missing values in the DataFrame.

    Args:
    - data (pd.DataFrame): Input data

    Returns:
    - pd.DataFrame: Data with missing values handled
    """
    imputer = SimpleImputer(strategy="mean")
    data_filled = imputer.fit_transform(data)
    return pd.DataFrame(data_filled, columns=data.columns)


def scale_features(data):
    """
    Scale numerical features in the DataFrame.

    Args:
    - data (pd.DataFrame): Input data

    Returns:
    - pd.DataFrame: Data with scaled features
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.select_dtypes(include=["float64", "int64"]))
    return pd.DataFrame(
        data_scaled, columns=data.select_dtypes(include=["float64", "int64"]).columns
    )
