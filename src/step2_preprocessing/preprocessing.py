import calendar

import pandas as pd
import xarray as xr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class XArrayScaler(object):

    def __init__(self):
        self.series_dim = None
        self.is_fitted = False
        self.means = None
        self.std = None
        self.dims = None
        self.coords = None

    def fit(self, X: xr.DataArray, y=None):
        assert X.dims[0] == "Date"
        self.series_dim = X.dims[-1]  # iterate over the last dim

        self.means = X.mean("Date")
        self.std = X.std("Date")

        self.is_fitted = True

    def transform(self, X: xr.DataArray, y=None):
        return (X - self.means) / self.std

    def fit_transform(self, X: xr.DataArray, y=None):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: xr.DataArray, y=None):
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


class CreateMonthDummies(object):

    def __init__(self, encoder=None):
        self.encoder = encoder or OneHotEncoder(
            categories="auto", sparse_output=False, drop=[1]
        )

    def fit(self, X: xr.DataArray, y=None):
        months = X.indexes["Date"].month.values.reshape(-1, 1)
        self.encoder.fit(months)

    def transform(self, X: xr.DataArray, y=None):
        months = X.indexes["Date"].month.values.reshape(-1, 1)
        month_dummies = self.encoder.transform(months)

        drop_categories = self.encoder.get_params().get("drop") or []

        coords = {
            "Date": X.indexes["Date"],
            "month": [
                calendar.month_abbr[i]
                for i in self.encoder.categories_[0]
                if i not in drop_categories
            ],
        }

        return xr.DataArray(month_dummies, dims=["Date", "month"], coords=coords)
