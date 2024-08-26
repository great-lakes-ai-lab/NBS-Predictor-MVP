import calendar

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


class XArrayStandardScaler(object):
    """
    A class used to standardize xarray DataArrays by removing the mean and
    scaling to unit variance along the 'Date' dimension.

    Attributes
    ----------
    is_fitted : bool
        A flag indicating whether the scaler has been fitted.
    means : xr.DataArray or None
        The means of the DataArray along the 'Date' dimension.
    std : xr.DataArray or None
        The standard deviations of the DataArray along the 'Date' dimension.
    dims : tuple or None
        The dimensions of the DataArray being processed.
    coords : dict or None
        The coordinates of the DataArray being processed.
    """

    def __init__(self):
        """
        Initializes the XArrayStandardScaler with initial values.

        Parameters
        ----------
        None
        """
        self.is_fitted = False
        self.means = None
        self.std = None
        self.dims = None
        self.coords = None

    def fit(self, X: xr.DataArray, y=None):
        """
        Compute the mean and standard deviation of the DataArray along the 'Date'
        dimension for later scaling.

        Parameters
        ----------
        X : xr.DataArray
            Input DataArray with 'Date' as the first dimension.
        y : None, default=None
            Ignored. This parameter exists for compatibility with sklearn's API.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        assert X.dims[0] == "Date"

        self.means = X.mean("Date")
        self.std = X.std("Date")
        self.is_fitted = True

    def transform(self, X: xr.DataArray, y=None) -> xr.DataArray:
        """
        Standardize the DataArray by removing the mean and scaling to unit variance
        using previously computed means and standard deviations.

        Parameters
        ----------
        X : xr.DataArray
            Input DataArray to be transformed.
        y : None, default=None
            Ignored. This parameter exists for compatibility with sklearn's API.

        Returns
        -------
        X_transformed : xr.DataArray
            Standardized DataArray.
        """
        return (X - self.means) / self.std

    def fit_transform(self, X: xr.DataArray, y=None) -> xr.DataArray:
        """
        Fit to the data and then transform it.

        Parameters
        ----------
        X : xr.DataArray
            Input DataArray to be fitted and transformed.
        y : None, default=None
            Ignored. This parameter exists for compatibility with sklearn's API.

        Returns
        -------
        X_transformed : xr.DataArray
            Standardized DataArray.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: xr.DataArray, y=None) -> xr.DataArray:
        """
        Scale back the DataArray to the original distribution by reversing the
        standardization.

        Parameters
        ----------
        X : xr.DataArray
            Standardized DataArray to be inversely transformed.
        y : None, default=None
            Ignored. This parameter exists for compatibility with sklearn's API.

        Returns
        -------
        X_original : xr.DataArray
            DataArray restored to the original distribution.
        """
        return X * self.std + self.means


class MinMaxScaler(object):

    def __init__(self):
        self.is_fitted = False
        self.mins = None
        self.maxes = None
        self.dims = None
        self.coords = None

    def fit(self, X: xr.DataArray, y=None):
        assert X.dims[0] == "Date"

        self.mins = X.min("Date")
        self.maxes = X.max("Date")

        self.is_fitted = True

    def transform(self, X: xr.DataArray, y=None):
        return (X - self.mins) / (self.maxes - self.mins)

    def fit_transform(self, X: xr.DataArray, y=None):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: xr.DataArray, y=None):
        return X * (self.maxes - self.mins) + self.mins


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
    scaler = XArrayStandardScaler()
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


class SeasonalFeatures(object):

    def __init__(self, period=12):
        self.period = period

    def fit(self, X: xr.DataArray, y=None):
        pass

    def transform(self, X: xr.DataArray, y=None):
        features = np.append(
            sin_feature(X.indexes["Date"].month.values, self.period).reshape(-1, 1),
            cos_feature(X.indexes["Date"].month.values, self.period).reshape(-1, 1),
            axis=1,
        )
        return xr.DataArray(
            features,
            coords={
                "Date": X.indexes["Date"],
                "variable": pd.Index([f"sin_{self.period}", f"cos_{self.period}"]),
            },
            dims=["Date", "variable"],
        )

    def fit_transform(self, X: xr.DataArray, y=None):
        return self.transform(X)


def cos_feature(x, period):
    return np.cos(x / period * 2 * np.pi)


def sin_feature(x, period):
    return np.sin(x / period * 2 * np.pi)


class XArrayAdapter(object):

    def __init__(self, sklearn_preprocessor, feature_prefix="f"):
        super().__init__()
        self.sklearn_preprocessor = sklearn_preprocessor
        self.feature_prefix = feature_prefix

    def fit(self, X: xr.DataArray, y=None):
        self.sklearn_preprocessor.fit(X, y)

    def transform(self, X: xr.DataArray, y=None) -> xr.DataArray:
        """
        Runs the transform method of the sklearn preprocessor, but returns
        the coordinates and dimensions of the original data. Note that this assumes that new
        dimensions / columns are not created
        """
        transformed_X = self.sklearn_preprocessor.transform(X)
        return xr.DataArray(
            transformed_X,
            coords={
                "Date": X.coords["Date"],
                "variable": [
                    f"{self.feature_prefix}_{i}" for i in range(transformed_X.shape[1])
                ],
            },
            dims=["Date", "variable"],
        )

    def fit_transform(self, X: xr.DataArray, y=None):
        self.fit(X, y)
        return self.transform(X)


class XArrayFeatureUnion(object):

    def __init__(self, transformers):
        self.transformers = transformers

    def transform(self, X: xr.DataArray, y=None):
        values = [transformer.transform(X) for _, transformer in self.transformers]
        return xr.concat(values, dim="variable")

    def fit(self, X: xr.DataArray, y=None):
        for _, transformer in self.transformers:
            transformer.fit(X, y)

    def fit_transform(self, X: xr.DataArray, y=None):
        self.fit(X, y)
        return self.transform(X)
