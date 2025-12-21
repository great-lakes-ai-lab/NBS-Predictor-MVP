import calendar
from collections.abc import Iterable
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr
from skfda import FDataBasis, FDataGrid
from skfda.representation.basis import Basis, BSplineBasis
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


class BasisFunctionTransformer(object):
    def __init__(
        self,
        default_basis: Iterable[Basis] | partial | Basis = BSplineBasis(n_basis=5),
        basis_mapping: dict | None = None,
    ):
        self.default_basis = default_basis
        self.basis_functions = default_basis
        self.domain = None
        self.basis_mapping = basis_mapping or {}
        self.original_coords = None
        self.original_dims = None

    def fit(self, X: xr.DataArray, y=None):
        self.domain = float(X["Date"].min()), float(X["Date"].max())
        self.basis_lookup = {}
        self.original_coords = X.coords
        self.original_dims = X.dims
        for lake_arr in X.transpose("lake", "variable", ...):
            lake_name = str(lake_arr.coords["lake"].values)
            self.basis_lookup.update({lake_name: {}})
            for var in lake_arr:
                var_name = str(var.coords["variable"].values)
                self.basis_lookup[lake_name].update(
                    {
                        var_name: self._grid_to_basis(
                            var,
                            self.basis_mapping.get(var_name, self.default_basis)(
                                domain_range=self.domain
                            ),
                        )
                    }
                )
        return

    def _grid_to_basis(self, data_array: xr.DataArray, basis_fn: Basis):
        data_grid = FDataGrid(
            data_array.values,
            grid_points=np.linspace(
                self.domain[0], self.domain[1], data_array.coords["Date"].shape[0]
            ),
        )
        grid_basis = data_grid.to_basis(basis_fn)
        return grid_basis

    def transform(self, X: xr.DataArray, y=None):
        lake_data = []
        for lake_arr in X.transpose("lake", "variable", ...):
            lake_name = str(lake_arr.lake.values)
            var_curves = []
            for var in lake_arr:
                var_name = str(var.coords["variable"].values)
                basis_coeff = xr.DataArray(
                    self._grid_to_basis(
                        var, self.basis_lookup[lake_name][var_name].basis
                    ).coefficients,
                    dims=("start_date", "basis_dim"),
                    coords={
                        "start_date": X.coords["start_date"],
                        "basis_dim": np.arange(
                            self.basis_lookup[lake_name][var_name].basis.n_basis
                        ),
                    },
                )
                var_curves.append(basis_coeff)
            all_vars = xr.concat(
                var_curves,
                dim=lake_arr.coords["variable"],
                # coords={"variable": lake_arr.coords["variable"]},
            )
            lake_data.append(all_vars)
        lake_coeff = xr.concat(lake_data, dim=X.coords["lake"]).transpose(
            "start_date", "basis_dim", "variable", "lake"
        )
        return lake_coeff

    def fit_transform(self, X: xr.DataArray, y=None):
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, coefficients: xr.DataArray, grid=None):
        full_output = []
        for lake in coefficients.coords["lake"].values:
            var_output = []
            if grid is not None:
                grid_points = grid
            else:
                grid_points = np.arange(self.domain[0], self.domain[1] + 1, 1)

            for var in coefficients.coords["variable"].values:
                coeffs = coefficients.sel(lake=lake, variable=var).values
                basis_fn = self.basis_lookup[lake][var].basis
                basis_output = (
                    FDataBasis(basis_fn, coefficients=coeffs)
                    .to_grid(grid_points)
                    .data_matrix[:, :, 0]
                )
                output_arr = xr.DataArray(
                    basis_output,
                    dims=["start_date", "Date"],
                    coords={
                        "start_date": self.original_coords["start_date"],
                        "Date": self.original_coords["Date"],
                    },
                )
                var_output.append(output_arr)

            var_output = xr.concat(var_output, dim=coefficients.coords["variable"])
            full_output.append(var_output)

        all_output = xr.concat(full_output, dim=coefficients.coords["lake"])
        return all_output.transpose(*self.original_dims)


def time_window_generator(input_data, train_size, test_size, reindex_domain=True):
    train_idx = 0
    test_idx = train_idx + train_size
    while (train_idx + test_idx) <= len(input_data):
        prior_set, forecast_set = (
            input_data[train_idx:test_idx],
            input_data[test_idx : (test_idx + test_size)],
        )
        prior_set.attrs["start_date"] = str(prior_set.indexes["Date"][0].date())
        forecast_set.attrs["start_date"] = str(forecast_set.indexes["Date"][0].date())
        prior_set.attrs["end_date"] = str(prior_set.indexes["Date"][-1].date())
        forecast_set.attrs["end_date"] = str(forecast_set.indexes["Date"][-1].date())
        if reindex_domain:
            train_lead_dim, test_lead_dim = (
                prior_set.dims[0],
                forecast_set.dims[0],
            )
            prior_set.coords[train_lead_dim] = np.arange(0, train_size)
            forecast_set.coords[test_lead_dim] = np.arange(0, test_size)

        yield prior_set, forecast_set
        train_idx += 1
        test_idx += 1


class WindowTransformer(object):
    def __init__(self, train_size=12, test_size=12, **kwargs):
        self.train_size = train_size
        self.test_size = test_size
        self.generator_kwargs = kwargs

    def transform(self, X: xr.DataArray):
        data_windows = list(
            time_window_generator(
                X,
                train_size=self.train_size,
                test_size=self.test_size,
                **self.generator_kwargs,
            )
        )

        train_window_dates = xr.DataArray(
            [train.attrs["start_date"] for train, _ in data_windows], dims="start_date"
        )
        test_window_dates = xr.DataArray(
            [test.attrs["start_date"] for _, test in data_windows], dims="start_date"
        )

        train_outputs = xr.concat(
            [train for train, _ in data_windows], dim=train_window_dates
        )
        test_outputs = xr.concat(
            [test for _, test in data_windows], dim=test_window_dates
        )

        return train_outputs, test_outputs

    def fit(self, X: xr.DataArray, y=None):
        pass

    def fit_transform(self, X: xr.DataArray, y=None):
        self.fit(X, y)
        return self.transform(X)
