from skfda.datasets import fetch_weather

X_weather, y_weather = fetch_weather(
    return_X_y=True,
    as_frame=True,
)
fd = X_weather.iloc[:, 0].array

fd.plot()


import numpy as np
from sklearn.preprocessing import OneHotEncoder

# We first create the one-hot encoding of the climates.
enc = OneHotEncoder(handle_unknown="ignore")
enc.fit([["Atlantic"], ["Continental"], ["Pacific"]])
X = np.array(y_weather).reshape(-1, 1)
X = enc.transform(X).toarray()


import pandas as pd
from skfda.representation.basis import FourierBasis

X_df = pd.DataFrame(X)

y_basis = FourierBasis(n_basis=65)
y_fd = fd.coordinates[0].to_basis(y_basis)

from skfda.ml.regression import LinearRegression

funct_reg = LinearRegression(fit_intercept=True)
funct_reg.fit(X_df, y_fd)


import matplotlib.pyplot as plt

funct_reg.intercept_.plot()
funct_reg.coef_[0].plot()
funct_reg.coef_[1].plot()
funct_reg.coef_[2].plot()

plt.show()


X_test = pd.DataFrame(
    [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
)
predictions = funct_reg.predict(X_test)
predictions.plot()

plt.show()
