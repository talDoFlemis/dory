import numpy as np
from dory.linear.gd import GradientDescentRegressor
import pandas as pd


def test_gradient_descent():
    # Arrange
    # Function to approximate f(x) = 2^x
    initial_weights = np.array([0.0, 0.0])
    artificial_df = pd.read_csv("/home/flemis/codes/dory/lista_01/artificial1d.csv", header=None)
    X = artificial_df.iloc[:, :-1]
    y = artificial_df.iloc[:, -1]
    model = GradientDescentRegressor(initial_weights=initial_weights)
    model.fit(X.to_numpy(), y.to_numpy())
