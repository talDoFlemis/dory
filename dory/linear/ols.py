from dory.linear import LinearRegression
import pandas as pd
import numpy as np


class OrdinaryLeastSquares(LinearRegression):
    def fit(self, X: pd.DataFrame, y: pd.Series):
        x_n = X.to_numpy()
        y_n = y.to_numpy()
        X = np.c_[np.ones(x_n.shape[0]).T, x_n]
        w = np.linalg.inv(X.T @ X) @ X.T @ y_n
        self.parameters = pd.Series(w)


    def predict(self, X: pd.DataFrame) -> pd.Series:
        x_n = X.to_numpy()
        ndarray =  x_n @ self.parameters.to_numpy()
        return pd.Series(ndarray)

    def get_parameters(self) -> pd.Series:
        return self.parameters