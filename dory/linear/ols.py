import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import validate_data, check_is_fitted


class OrdinaryLeastSquares(RegressorMixin, BaseEstimator):
    w_: np.ndarray
    def __init__(self, *, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X, y = validate_data(self, X, y)

        X = np.c_[np.ones(X.shape[0]).T, X]
        self.w_ = np.linalg.inv(X.T @ X) @ X.T @ y

        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X: np.ndarray) -> pd.Series:
        check_is_fitted(self)
        independent_term = self.w_[0]
        ndarray = X @ self.w_[1:]
        ndarray = ndarray + independent_term
        return pd.Series(ndarray)
