import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import validate_data, check_is_fitted
import logging

logger = logging.getLogger(__name__)


class OrdinaryLeastSquaresRegressor(RegressorMixin, BaseEstimator):
    w_: np.ndarray

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        l2_alpha: float = 0.0,
    ):
        self.fit_intercept = fit_intercept
        self.l2_alpha = l2_alpha

    def fit(self, X, y):
        X, y = validate_data(self, X, y)
        X = np.c_[np.ones(X.shape[0]).T, X]

        if self.l2_alpha == 0.0:
            self.w_ = np.linalg.inv(X.T @ X) @ X.T @ y
            return self

        logger.info(f"Using L2 regularization with alpha {self.l2_alpha}")

        # Do not regularize the intercept term
        identity = np.identity(X.shape[1])
        identity[0, 0] = 0

        self.w_ = np.linalg.inv(X.T @ X + self.l2_alpha * identity) @ X.T @ y

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = np.c_[np.ones(X.shape[0]).T, X]
        ndarray = X @ self.w_
        return ndarray
