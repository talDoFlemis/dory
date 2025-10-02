import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class StandardScaler(TransformerMixin, BaseEstimator):
    mean_: np.ndarray
    scale_: np.ndarray

    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        self.mean_ = X.mean()
        self.scale_ = X.std(ddof=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        arr = (X - self.mean_) / self.scale_
        return arr
