import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, check_array, check_is_fitted


class PCACluster(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.pca = None

    def fit(self, X: np.ndarray, y=None):
        X = check_array(X)
        cov = np.cov(X, rowvar=False)
        U, S , Vt = np.linalg.svd(cov)
        self.means_ = np.mean(X, axis=0)

        self.explained_variance_ = S[:self.n_components]
        self.explained_variance_ratio_ = S[:self.n_components] / S.sum()
        self.components_ = Vt[:self.n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)
        return (X - self.means_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        Z = check_array(Z)
        return Z @ self.components_ + self.means_