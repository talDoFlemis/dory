import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class GaussianDiscriminatorAnalysisClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, priors: np.ndarray | None = None):
        self.priors = priors

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.covariance_ = np.cov(X.T)

        if self.priors is None:
            self.priors_ = np.array([np.mean(y == c) for c in self.classes_])
        else:
            self.priors_ = self.priors

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = np.array(
            [self.__density_fn(X, mean, self.covariance_) for mean in self.means_]
        )
        return self.classes_[np.argmax(probabilities, axis=0)]

    def __density_fn(
        self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray
    ) -> np.ndarray:
        cov_inv = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm_const = 1.0 / (np.sqrt((2 * np.pi) ** X.shape[1] * det_cov))
        X_mean = X - mean
        exponent = np.einsum("ij,ji->i", X_mean.dot(cov_inv), X_mean.T)
        return norm_const * np.exp(-0.5 * exponent)
