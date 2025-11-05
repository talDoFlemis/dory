import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted
from typing import Literal


class KNNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_neighbors: int = 5,
        metric: Literal["euclidean", "mahalanobis"] = "euclidean",
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)

        if self.metric == "mahalanobis":
            self.inv_covariance_ = np.linalg.inv(np.cov(X.T))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        indexes = np.array([])  # Shape (n_samples, n_neighbors)

        if self.metric == "euclidean":
            indexes = self.__euclidean_distance(X)
        elif self.metric == "mahalanobis":
            indexes = self.__mahalanobis_distance(X)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        clazzes = [
            self.classes_[np.argmax(np.bincount(self.y_train_[neighbor]))]
            for neighbor in indexes
        ]

        return np.array(clazzes).reshape(-1, 1)

    def __mahalanobis_distance(self, X: np.ndarray) -> np.ndarray:
        mean = np.mean(self.X_train_)
        delta = X - mean
        D = np.sqrt(np.einsum("nj,jk,nk->n", delta, self.inv_covariance_, delta))
        return D

    def __euclidean_distance(self, X: np.ndarray) -> np.ndarray:
        # (A - B)^2 = A^2 + B^2 - 2AB
        a_square = np.sum(X**2, axis=1)[:, np.newaxis]  # Shape (n_samples, 1) and force broadcast
        b_square = np.sum(self.X_train_**2, axis=1)  # Shape (n_train_samples,)
        distances = (
            -2 * X @ self.X_train_.T + a_square + b_square
        )  # Shape (n_samples, n_train_samples)
        distances[distances < 0] = 0
        distances = np.sqrt(distances)
        return np.argsort(distances, axis=1)[
            :, : self.n_neighbors
        ]  # Shape (n_samples, n_neighbors)
