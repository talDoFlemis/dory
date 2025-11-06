import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, check_array, check_is_fitted
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
        self.classes_ = np.unique(y)
        # Problem when mapping labels using bincount
        self.label_to_idx_ = {label: idx for idx, label in enumerate(self.classes_)}
        self.y_train_ = np.array([self.label_to_idx_[label] for label in y.ravel()])

        if self.metric == "mahalanobis":
            cov = np.cov(X.T)
            # Add a small value (regularization) for numerical stability
            # in case the matrix is singular
            cov += np.eye(cov.shape[0]) * 1e-6
            self.inv_covariance_ = np.linalg.inv(cov)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = check_array(X)

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
        # Distances = sqrt(delta^T * S_inv * delta)
        # Use broadcasting to create a matrix of all differences
        # delta shape: (n_test, n_train, n_features)
        delta = X[:, np.newaxis, :] - self.X_train_[np.newaxis, :, :]

        # temp = (x-y)^T * S_inv
        # (n, m, j) @ (j, k) -> (n, m, k)
        # n=n_test, m=n_train, j=n_features, k=n_features
        temp = np.einsum("nmj,jk->nmk", delta, self.inv_covariance_)

        # distances_sq = temp * (x-y)
        # (n, m, k) * (n, m, k) -> (n, m)
        distances_sq = np.einsum("nmk,nmk->nm", temp, delta)

        distances_sq[distances_sq < 0] = 0
        distances = np.sqrt(distances_sq)

        return np.argsort(distances, axis=1)[:, : self.n_neighbors]

    def __euclidean_distance(self, X: np.ndarray) -> np.ndarray:
        # (A - B)^2 = A^2 + B^2 - 2AB
        a_square = np.sum(X**2, axis=1)[
            :, np.newaxis
        ]  # Shape (n_samples, 1) and force broadcast

        b_square = np.sum(self.X_train_**2, axis=1)  # Shape (n_train_samples,)

        distances = (
            -2 * X @ self.X_train_.T + a_square + b_square
        )  # Shape (n_samples, n_train_samples)

        distances[distances < 0] = 0
        distances = np.sqrt(distances)

        return np.argsort(distances, axis=1)[
            :, : self.n_neighbors
        ]  # Shape (n_samples, n_neighbors)
