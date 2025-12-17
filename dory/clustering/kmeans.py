import logging
from typing import Literal
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin, check_array
from dory.utils.distances import euclidean_distance, mahalanobis_distance
import numpy as np

logger = logging.getLogger(__name__)


class KMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42069,
        metric: Literal["euclidean", "mahalanobis"] = "euclidean",
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.metric = metric


    def __get_dist(self, A: np.ndarray, B: np.ndarray):
        if self.metric == "euclidean":
            return euclidean_distance(A, B)
        elif self.metric == "mahalanobis":
            return mahalanobis_distance(A, B)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def __find_centroids(self, X: np.ndarray) -> np.ndarray:
        # Mister K-Means++
        # 1. Randomly choose the first centroid from the data points
        # 2. For each data point x, compute D(x), the distance between x
        #    and the nearest centroid that has already been chosen.
        # 3. Choose one new data point at random as a new centroid,
        #    using a weighted probability proportional to D(x)^2
        # 4. Repeat Steps 2 and 3 until k centroids have been chosen
        n_samples, _ = X.shape
        first_centroid_idx = self.random_.choice(n_samples, 1)
        centroids = X[first_centroid_idx]

        for _ in range(1, self.n_clusters):
            distances = np.min(self.__get_dist(X, centroids), axis=1)
            distances_squared = distances ** 2
            probabilities = distances_squared / distances_squared.sum()
            centroids_idx = self.random_.choice(n_samples, p=probabilities)
            centroids = np.vstack((centroids, X[centroids_idx]))

        return centroids

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        X = check_array(X)

        self.random_ = np.random.default_rng(self.random_state)

        n_samples, n_features = X.shape
        self.cluster_centers_ = self.__find_centroids(X)

        for i in range(self.max_iter):
            logger.debug(f"KMeans iteration {i + 1}")

            distances = self.__get_dist(X, self.cluster_centers_)
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array(
                [
                    X[labels == k].mean(axis=0)
                    if np.any(labels == k)
                    else self.cluster_centers_[k]
                    for k in range(self.n_clusters)
                ]
            )

            centroids_shift = self.__get_dist(self.cluster_centers_, new_centroids)
            if np.all(centroids_shift.diagonal() <= self.tol):
                logger.debug("Convergence reached.")
                break

            self.cluster_centers_ = new_centroids

        final_distances = self.__get_dist(X, self.cluster_centers_)
        labels = np.argmin(final_distances, axis=1)
        self.labels_ = labels

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        distances = self.__get_dist(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)
