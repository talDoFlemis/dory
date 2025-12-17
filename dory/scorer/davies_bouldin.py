from typing import Literal
from dory.utils.distances import euclidean_distance, mahalanobis_distance
import numpy as np


def davies_bouldin_scorer(
    distance: Literal["euclidean", "mahalanobis"] = "euclidean",
):
    def calculate_index(X: np.ndarray, labels: np.ndarray):
        distance_func = None
        if distance == "euclidean":
            distance_func = euclidean_distance
        elif distance == "mahalanobis":
            cov = np.cov(X.T)
            cov += np.eye(cov.shape[0]) * 1e-6
            distance_func = lambda A, B: mahalanobis_distance(A, B, cov)
        else:
            raise ValueError(f"Unsupported distance metric: {distance}")

        n_cluster = len(np.bincount(labels))
        cluster_k = [X[labels == k] for k in range(n_cluster)]
        centroids = [np.mean(k, axis=0) for k in cluster_k]
        variances = [
            np.mean([distance_func(p, centroids[i]) for p in k])
            for i, k in enumerate(cluster_k)
        ]
        db = []
        for i in range(n_cluster):
            dbij = []
            for j in range(n_cluster):
                if j != i:
                    dbij.append(
                        (variances[i] + variances[j])
                        / distance_func(centroids[i], centroids[j])
                    )
            db.append(max(dbij))

        dbi = np.mean(db)
        return dbi

    return calculate_index
