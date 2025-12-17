import numpy as np


def euclidean_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    # (A - B)^2 = A^2 + B^2 - 2AB
    a_square = np.sum(A**2, axis=1)[
        :, np.newaxis
    ]  # Shape (n_samples, 1) and force broadcast

    b_square = np.sum(B**2, axis=1)  # Shape (n_train_samples,)

    distances = -2 * A @ B.T + a_square + b_square  # Shape (n_samples, n_train_samples)

    distances[distances < 0] = 0
    distances = np.sqrt(distances)

    return distances


def mahalanobis_distance(A: np.ndarray, B: np.ndarray, cov_matrix: np.ndarray | None = None) -> np.ndarray:
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    # Distances = sqrt(delta^T * S_inv * delta)
    # Use broadcasting to create a matrix of all differences
    # delta shape: (n_A, n_B, n_features)
    # Return shape should be (n_A, n_B) to match euclidean_distance
    delta = A[:, np.newaxis, :] - B[np.newaxis, :, :]

    if cov_matrix is not None:
        cov = cov_matrix
    else:
        cov = np.cov(A.T)
        # Add a small value (regularization) for numerical stability
        # in case the matrix is singular
        cov += np.eye(cov.shape[0]) * 1e-6
    inv_covariance = np.linalg.inv(cov)

    # temp = (x-y)^T * S_inv
    # (n, m, j) @ (j, k) -> (n, m, k)
    # n=n_A, m=n_B, j=n_features, k=n_features
    temp = np.einsum("nmj,jk->nmk", delta, inv_covariance)

    # distances_sq = temp * (x-y)
    # (n, m, k) * (n, m, k) -> (n, m)
    distances_sq = np.einsum("nmk,nmk->nm", temp, delta)

    distances_sq[distances_sq < 0] = 0
    distances = np.sqrt(distances_sq)

    return distances
