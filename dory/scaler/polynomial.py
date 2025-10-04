import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data, check_is_fitted

import logging

logger = logging.getLogger(__name__)


class NonCombinatoryExplosionPolynomialFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, degree: int = 2):
        self.degree = degree

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        X = validate_data(self, X, accept_sparse=True)
        if self.degree == 0:
            self.degree = 1
            logger.warning(
                "degree > 0 is not supported, setting degree to 1. This is a bug."
            )

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        augumented_X = X
        for degree in range(2, self.degree + 1):
            for idx in range(X.shape[1]):
                feature_to_degree = X[:, idx] ** degree
                augumented_X = np.c_[augumented_X, feature_to_degree]

        return augumented_X
