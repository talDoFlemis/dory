import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, check_is_fitted


class GaussianNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, priors: np.ndarray | None = None):
        self.priors = priors

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.vars_ = np.array([X[y == c].var(axis=0) for c in self.classes_])

        if self.priors is None:
            self.priors_ = np.array([np.mean(y == c) for c in self.classes_])
        else:
            self.priors_ = self.priors

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        likelihoods = np.array(
            [
                self.__density_fn(X, mean, var)
                for mean, var in zip(self.means_, self.vars_)
            ]
        )
        posteriors = likelihoods * self.priors_[:, np.newaxis]

        return self.classes_[np.argmax(posteriors, axis=0)]

    def __density_fn(
        self, X: np.ndarray, mean: np.ndarray, var: np.ndarray
    ) -> np.ndarray:
        exponent = np.exp(-((X - mean) ** 2) / (2 * var))
        return exponent.prod(axis=1) / np.sqrt(2 * np.pi * var).prod()
