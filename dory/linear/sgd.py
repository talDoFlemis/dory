import logging

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, validate_data
import pandas as pd

logger = logging.getLogger(__name__)


class StochasticGradientDescentRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        *,
        epochs: int = 100,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        shuffle: bool = True,
        seed: int = 42069,
        initial_weights: np.ndarray | None = None,
        validation_fraction: float = 0.1,
    ):
        self.epochs = epochs
        self.epsilon = epsilon
        self.alpha = alpha
        self.shuffle = shuffle
        self.seed = seed
        self.initial_weights = initial_weights
        self.validation_fraction = validation_fraction

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = validate_data(self, X, y)
        np.random.seed(self.seed)

        X = np.c_[np.ones(X.shape[0]).T, X]
        y = y.reshape(y.size, -1)

        self.weights_ = (
            self.initial_weights.reshape(self.initial_weights.size, -1)
            if self.initial_weights is not None
            else np.random.randn(X.shape[1], 1)
        )

        self.history_ = pd.DataFrame(
            {
                "epoch": [],
                "train_loss": [],
                "val_loss": [],
            }
        )

        for epoch in range(self.epochs):
            logger.debug(f"Starting epoch {epoch}")
            error = 0
            train_loss = 0

            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_fraction,
                random_state=self.seed,
                shuffle=self.shuffle,
            )

            for i in range(X_train.shape[0]):
                logger.debug(f"Processing row {i}")
                xi = X_train[i].reshape(1, -1)
                yi = y_train[i]

                prediction = xi @ self.weights_
                error = yi - prediction
                gradient = xi.T @ error
                train_loss = error**2

                self.weights_ += self.alpha * gradient

            train_loss = error**2
            y_val_pred = X_val @ self.weights_
            val_loss = np.sum((y_val - y_val_pred) ** 2) / y_val.size

            self.history_ = pd.concat(
                [
                    self.history_,
                    pd.DataFrame(
                        {
                            "epoch": [epoch],
                            "train_loss": [train_loss],
                            "val_loss": [val_loss],
                        }
                    ),
                ],
                ignore_index=True,
            )

            logger.info(
                f"Epoch {epoch} completed. Train loss: {train_loss}, Val loss: {val_loss}, weights: {self.weights_.T}"
            )

            if train_loss < self.epsilon:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = np.c_[np.ones(X.shape[0]).T, X]
        return X @ self.weights_
