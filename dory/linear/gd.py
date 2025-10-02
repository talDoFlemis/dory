import logging
from sklearn.base import RegressorMixin, BaseEstimator
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import ShuffleSplit

logger = logging.getLogger(__name__)


class GradientDescentRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        *,
        epochs: int = 100,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        shuffle: bool = True,
        seed: int = 42069,
        initial_weights: np.ndarray | None = None,
        validation_fraction: float = 0.1
    ):
        self.epochs = epochs
        self.epsilon = epsilon
        self.alpha = alpha
        self.shuffle = shuffle
        self.seed = seed
        self.initial_weights = initial_weights
        self.validation_fraction = validation_fraction

    def fit(self, X: np.ndarray, y: np.ndarray):
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

        splitter = ShuffleSplit(test_size=self.validation_fraction, random_state=self.seed)

        for i in range(self.epochs):
            logger.debug(f"Starting epoch {i}")

            train_idx, val_idx = next(splitter.split(X, y))
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            n = X_train.size

            if self.shuffle:
                indices = np.arange(X_train.shape[0])
                np.random.shuffle(indices)
                X_train = X_train[indices]
                y_train = y_train[indices]

            y_pred = X_train @ self.weights_
            error = y_train - y_pred
            df_dw = 1 / n * (X_train.T @ error)
            df_dw = df_dw.reshape(len(df_dw), -1)

            train_mse = np.sum((error ** 2), axis=0) / len(X_train)

            # Validation prediction and loss
            y_val_pred = X_val @ self.weights_
            val_mse = np.sum(((y_val - y_val_pred) ** 2), axis=0) / len(X_val)

            self.weights_ += self.alpha * df_dw

            self.history_ = pd.concat(
                [
                    self.history_,
                    pd.DataFrame(
                        {
                            "epoch": [i],
                            "train_loss": [train_mse],
                            "val_loss": [val_mse],
                        }
                    ),
                ],
                ignore_index=True,
            )

            if sum(train_mse) < self.epsilon:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        X = np.c_[np.ones(X.shape[0]).T, X]
        return X @ self.weights_
