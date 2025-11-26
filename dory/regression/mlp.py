import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, check_array
import pandas as pd
import logging

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class MultilayerPerceptronRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        hidden_layer_sizes: list[int] = [4],
        batch_size: int = 32,
        max_iter: int = 200,
        random_state: int = 42069,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        validation_size: float = 0.2,
        shuffle: bool = True,
        should_validate: bool = True,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.validation_size = validation_size
        self.should_validate = should_validate
        self.shuffle = shuffle

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        _, n_features = X.shape

        X = np.c_[np.ones(X.shape[0]).T, X]
        self.rng_ = np.random.default_rng(self.random_state)

        self.real_learning_rate_ = self.learning_rate / self.batch_size
        self.weights_ = self.__get_weights(n_features)
        self.velocity_ = [np.zeros_like(w) for w in self.weights_]
        self.train_errors_ = pd.DataFrame([], columns=["mse", "rmse", "mae", "mre"])
        self.val_errors_ = pd.DataFrame([], columns=["mse", "rmse", "mae", "mre"])

        X_train, X_val, Y_train, Y_val = train_test_split(
            X, y, test_size=self.validation_size, random_state=self.random_state
        )

        for epoch in range(self.max_iter):
            logger.debug(f"Epoch {epoch + 1}/{self.max_iter}")

            if self.shuffle:
                indices = np.arange(X_train.shape[0])
                self.rng_.shuffle(indices)
                X_train = X_train[indices]
                Y_train = Y_train[indices]

            for start in range(0, X_train.shape[0], self.batch_size):
                end = start + self.batch_size

                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end].reshape(
                    -1, 1
                )  # Maybe this reshape can break on classification?

                Z, U = self.__forward(X_batch)
                self.__backward(Z, U, Y_batch)

            if not self.should_validate:
                continue

            Z_train, _ = self.__forward(X_train)
            Y_pred_train = Z_train[-1]

            train_mse = np.mean((Y_train.reshape(-1, 1) - Y_pred_train) ** 2)
            train_rmse = np.sqrt(train_mse)
            train_mae = np.mean(np.abs(Y_train.reshape(-1, 1) - Y_pred_train))
            train_mre = np.mean(
                np.abs((Y_train.reshape(-1, 1) - Y_pred_train) / Y_train.reshape(-1, 1))
            )

            self.train_errors_.loc[len(self.train_errors_)] = [
                train_mse,
                train_rmse,
                train_mae,
                train_mre,
            ]

            Z_val, _ = self.__forward(X_val)
            Y_pred_val = Z_val[-1]

            val_mse = np.mean((Y_val.reshape(-1, 1) - Y_pred_val) ** 2)
            val_rmse = np.sqrt(val_mse)
            val_mae = np.mean(np.abs(Y_val.reshape(-1, 1) - Y_pred_val))
            val_mre = np.mean(
                np.abs((Y_val.reshape(-1, 1) - Y_pred_val) / Y_val.reshape(-1, 1))
            )

            self.val_errors_.loc[len(self.val_errors_)] = [
                val_mse,
                val_rmse,
                val_mae,
                val_mre,
            ]

        return self

    def __backward(self, Z: list[np.ndarray], U: list[np.ndarray], Y_batch: np.ndarray):
        deltas = []

        # Output layer
        Y_pred = Z[-1]
        little_delta = -(Y_batch - Y_pred)  # Shape: (batch_size, 1), also MSE

        deltas.append(little_delta)

        # Hidden layers delta until input layer
        for i in range(len(self.weights_) - 1, 0, -1):
            U_layer = U[i - 1]  # Shape: (batch_size, n_neurons)
            weight_next = self.weights_[i]  # Shape: (n_neurons_next, n_neurons + 1)

            # Remove 1 column from activation weights for backpropagation
            weight_next_l_no_bias = weight_next[
                :, 1:
            ]  # Shape: (n_neurons_next, n_neurons)

            delta_next = deltas[-1]  # Shape: (batch_size, n_neurons_next)

            # Hadamard product
            delta_l = (self.__derivative_activation(U_layer)) * (
                delta_next @ weight_next_l_no_bias
            )  # Shape: (batch_size, n_neurons)
            deltas.append(delta_l)

        deltas.reverse()

        for i in range(len(self.weights_), 0, -1):
            previous_layer_activation = Z[i - 1]
            delta = deltas[i - 1]  # Shape: (batch_size, n_neurons)

            gradient = delta.T @ previous_layer_activation

            self.velocity_[i - 1] = (
                self.momentum * self.velocity_[i - 1]
                - self.real_learning_rate_ * gradient
            )

            self.weights_[i - 1] += self.velocity_[i - 1]

    def __forward(
        self, X_batch: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        Z = [X_batch]  # Shape of Z[0]: (batch_size, n_features + 1)
        Us = []

        # Forward pass
        # Weights shape: (n_neurons, fan_in + 1)
        for idx, weight in enumerate(self.weights_):
            U = Z[-1] @ weight.T  # U Shape: (batch_size, n_neurons)

            if idx == len(self.weights_) - 1:
                activation = U
            else:
                activation = self.__activation(
                    U
                )  # Activation shape: (batch_size, n_neurons)

            # Set Z0 unit to 1 on hidden layers and input layer
            if idx < len(self.weights_) - 1:
                activation = np.c_[
                    np.ones(activation.shape[0]).T, activation
                ]  # Activation shape: (batch_size, n_neurons + 1)

            Z.append(activation)
            Us.append(U)

        return Z, Us

    def __get_weights(self, n_features: int) -> list[np.ndarray]:
        hidden_layer_sizes = [
            size + 1 for size in self.hidden_layer_sizes
        ]  # +1 for bias
        layer_sizes = [n_features] + hidden_layer_sizes + [1]
        weights = []

        for i in range(1, len(layer_sizes)):
            # Weights = 1 / sqrt(n_neurons) * N(0, 1)
            weight_matrix = self.rng_.normal(
                loc=0.0,
                scale=1.0,
                size=(layer_sizes[i], layer_sizes[i - 1] + 1),
            )  # Shape (n_neurons, fan_in + 1)

            weight_matrix /= np.sqrt(layer_sizes[i])

            weights.append(weight_matrix)

        # Set bias weights to zero
        for i in range(len(weights)):
            weights[i][:, 0] = 0.0

        return weights

    def __derivative_activation(self, U: np.ndarray) -> np.ndarray:
        tanh_U = np.tanh(U)
        return 1 - tanh_U**2

    def __activation(self, U: np.ndarray) -> np.ndarray:
        return np.tanh(U)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        X = np.c_[np.ones(X.shape[0]).T, X]

        Z, _ = self.__forward(X)
        Y_pred = Z[-1]

        return Y_pred.flatten()
