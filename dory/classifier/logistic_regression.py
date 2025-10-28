import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LogisticRegressionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.c_[np.ones(X.shape[0]).T, X]
        y = y.reshape(y.size, -1)

        self.weights_ = np.random.randn(X.shape[1], 1)

        for _ in range(self.epochs):
            linear_model = X @ self.weights_
            y_predicted = self.__sigmoid(linear_model)

            # Gradient descent
            gradients = X.T @ (y_predicted - y) / y.size
            self.weights_ -= self.learning_rate * gradients

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.c_[np.ones(X.shape[0]).T, X]
        linear_model = X @ self.weights_
        y_predicted = self.__sigmoid(linear_model)
        return (y_predicted >= 0.5).astype(int)
