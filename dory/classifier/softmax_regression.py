import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class SoftmaxRegressionClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def __softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        self.num_classes_ = len(self.classes_)

        y = y.reshape(y.size, -1)
        y = y.astype(int)

        # Add bias
        X = np.c_[np.ones(X.shape[0]).T, X]
        self.weights_ = np.zeros((X.shape[1], self.num_classes_))

        n_samples = len(y)
        y_one_hot = np.zeros((n_samples, self.num_classes_))
        y_one_hot[np.arange(n_samples), y] = 1

        for _ in range(self.epochs):
            scores = X @ self.weights_
            y_predicted = self.__softmax(scores)
            gradients = X.T @ (y_predicted - y_one_hot) / y.size
            self.weights_ -= self.learning_rate * gradients

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.c_[np.ones(X.shape[0]).T, X]
        scores = X @ self.weights_
        y_predicted = self.__softmax(scores)
        return np.argmax(y_predicted, axis=1)

    