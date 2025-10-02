from abc import ABC, abstractmethod
import pandas as pd

class LinearRegression(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def get_parameters(self) -> pd.Series:
        pass