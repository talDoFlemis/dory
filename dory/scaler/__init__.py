from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from .standard import StandardScaler

class Scaler(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series | None):
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        pass

__all__ = ["Scaler", "StandardScaler"]