from dory.scaler import Scaler
import pandas as pd
import numpy as np


class StandardScaler(Scaler):
    mean_: np.ndarray
    scale_: np.ndarray

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        self.mean_ = X.mean().to_numpy()
        self.scale_ = X.std(ddof=0).to_numpy()

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        arr = (X.to_numpy() - self.mean_) / self.scale_
        return arr
