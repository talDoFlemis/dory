import logging
import numpy as np
from sklearn.dummy import check_random_state
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)

class DoryKFoldCrossValidator(BaseCrossValidator):
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state


    def split(self, X, y= None, groups=None):
        X = np.asarray(X)

        n_samples = len(X)

        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have number of splits n_splits={self.n_splits} greater than the number of samples: n_samples={n_samples}."
            )

        indices = np.arange(n_samples)

        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            logger.debug(f"Train indices: {train_index}, Test indices: {test_index}")
            yield train_index, test_index


    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1
        current = 0

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop

            
    def _iter_test_masks(self, X, y=None, groups=None):
        n_samples = len(X)

        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def get_n_splits(self, X: np.ndarray | None = None, y: np.ndarray | None = None, groups=None):
        return self.n_splits