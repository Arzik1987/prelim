import numpy as np
import warnings
from .base import BaseGenerator


class Gen_perfect(BaseGenerator):

    def __init__(self, seed=2020):
        super().__init__("perfect", seed=seed)
        self.data_ = None
        
    def fit(self, X, y=None, metamodel=None):
        self.data_ = X.copy()
        return self

    def sample(self, n_samples=1):
        n_available = self.data_.shape[0]
        n_return = min(n_available, n_samples)
        if n_samples > n_available:
            warnings.warn(
                "Requested more points than available. Returning all stored rows without resampling"
            )
        if n_return == n_available:
            return self.data_.copy()
        return self.data_[self.rng_.choice(n_available, n_return, replace=False), :].copy()
