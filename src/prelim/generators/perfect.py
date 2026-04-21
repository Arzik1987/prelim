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
        res = self.data_.copy()
        if n_samples >= self.data_.shape[0]:
            # TODO is this desired behavior or do we want to abort?
            warnings.warn("Too many points are requested. Returning the complete stored set")
        else:
            res = res[self.rng_.choice(res.shape[0], n_samples, replace=False), :]
        return res
