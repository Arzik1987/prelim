import numpy as np
from .base import BaseGenerator


class Gen_noise(BaseGenerator):

    def __init__(self, scale=0.3, seed=2020):
        super().__init__("noise", seed=seed)
        self.scale_ = scale

    def fit(self, X, y=None, metamodel=None):
        self.data_ = X.copy()
        self.data_ = self.data_.astype(float)
        return self

    def sample(self, n_samples=1):
        mod_data = self.data_.copy()
        for col in range(0,mod_data.shape[1]):
            mindist = min(np.diff(np.unique(mod_data[:,col])))*self.scale_
            mod_data[:,col] = mod_data[:,col] + self.rng_.uniform(-mindist, mindist, len(mod_data[:,col]))
        return mod_data
