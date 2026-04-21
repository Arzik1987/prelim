import numpy as np
from sklearn.covariance import MinCovDet
from .base import BaseGenerator


class Gen_randn(BaseGenerator):

    def __init__(self, seed=2020):
        super().__init__("randn", seed=seed)
        self.covariance_ = None
        self.location_ = None

    def fit(self, X, y=None, metamodel=None):
        cov = MinCovDet(random_state=self.seed_)
        cov.fit(X)
        self.covariance_ = cov.covariance_
        self.location_ = cov.location_
        return self

    def sample(self, n_samples=1):
        return self.rng_.multivariate_normal(self.location_, self.covariance_, n_samples)
    

class Gen_randu(BaseGenerator):

    def __init__(self, seed=2020):
        super().__init__("randu", seed=seed)
        self.range_ = None
        self.minimum_ = None

    def fit(self, X, y=None, metamodel=None):
        self.range_ = X.max(axis=0) - X.min(axis=0)
        self.minimum_ = X.min(axis=0)
        return self

    def sample(self, n_samples=1):
        return self.rng_.random_sample((n_samples, len(self.range_))) * self.range_ + self.minimum_
