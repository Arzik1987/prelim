import numpy as np
from imblearn.over_sampling import SMOTE
import warnings
from .base import BaseGenerator
from .rand import Gen_randu


class Gen_smote(BaseGenerator):

    def __init__(self, seed=2020):
        super().__init__("smote", seed=seed)
        self.X_ = None

    def fit(self, X, y=None, metamodel=None):
        self.X_ = X.copy()
        return self

    def sample(self, n_samples=1):
        parss = 'not majority'
        if self.X_.shape[0] > n_samples:
            warnings.warn("The required sample size is smaller that the number of observations in train")
            parss = 'all'
        parknn = min(5, n_samples, self.X_.shape[0])
        y = np.concatenate((np.ones(self.X_.shape[0]), np.zeros(n_samples)))
        X = np.concatenate((self.X_, Gen_randu(seed=self.seed_).fit(self.X_).sample(n_samples=n_samples)))
        X, y = SMOTE(
            sampling_strategy=parss,
            k_neighbors=parknn,
            random_state=self.seed_,
        ).fit_resample(X, y)
        return X[y == 1, :][0:n_samples, :]
