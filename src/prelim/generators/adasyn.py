import warnings

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE

from .base import BaseGenerator
from .rand import Gen_randu


class Gen_adasyn(BaseGenerator):

    def __init__(self, seed=2020):
        super().__init__("adasyn", seed=seed)
        self.X_ = None
        self.used_smote_fallback_ = False

    def fit(self, X, y=None, metamodel=None):
        self.X_ = X.copy()
        return self

    def sample(self, n_samples=1):
        parss = "not majority"
        self.used_smote_fallback_ = False
        if self.X_.shape[0] > n_samples:
            warnings.warn("The required sample size is smaller than the number of observations in train")
            parss = "all"

        y_seed = np.concatenate((np.ones(self.X_.shape[0]), np.zeros(n_samples)))
        X = np.concatenate((self.X_, Gen_randu(seed=self.seed_).fit(self.X_).sample(n_samples=n_samples)))
        Xnew = None
        parknn = min(5, n_samples, self.X_.shape[0])
        y_resampled = None

        while not isinstance(Xnew, np.ndarray) and parknn <= n_samples and parknn <= self.X_.shape[0]:
            try:
                Xnew, y_resampled = ADASYN(
                    sampling_strategy=parss,
                    n_neighbors=parknn,
                    random_state=self.seed_,
                ).fit_resample(X, y_seed)
            except (ValueError, RuntimeError):
                parknn *= 2

        if not isinstance(Xnew, np.ndarray) or Xnew[y_resampled == 1, :].shape[0] < n_samples:
            parknn = min(5, n_samples, self.X_.shape[0])
            Xnew, y_resampled = SMOTE(
                sampling_strategy=parss,
                k_neighbors=parknn,
                random_state=self.seed_,
            ).fit_resample(X, y_seed)
            self.used_smote_fallback_ = True

        return Xnew[y_resampled == 1, :][0:n_samples, :]
