import sys
import numpy as np
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott
from .base import BaseGenerator
# to choose bandwidth via CV, see, for instance, 
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_digits_kde_sampling.html#sphx-glr-auto-examples-neighbors-plot-digits-kde-sampling-py


class Gen_kdebwm(BaseGenerator):

    def __init__(self, method='silverman', seed=2020):
        super().__init__("kdebwm", seed=seed)
        if method == 'silverman':
            self.bw_method_ = bw_silverman
        elif method == 'scott':
            self.bw_method_ = bw_scott
        else:
            raise ValueError("The method must be either scott or silverman")
        self.model_ = None

    def fit(self, X, y=None, metamodel=None):
        bw = self.bw_method_(X)
        self.model_ = []
        for i in np.arange(X.shape[1]):
            self.model_.append(KernelDensity(bandwidth=bw[i]).fit(X[:,i].reshape(-1, 1)))
        return self

    def sample(self, n_samples=1):
        newdata = []
        for i in np.arange(len(self.model_)):
            newdata.append(self.model_[i].sample(n_samples, random_state=self.rng_))
        return np.hstack(newdata)
