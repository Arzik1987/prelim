
import sys
import warnings
import numpy as np
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott
from .base import BaseGenerator

# to choose bandwidth via CV, see, for instance, 
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_digits_kde_sampling.html#sphx-glr-auto-examples-neighbors-plot-digits-kde-sampling-py


class Gen_kdebw(BaseGenerator):

    def __init__(self, method='silverman', seed=2020):
        super().__init__("kdebw", seed=seed)
        if method == 'silverman':
            self.bw_method_ = bw_silverman
        elif method == 'scott':
            self.bw_method_ = bw_scott
        else:
            raise ValueError("The method must be either scott or silverman")
        self.model_ = None

    def fit(self, X, y=None, metamodel=None):
        bw = self.bw_method_(X)
        if bw.max()/bw.min() > 10:
            warnings.warn("Bandwidths for different dimensions differ by more than order of magnitude. "
                          "Consider using z-score scaling")
        bw = bw.mean()
        self.model_ = KernelDensity(bandwidth=bw).fit(X)
        return self

    def sample(self, n_samples=1):
        return self.model_.sample(n_samples, random_state=self.rng_)

# =============================================================================
# # TEST 
# 
# mean = [0, 0]
# cov = [[1, 0], [0, 1]]
# x = np.random.multivariate_normal(mean, cov, 500)
# mean = [5, 5]
# x = np.vstack((x,np.random.multivariate_normal(mean, cov, 500)))
# x = x[((x <= [6,6]) & (x>=[-1,-1])).all(axis = 1)]
# import matplotlib.pyplot as plt
# plt.scatter(x[:,0], x[:,1])
# 
# kde = Gen_kdebw()
# kde.fit(x)
# df1 = kde.sample(n_samples = 200, hard_limits = True)
# plt.scatter(df1[:,0], df1[:,1])
# df2 = kde.sample(n_samples = 200)
# plt.scatter(df2[:,0], df2[:,1])
#       
# x[:,1] = x[:,1]*100
# kde.fit(x)
# 
# =============================================================================

class Gen_kdebwhl(BaseGenerator):

    def __init__(self, method = 'silverman', seed=2020):
        super().__init__("kdebwhl", seed=seed)
        if method == 'silverman':
            self.bw_method_ = bw_silverman
        elif method == 'scott':
            self.bw_method_ = bw_scott
        else:
            sys.exit("The method must be either scott or silverman")
        self.model_ = None
        self.limits_ = None

    def fit(self, X, y=None, metamodel=None):
        bw = self.bw_method_(X)
        if bw.max()/bw.min() > 10:
            warnings.warn("Bandwidths for different dimensions differ by more than order of magnitude. "
                          "Consider using z-score scaling")
        bw = bw.mean()
        self.model_ = KernelDensity(bandwidth=bw).fit(X)
        self.limits_ = (X.min(axis=0), X.max(axis=0))
        return self

    def sample(self, n_samples = 1):
        return self._generate_w_hard_limits(n_samples)
    
    def _generate_w_hard_limits(self, n_samples):
        sample = self._cleaned_sample(n_samples)
        mult = int(min(20, n_samples/max(sample.shape[0], 10) + 1))
        while sample.shape[0] < n_samples:
            additional = self._cleaned_sample(n_samples * mult)
            sample = np.append(sample, additional, axis = 0)
            if (sample.shape[0]/n_samples < 0.01):
                sys.exit("< 1 % of generated points are within the limits; please make sure you scaled the data")
        return sample[:n_samples]

    def _cleaned_sample(self, n_samples):
        new_samples = self.model_.sample(n_samples, random_state=self.rng_)
        new_samples = new_samples[((new_samples <= self.limits_[1]) & (new_samples >= self.limits_[0])).all(axis = 1)]
        return new_samples
