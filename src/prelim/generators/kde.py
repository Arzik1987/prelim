
import sys
import warnings
import numpy as np
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott

# to choose bandwidth via CV, see, for instance, 
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_digits_kde_sampling.html#sphx-glr-auto-examples-neighbors-plot-digits-kde-sampling-py


class Gen_kdebw:

    def __init__(self, method='silverman'):
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
        return self.model_.sample(n_samples)
    
    def my_name(self):
        return "kdebw"

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

class Gen_kdebwhl:

    def __init__(self, method = 'silverman'):
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
        new_samples = self.model_.sample(n_samples)
        new_samples = new_samples[((new_samples <= self.limits_[1]) & (new_samples >= self.limits_[0])).all(axis = 1)]
        return new_samples
    
    def my_name(self):
        return "kdebwhl"


'''
from sklearn.model_selection import GridSearchCV
from typing import List, Union, Tuple

bw_method_silverman = 'silverman'
bw_method_scott = 'scott'

class KernelDensityCV:

    def __init__(self, bandwidth_list: Union[np.ndarray, List[float]], cv=5, hard_limits=False, sampling_multiplier: int = None):
        self.model: KernelDensity = KernelDensity()
        self.bandwidth_list: List[float] = bandwidth_list
        self.cv = cv
        self.hard_limits = hard_limits
        self._limits = ()
        assert not (hard_limits and sampling_multiplier is None)
        self.sampling_multiplier = sampling_multiplier

    def fit(self, X: pd.DataFrame, **kwargs):
        kde_params = {"bandwidth": self.bandwidth_list}
        kde_cv = GridSearchCV(self.model, kde_params, cv=self.cv)
        kde_cv.fit(X)
        self.model = kde_cv.best_estimator_
        self.model.fit(X)
        if self.hard_limits:
            self._limits = (X.min(axis=0).to_numpy(), X.max(axis=0).to_numpy())
        return self

    def sample(self, size: int) -> np.ndarray:
        if self.hard_limits:
            samples = _generate_w_hard_limits(self.model, size, self.sampling_multiplier, self._limits)
        else:
            samples = self.model.sample(size)
        return samples


def _generate_w_hard_limits(kde, n: int, sampling_multiplier: int, limits: Tuple) -> np.ndarray:
    result: np.ndarray = _cleaned_sample(kde, n, limits)
    while result.shape[0] != n:
        additional = _cleaned_sample(kde, sampling_multiplier * n, limits)
        p_needed = n - result.shape[0]
        p_available = len(additional)
        p = p_needed if p_needed <= p_available else p_available
        result = np.append(result, additional[:p], axis=0)
        assert result.shape[0] <= n
    return result


def _cleaned_sample(kde, n: int, limits: Tuple) -> np.ndarray:
    new_samples = kde.sample(n)
    new_samples = _in_bounds(new_samples, limits[0], limits[1])
    return new_samples


def _in_bounds(data: np.ndarray, minima: np.ndarray, maxima: np.ndarray) -> np.ndarray:
    logical: np.ndarray = (data <= maxima) & (data >= minima)
    logical = logical.all(axis=1)
    return data[logical]
'''