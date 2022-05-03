import sys
import numpy as np
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.bandwidths import bw_silverman, bw_scott
# to choose bandwidth via CV, see, for instance, 
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_digits_kde_sampling.html#sphx-glr-auto-examples-neighbors-plot-digits-kde-sampling-py


class Gen_kdebwm:

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
        self.model_ = []
        for i in np.arange(X.shape[1]):
            self.model_.append(KernelDensity(bandwidth=bw[i]).fit(X[:,i].reshape(-1, 1)))
        return self

    def sample(self, n_samples=1):
        newdata = []
        for i in np.arange(len(self.model_)):
            newdata.append(self.model_[i].sample(n_samples))
        return np.hstack(newdata)
    
    def my_name(self):
        return "kdebwm"


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
# kde = Gen_kdebwm()
# kde.fit(x)
# df1 = kde.sample(n_samples = 1000)
# plt.scatter(df1[:,0], df1[:,1])
#       
# x[:,1] = x[:,1]*100
# kde.fit(x)
# df2 = kde.sample(n_samples = 1000)
# plt.scatter(df2[:,0], df2[:,1])
# =============================================================================


