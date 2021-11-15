import numpy as np
from sklearn.covariance import MinCovDet


class Gen_randn:

    def __init__(self, seed=2020):
        self.seed_ = seed
        self.covariance_ = None
        self.location_ = None

    def fit(self, X, y=None, metamodel=None):
        cov = MinCovDet(random_state=self.seed_)
        cov.fit(X)
        self.covariance_ = cov.covariance_
        self.location_ = cov.location_
        return self

    def sample(self, n_samples=1):
        return np.random.multivariate_normal(self.location_, self.covariance_, n_samples)

    def my_name(self):
        return "randn"
    

class Gen_randu:

    def __init__(self):
        self.range_ = None
        self.minimum_ = None

    def fit(self, X, y=None, metamodel=None):
        self.range_ = X.max(axis=0) - X.min(axis=0)
        self.minimum_ = X.min(axis=0)
        return self

    def sample(self, n_samples=1):
        return np.random.random((n_samples, len(self.range_)))*self.range_ + self.minimum_
    
    def my_name(self):
        return "randu"


# =============================================================================
# # TEST 
# 
# mean = [0, 0]
# cov = [[1, 0], [0, 1]]
# x = np.random.multivariate_normal(mean, cov, 500)
# mean = [5, 5]
# x = np.vstack((x,np.random.multivariate_normal(mean, cov, 500)))
# import matplotlib.pyplot as plt
# plt.scatter(x[:,0], x[:,1])
# 
# nr = Gen_randn()
# nr.fit(x)
# df = nr.sample(n_samples = 201)
# plt.scatter(df[:,0], df[:,1])
# 
# ur = Gen_randu()
# ur.fit(x)
# df = ur.sample(n_samples = 201)
# plt.scatter(df[:,0], df[:,1])
# =============================================================================

    



