
class Gen_dummy:
    def __init__(self):
        self.X_ = None

    def fit(self, X):
        self.X_ = X.copy()
        return self

    def sample(self, n_samples = 1):
        return self.X_.copy()



# =============================================================================
# # This generator always returns the same dataset
# 
# import numpy as np
# 
# mean = [0, 0]
# cov = [[1, 0], [0, 1]]
# x = np.random.multivariate_normal(mean, cov, 500)
# mean = [5, 5]
# x = np.vstack((x,np.random.multivariate_normal(mean, cov, 500)))
# 
# dg = Gen_dummy()
# dg.fit(x)
# dg.sample(n_samples = 201) - x
# =============================================================================
