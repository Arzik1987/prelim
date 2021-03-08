import numpy as np
import warnings

class Gen_perfect:
    def __init__(self):
        return None
        
    def fit(self, X):
        self.data_ = X.copy()
        return self

    def sample(self, n_samples=1):
        res = self.data_.copy()
        if n_samples >= self.data_.shape[0]:
            warnings.warn("Too many points are requested. Returning the complete stored set")
        else:
            res = res[np.random.choice(res.shape[0], n_samples, replace = False), :]
        return res


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
# pg = Gen_perfect()
# pg.fit(x)
# df = pg.sample(n_samples = 20000)
# plt.scatter(df[:,0], df[:,1])
# =============================================================================
