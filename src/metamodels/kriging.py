import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


class Meta_kriging:
    def __init__(self, seed=2020):
        self.seed_ = seed
        self.model_ = None

    def fit(self, X, y):
        self.model_ = GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=self.seed_)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, int(np.where(self.model_.classes_ == 1)[0])]
    
    def my_name(self):
        return "kriging"

    
# =============================================================================
# # TEST 
# 
# import matplotlib.pyplot as plt
# mean = [0, 0]
# cov = [[1, 0], [0, 1]]
# x = np.random.multivariate_normal(mean, cov, 500)
# mean = [5, 5]
# x = np.vstack((x,np.random.multivariate_normal(mean, cov, 500)))
# y = np.hstack((np.zeros(500), np.ones(500))).astype(int)
# plt.scatter(x[:,0], x[:,1], c = y)
# 
# gp = Meta_kriging()
# gp.fit(x, y)
# sum(abs(gp.predict(x) - y)) 
# sum(abs(gp.predict_proba(x) - y))
# =============================================================================
