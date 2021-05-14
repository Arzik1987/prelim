
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV


class Gen_gmm:
    
    def __init__(self, params = {"n_components": list(range(1, 30)), 
                                 "covariance_type": ["full", "tied", "diag", "spherical"]}, cv = 5):
        self.params_ = params
        self.cv_ = cv

    def fit(self, X):
        self.model_ = GridSearchCV(GaussianMixture(), self.params_, cv = self.cv_).fit(X).best_estimator_
        return self

    def sample(self, n_samples = 1, X_new = None):
        return self.model_.sample(n_samples)[0]
    
    def my_name(self):
        return "gmm"
    
    
class Gen_gmmbic:

    def __init__(self, params = {"n_components": list(range(1, 30)), 
                                 "covariance_type": ["full", "tied", "diag", "spherical"]}, cv = 5):
        self.params_ = params

    def fit(self, X):
        # see https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html
        lowest_bic = np.infty
        for cv_type in self.params_['covariance_type']:
            for n_components in self.params_['n_components']:
                gmm = GaussianMixture(n_components = n_components, covariance_type = cv_type)
                gmm.fit(X)
                bic = gmm.bic(X)
                if bic < lowest_bic:
                    lowest_bic = bic
                    best_gmm = gmm
        
        self.model_ = best_gmm
        return self

    def sample(self, n_samples = 1):
        return self.model_.sample(n_samples)[0]

    def my_name(self):
        return "gmmbic"



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
# gmm = Gen_gmm()
# gmm.fit(x)
# df = gmm.sample(n_samples = 201)
# plt.scatter(df[:,0], df[:,1])
# 
# gmm = Gen_gmmbic()
# gmm.fit(x)
# df = gmm.sample(n_samples = 201)
# plt.scatter(df[:,0], df[:,1])
# =============================================================================



