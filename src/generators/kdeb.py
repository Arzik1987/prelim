import numpy as np
from sklearn.neighbors import NearestNeighbors


class Gen_kdeb:
    
    def __init__(self, knn=10, seed=2020):
        self.knn_ = knn
        self.seed_ = seed
        self.X_ = None
        self.dist_ = None

    def fit(self, X, y=None, metamodel=None):
        self.X_ = X.copy()
        if self.knn_ == 0:
            self.dist_ = 1
        elif self.knn_ >= X.shape[0]:
            raise RuntimeError("The dataset is too small or the knn value is too large."
                               "Number of data points must be greater than k.")
        else:
            self.dist_ = np.mean(NearestNeighbors(n_neighbors=self.knn_).fit(X).kneighbors()[0][:, self.knn_ - 1])
        return self

    def sample(self, n_samples):
        # http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        d = self.X_.shape[1]
        u = np.random.normal(0, 1, (n_samples, d + 2))  # an array of (d+2) normally distributed random variables
        den = np.sum(u**2, axis=1) ** 0.5
        u = u/den[:, None]

        return self.X_[np.random.choice(self.X_.shape[0], n_samples),:] + u[:,0:d]

    def my_name(self):
        return "kdeb"


# =============================================================================
# # TEST 
# 
# x = np.array([[0,0]])
# kdeb = Gen_kdeb(knn=0)
# kdeb.fit(x)
# df = kdeb.sample(n_samples = 1000)
# plt.scatter(df[:,0], df[:,1])
#
# mean = [0, 0]
# cov = [[1, 0], [0, 1]]
# x = np.random.multivariate_normal(mean, cov, 50)
# mean = [5, 5]
# x = np.vstack((x,np.random.multivariate_normal(mean, cov, 50)))
# import matplotlib.pyplot as plt
# plt.scatter(x[:,0], x[:,1])
#
# kdeb = Gen_kdeb()
# kdeb.fit(x)
# df = kdeb.sample(n_samples = 1000)
# plt.scatter(df[:,0], df[:,1])
# =============================================================================





