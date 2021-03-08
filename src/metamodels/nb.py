import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB


class Meta_nb:
    def __init__(self):
        return None

    def fit(self, X, y):
        self.model_ = CalibratedClassifierCV(base_estimator = GaussianNB()).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, int(np.where(self.model_.classes_ == 1)[0])]
    
   
    
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
# nb = Meta_nb()
# nb.fit(x, y)
# sum(abs(nb.predict(x) - y)) 
# sum(abs(nb.predict_proba(x) - y))
# =============================================================================
