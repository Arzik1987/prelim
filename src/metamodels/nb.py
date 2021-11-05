import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


class Meta_nb:

    def __init__(self):
        self.model_ = None
        self.cvscore_ = None

    def fit(self, X, y):
        self.model_ = CalibratedClassifierCV(base_estimator=GaussianNB())
        self.model_.fit(X, y)
        self.cvscore_ = np.nanmean(cross_val_score(self.model_, X, y))
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, int(np.where(self.model_.classes_ == 1)[0])]
    
    def fit_score(self):
        return self.cvscore_
    
    def my_name(self):
        return "nb"
    

# =============================================================================
# # TEST 
# 
# import matplotlib.pyplot as plt
# mean = [0, 0]
# cov = [[1, 0], [0, 1]]
# x = np.random.multivariate_normal(mean, cov, 500)
# mean = [3,3]
# x = np.vstack((x,np.random.multivariate_normal(mean, cov, 500)))
# y = np.hstack((np.zeros(500), np.ones(500))).astype(int)
# plt.scatter(x[:,0], x[:,1], c = y)
# 
# nb = Meta_nb()
# nb.fit(x, y)
# sum(abs(nb.predict(x) - y)) 
# sum(abs(nb.predict_proba(x) - y))
# nb.fit_score()
# =============================================================================
