import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC


class Meta_svm:
    def __init__(self, params=None, cv=5, seed=2020):
        if params is None:
            self.params = {
                "C": [0.1, 1, 10, 100],
                "gamma": [0.001, 0.01, 0.1, 1]
            }
        self.params_ = params
        self.model_ = None
        self.cv_ = cv
        self.seed_ = seed

    def fit(self, X, y):
        self.model_ = CalibratedClassifierCV(GridSearchCV(SVC(random_state=self.seed_), self.params_, cv=self.cv_))
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, int(np.where(self.model_.classes_ == 1)[0])]
    
    def my_name(self):
        return "svm"
    
    
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
# svc_cv = Meta_svm()
# svc_cv.fit(x, y)
# sum(abs(svc_cv.predict(x) - y))
# sum(abs(svc_cv.predict_proba(x) - y))
# =============================================================================

