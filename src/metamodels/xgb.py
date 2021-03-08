import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

    
class Meta_xgb:
    def __init__(self, params = {'n_estimators' : randint(10,990),
        'learning_rate' : uniform(0.0001,0.2),
        'gamma' : uniform(0,0.4),
        'max_depth' : [6],
        'subsample' : uniform(0.5,0.5)}, cv = 5, seed = 2020):
        
        self.params_ = params
        self.cv_ = cv
        self.seed_ = seed

    def fit(self, X, y):
        self.model_ = RandomizedSearchCV(XGBClassifier(nthread = 1, verbosity = 0, use_label_encoder = False), 
                                         self.params_, random_state = self.seed_,
                                         cv = self.cv_, n_iter = 50, n_jobs = 1).fit(X, y).best_estimator_
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
# xgb = Meta_xgb()
# xgb.fit(x, y)
# sum(abs(xgb.predict(x) - y)) 
# sum(abs(xgb.predict_proba(x) - y))
# =============================================================================


