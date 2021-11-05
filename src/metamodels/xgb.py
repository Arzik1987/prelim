import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import uniform, randint

    
class Meta_xgb:
    def __init__(self, params=None, cv=5, seed=2020):
        if params is None:
            params = {
                'n_estimators': randint(10, 990),
                'learning_rate': uniform(0.0001, 0.2),
                'gamma': uniform(0, 0.4),
                'max_depth': [6],
                'subsample': uniform(0.5, 0.5)
            }
        self.params_ = params
        self.cv_ = cv
        self.seed_ = seed
        self.model_ = None
        self.cvscore_ = None

    def fit(self, X, y):
        tmp = RandomizedSearchCV(XGBClassifier(nthread=1, verbosity=0, use_label_encoder=False), self.params_,
                                 random_state=self.seed_, cv=self.cv_, n_iter=50, n_jobs=1)
        tmp.fit(X, y)
        self.model_ = tmp.best_estimator_
        self.cvscore_ = tmp.best_score_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)[:, int(np.where(self.model_.classes_ == 1)[0])]
    
    def fit_score(self):
        return self.cvscore_
    
    def my_name(self):
        return "xgb"

    
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
# xgb = Meta_xgb()
# xgb.fit(x, y)
# sum(abs(xgb.predict(x) - y)) 
# sum(abs(xgb.predict_proba(x) - y))
# xgb.fit_score()
# =============================================================================

