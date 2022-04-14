import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class Meta_rf_bal:

    def __init__(self, params=None, cv=5, seed=2020):
        if params is None:
            params = {"max_features": [2, "sqrt", None]}
        self.params_ = params
        self.cv_ = cv
        self.seed_ = seed
        self.model_ = None
        self.cvscore_ = None

    def fit(self, X, y):
        tmp = GridSearchCV(RandomForestClassifier(random_state=self.seed_, class_weight = 'balanced'), self.params_, cv=self.cv_, scoring = 'balanced_accuracy')
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
        return "rfb"


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
# rfb = Meta_rf_bal()
# rfb.fit(x, y)
# sum(abs(rfb.predict(x) - y))
# sum(abs(rfb.predict_proba(x) - y))
# rfb.fit_score()
# =============================================================================
