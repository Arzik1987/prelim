import numpy as np

class Gen_rerx:
    
    def __init__(self, rho = 0.2):
        self.rho_ = rho

    def fit(self, X, y, metamodel):
        ypred = metamodel.predict(X)
        self.X_ = X[y == ypred]
        return self

    def sample(self):
        return self.X_

    def my_name(self):
        return "rerx"



# =============================================================================
# # TEST 
# 
# mean = [0, 0]
# cov = [[1, 0], [0, 1]]
# x = np.random.multivariate_normal(mean, cov, 100)
# mean = [1,1]
# x = np.vstack((x,np.random.multivariate_normal(mean, cov, 100)))
# y = np.hstack((np.zeros(100), np.ones(100))).astype(int)
# 
# from src.metamodels.xgb import Meta_xgb
# metamodel = Meta_rf()
# metamodel.fit(x, y)
# ypred = metamodel.predict_proba(x)
# np.sum(y != ypred) # everything is usually classified correctly on train
# 
# import matplotlib.pyplot as plt
# plt.scatter(x[:,0], x[:,1], c = ypred)
# 
# rerx = Gen_rerx(rho = 0.2)
# rerx.fit(x, y, metamodel)
# df = rerx.sample()
# plt.scatter(df[:,0], df[:,1])
# =============================================================================


