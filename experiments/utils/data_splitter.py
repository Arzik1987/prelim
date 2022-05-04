from sklearn.utils.validation import check_is_fitted
import numpy as np

class DataSplitter:
    def __init__(self, seed = 2020): # set seed to None for a random seed
        self.seed_ = seed

    def fit(self, X, y):
        inds = np.random.RandomState(self.seed_).choice(np.arange(len(y)), size = len(y), replace = False)
        self.y_ = y[inds].copy()
        self.X_ = X[inds,:].copy()

    def configure(self, nparts, npoints):
        check_is_fitted(self)
        self.npoints_ = npoints
        self.startpts_ = np.linspace(0, len(self.y_)-npoints, num = nparts, endpoint = True, dtype = 'int')

    def get_train(self, ind):
        strpt = self.startpts_[ind]
        return self.X_[strpt:(strpt + self.npoints_),:].copy(), self.y_[strpt:(strpt + self.npoints_)].copy()

    def get_test(self, ind):
        strpt = self.startpts_[ind]
        yret = np.hstack((self.y_[0:strpt], self.y_[(strpt + self.npoints_):])).copy()
        Xret = np.vstack((self.X_[0:strpt,:], self.X_[(strpt + self.npoints_):,:])).copy()
        return Xret, yret


# =============================================================================
# # TEST DataSplitter
# 
# x = np.vstack((np.linspace(1, 10, 10, endpoint = True, dtype = 'int'), np.linspace(1, 10, 10, endpoint = True, dtype = 'int'))).T
# y = np.linspace(1, 10, 10, endpoint = True, dtype = 'int')
# 
# ds = DataSplitter()
# ds.fit(x, y)
# ds.configure(5, 2)
# ds.get_train(2)
# ds.get_test(2)
# =============================================================================
