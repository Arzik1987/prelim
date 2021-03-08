from src.main.generators.GaussianMixtures import GMMBIC
from src.main.subgroup_discovery.PRIM import Prim
from src.main.metamodels.RF import RF
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler #, MinMaxScaler
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

d = np.genfromtxt('src/main/sylva.csv', delimiter=',')[1:,:]
X = d[:,0:(d.shape[1] - 1)]
y = d[:,d.shape[1] - 1]
y[y == -1] = 0	

ds = DataSplitter()
ds.fit(X, y)
ds.configure(50, 200)
gen = GMMBIC()
meta = RF()
sd = Prim()
ss = StandardScaler()                     # alternatively, MinMaxScaler()
old, new = np.empty(0), np.empty(0)


for i in range(0,50):
    print(i)
    X, y = ds.get_train(i)
    Xtest, ytest = ds.get_test(i)
    sd.fit(X, y)
    old = np.append(old, sd.score(Xtest, ytest))
    ss.fit(X)                                               # scale here
    X = ss.transform(X)
    gen.fit(X)
    meta.fit(X, y)
    Xnew = gen.sample(100000)
    ynew = meta.predict(Xnew)
    Xnew = ss.inverse_transform(Xnew)                       # scale back
    sd.fit(Xnew, ynew)
    new = np.append(new, sd.score(Xtest, ytest))


cvres = np.empty(0)
for i in range(0,50):
    print(i)
    X, y = ds.get_train(i)
    Xtest, ytest = ds.get_test(i)
    parameters = {'alpha':[0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2]}
    tmp = GridSearchCV(sd, parameters).fit(X, y).best_estimator_
    cvres = np.append(cvres, tmp.score(Xtest, ytest))

np.mean(old)
np.mean(cvres)
np.mean(new)