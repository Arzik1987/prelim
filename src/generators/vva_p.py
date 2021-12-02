import numpy as np
from itertools import cycle


class Gen_vva:
    
    def __init__(self, rho=0.2):
        self.rho_ = rho
        self.generate_ = None
        self.nn_ = None
        self.ordinds_ = None
        self.Xbound_ = None
        self.dim_ = None
        self.trainn_ = None

    def fit(self, X, metamodel, y=None):
        self.generate_ = True
        self.dim_ = X.shape[1]
        self.trainn_ = X.shape[0]
        Xtrain = X.copy()
        y = metamodel.predict_proba(Xtrain)[:, int(np.where(metamodel.classes_ == 1)[0])] - 0.5
        if sum(y < 0) == 0 or sum(y > 0) == 0:
            self.generate_ = False
            return self
        
        inds = np.concatenate((np.where(y == max(y[y < 0]))[0], np.where(y == min(y[y > 0]))[0]))
        Xbound = Xtrain[inds,:].copy()
        ybound = y[inds].copy()
        Xtrain = np.delete(Xtrain, inds, axis=0)
        y = np.delete(y, inds)
        nrest = int(np.ceil(X.shape[0]*self.rho_ - len(inds)))
        if nrest > 0:

            # TODO Enable this part by parameter?
            # commented part is more fair to duplicated scores, but can result in not enough boundary points
            # thr = np.sort(abs(y))[nrest]
            # if thr == max(abs(y)):
            #     thr = thr - 1e-8
            # inds = np.where(abs(y) <= thr)[0]

            inds = np.argsort(abs(y))[:nrest]
            Xbound = np.concatenate((Xbound, Xtrain[inds,:].copy()), axis = 0)
            ybound = np.concatenate((ybound, y[inds].copy()))
            
        self._find_neighbours(Xbound, ybound)
        return self
    
    def _find_neighbours(self, X, y):
        Xpos = X[y > 0, :]
        Xneg = X[y < 0, :]
        nnpos, distpos = self._nearest_neighbours(Xpos, Xneg)
        nnneg, distneg = self._nearest_neighbours(Xneg, Xpos)
        self.Xbound_ = np.concatenate((Xpos, Xneg), axis = 0)
        self.nn_ = np.concatenate((nnpos + len(nnpos), nnneg))
        self.ordinds_ = np.argsort(np.concatenate((distpos, distneg)))
    
    def _nearest_neighbours(self, X1, X2): # https://stackoverflow.com/questions/15363419/finding-nearest-items-across-two-lists-arrays-in-python/15366296
        X1, X2 = map(np.asarray, (X1, X2))
        nearest_neighbour = np.empty((len(X1),), dtype=np.intp)
        dist = np.empty((len(X1),), dtype=np.float32)
        for j, xj in enumerate(X1):
            idx = np.argmin(np.sum((X2 - xj)**2, axis=1))
            nearest_neighbour[j] = idx
            dist[j] = np.sqrt(np.sum((X2[idx] - xj)**2)) 

        return nearest_neighbour, dist

    def sample(self, r):  # r is from 0 to 2.5
        if r < 0 or r > 2.5:
            raise ValueError("the boundaries for r defined in the paper are from 0 to 2.5")
        if r == 0 or self.generate_ is False:
            return np.empty((0, self.dim_))
        
        ngen = int(np.ceil(self.trainn_*r)) 
        thetas = np.random.uniform(0, 1, (ngen, self.dim_))
        pool = cycle(self.ordinds_)
        k = 0
        newpts = []
        while k < ngen:
            indx = next(pool)
            indnn = self.nn_[indx]
            theta = thetas[k, :]
            newpts.append(self.Xbound_[indx, :]*theta + self.Xbound_[indnn, :]*(1 - theta))
            k = k + 1
        
        return np.vstack(newpts)

    def my_name(self):
        return "vva"
    
    def will_generate(self):
        return self.generate_


# =============================================================================
# # TEST 
# 
# mean = [0, 0]
# cov = [[1, 0], [0, 1]]
# x = np.random.multivariate_normal(mean, cov, 100)
# mean = [3,3]
# x = np.vstack((x,np.random.multivariate_normal(mean, cov, 100)))
# y = np.hstack((np.zeros(100), np.ones(100))).astype(int)
# 
# from src.metamodels.rf import Meta_rf
# metamodel = Meta_rf()
# metamodel.fit(x, y)
# ypred = metamodel.predict_proba(x)
# 
# import matplotlib.pyplot as plt
# plt.scatter(x[:,0], x[:,1], c = ypred)
# 
# vva = Gen_vva(rho = 0.2)
# vva.fit(x, metamodel) # uncomment predict_proba in the generator or replace with our metamodel
# df = vva.sample(r = 1)
# df = np.vstack([x, df])
# ypred = metamodel.predict_proba(df)
# plt.scatter(df[:,0], df[:,1], c= ypred)
# =============================================================================


