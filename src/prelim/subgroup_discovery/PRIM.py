import numpy as np
from sklearn.utils.validation import check_X_y, check_is_fitted

class PRIM:
    def __init__(self, alpha = 0.05):
        self.alpha = alpha

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.box_ = self._get_initial_restrictions(X)
        self.N_ = len(y)
        self.Np_ = np.sum(y)
        self.mult_ = self.N_**2/(self.N_-self.Np_)
        
        highest = hgh = self._target_fun(self.Np_, self.N_)
        cont = True       
        box = self.box_.copy()
        i = 1
        while i < 100 and cont:
            hgh, cont = self._peel_one()
            if hgh > highest:
                highest = hgh
                box = self.box_.copy()
            if np.sum(self.y_) < highest*self.mult_:
                cont = False
            i = i + 1
        
        self.X_ = None
        self.y_ = None
        self.box_ = box
        self.N_ = None
        self.Np_ = None
        
        return self
    
    def _peel_one(self):
        hgh, bnd = -np.inf, -np.inf
        rn, cn = -1, -1
        cont = False
        for i in range(0, self.X_.shape[1]):
            if len(np.unique(self.X_[:,i])) > 1:
                cont = True
                bound = np.quantile(self.X_[:,i], self.alpha, interpolation = 'midpoint')
                retain = self.X_[:,i] > bound
                if np.count_nonzero(retain) == 0:
                    retain = self.X_[:,i] >= bound
                tar = self._target_fun(np.sum(self.y_[retain]), np.count_nonzero(retain))
                if tar > hgh:
                    hgh = tar
                    inds = retain
                    rn = 0
                    cn = i
                    bnd = bound
                bound = np.quantile(self.X_[:,i], 1-self.alpha, interpolation = 'midpoint')
                retain = self.X_[:,i] < bound
                if np.count_nonzero(retain) == 0:
                    retain = self.X_[:,i] <= bound
                tar = self._target_fun(np.sum(self.y_[retain]), np.count_nonzero(retain))
                if tar > hgh:
                    hgh = tar
                    inds = retain
                    rn = 1
                    cn = i
                    bnd = bound
        
        if cont:
            self.X_ = self.X_[inds]
            self.y_ = self.y_[inds]
            self.box_[rn,cn] = bnd
        
        return hgh, cont
    
    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def score(self, X , y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Check is fit had been called
        check_is_fitted(self)
        box = self.box_
        ind_in_box = np.ones(len(y), dtype = bool)
        for i in range(0, box.shape[1]):
            ind_in_box = np.logical_and(ind_in_box, np.logical_and(X[:,i] >= box[0,i], X[:,i] <= box[1,i]))

        res = (np.sum(ind_in_box)/len(y))*(np.sum(y[ind_in_box])/np.sum(ind_in_box) - np.sum(y)/len(y))            
        return res              

    def _get_initial_restrictions(self, X):
        return np.vstack((np.full(X.shape[1],-np.inf), np.full(X.shape[1],np.inf)))
    
    def _target_fun(self, npos, n):
        tar = (n/self.N_)*(npos/n - self.Np_/self.N_)
        return tar
    
    def get_nrestr(self):
        # return self.box_
        return np.count_nonzero(np.any(np.all([[self.box_ != np.inf], [self.box_!= -np.inf]], axis = 0), axis = 1))
       


# =============================================================================
# # generated data 
# 
# np.random.seed(seed=1)
# dx = np.random.random((100000,4))
# dy = ((dx > 0.3).sum(axis = 1) == 4) - 0
# 
# import time
# pr_new = PRIM()
# start = time.time()
# pr_new.fit(dx,dy)  
# end = time.time()
# print(end - start)   # ~ 0.25 s
# pr_new.score(dx, dy)
# 
# pr_new.get_nrestr()
# 
# # real dataa
# 
# import pandas as pd
# df = pd.read_csv("src\\data\\dsgc_sym.csv")
# df.head()
# dx = df.to_numpy()[9500:,0:12].copy()
# dy = df.to_numpy()[9500:,12].copy()
# pr_new = PRIM()
# pr_new.fit(dx,dy)
# pr_new.score(dx, dy)
# 
# # HPO
# 
# from sklearn.utils.estimator_checks import check_estimator
# check_estimator(PRIM())
# from sklearn.model_selection import GridSearchCV
# 
# parameters = {'alpha':[0.01, 0.1, 0.45]}
# 
# pr_w = PRIM()
# pr_001_w = PRIM(alpha = 0.01)
# pr_01_w = PRIM(alpha = 0.1)
# pr_045_w = PRIM(alpha = 0.45)
# primcv_w = GridSearchCV(pr_w, parameters)
# 
# pr_w.fit(dx, dy)
# pr_001_w.fit(dx, dy)
# pr_01_w.fit(dx, dy)
# pr_045_w.fit(dx, dy)
# primcv_w.fit(dx, dy)
# primcv_w.best_params_
# 
# pr_hpo_w = primcv_w.best_estimator_
# 
# pr_hpo_w.score(dx, dy)
# primcv_w.score(dx, dy)
# pr_w.score(dx, dy)
# pr_001_w.score(dx, dy)
# pr_01_w.score(dx, dy)
# pr_045_w.score(dx, dy) # the hyperparameters were optimized
# =============================================================================
