import numpy as np
import sys
from sklearn.utils.validation import check_X_y, check_is_fitted

class PRIM:
    def __init__(self, alpha = 0.05, mass_min = 20, target = 'pr_auc', threshold = 1):
        self.alpha = alpha
        self.mass_min = mass_min
        self.target = target
        self.threshold = threshold

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.box_ = self._get_initial_restrictions(X)
        self.N_ = len(y)
        self.Np_ = np.sum(y)
        
        highest = hgh = self._target_fun(self.Np_, self.N_)
        cont = True
        ret_ind = 1        
        boxes = [self.box_.copy()]
        i = 1
        while self.X_.shape[0] >= self.mass_min and i < 100 and highest < self.threshold and cont:
            if hgh > highest:   # this comparison being first prevents self.X_.shape[0] < self.mass_min
                highest = hgh
                ret_ind = i
            i = i + 1
            hgh, cont = self._peel_one()
            boxes.append(self.box_.copy())
        
        self.X_ = None
        self.y_ = None
        self.box_ = None
        self.N_ = None
        self.Np_ = None
        self.boxes_ = boxes[0:ret_ind]
        
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
        return {"alpha": self.alpha, "mass_min": self.mass_min, "target": self.target}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def score(self, X , y, score_fun = None):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Check is fit had been called
        check_is_fitted(self)
        if np.sum(y) == 0: return 0
        if score_fun is None: score_fun = self.target
        if score_fun == 'pr_auc':
            prec, rec = np.empty(0), np.empty(0)
            for j in range(0, len(self.boxes_)):
                box = self.boxes_[j]
                ind_in_box = np.ones(len(y), dtype = bool)
                for i in range(0, box.shape[1]):
                    ind_in_box = np.logical_and(ind_in_box, np.logical_and(X[:,i] >= box[0,i], X[:,i] <= box[1,i]))
                if np.sum(ind_in_box) != 0:
                    prec = np.append(prec, np.sum(y[ind_in_box])/np.sum(ind_in_box))
                    rec = np.append(rec, np.sum(y[ind_in_box])/np.sum(y))
            res = -prec[0] - np.trapz(np.append(prec, prec[-1]), np.append(rec, 0))
            self.prec, self.rec = prec, rec
        else:
            box = self.boxes_[-1]
            ind_in_box = np.ones(len(y), dtype = bool)
            for i in range(0, box.shape[1]):
                ind_in_box = np.logical_and(ind_in_box, np.logical_and(X[:,i] >= box[0,i], X[:,i] <= box[1,i]))
            if score_fun == 'precision':
                res = np.sum(y[ind_in_box])/np.sum(ind_in_box)
            elif score_fun == 'wracc':
                res = (np.sum(ind_in_box)/len(y))*(np.sum(y[ind_in_box])/np.sum(ind_in_box) - np.sum(y)/len(y))
            else:
                sys.exit("The target function is unknown. It should be either wracc or precision")
                   
        return res              

    def _get_initial_restrictions(self, X):
        # return np.vstack((X.min(axis=0), X.max(axis=0)))
        return np.vstack((np.full(X.shape[1],-np.inf), np.full(X.shape[1],np.inf)))
    
    def _target_fun(self, npos, n):
        if self.target == 'precision' or self.target == 'pr_auc':
            tar = npos/n
        elif self.target == 'wracc':
            tar = (n/self.N_)*(npos/n - self.Np_/self.N_)
        else:
            sys.exit("The target function is unknown. It should be either wracc or precision")
        return tar
    
    def get_pr(self):
        return np.array2string(self.prec, separator = '_', precision = 3, max_line_width = 10000).strip('[]').replace(' ', ''), np.array2string(self.rec, separator = '_', precision = 3, max_line_width = 10000).strip('[]').replace(' ', '')



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
# pr_new.score(dx, dy, score_fun = 'wracc')
# pr_new.score(dx, dy, score_fun = 'precision')
# pr_new.score(dx, dy, score_fun = 'pr_auc')
# 
# pr_new.get_pr()
# 
# # compare different target functions
# 
# dy = np.linspace(0, 1, num = dx.shape[0])
# pr_prec = PRIM()
# pr_wracc = PRIM(target = 'wracc')
# pr_prec.fit(dx,dy)
# pr_wracc.fit(dx,dy)
# pr_prec.score(dx, dy, score_fun = 'precision')
# pr_wracc.score(dx, dy, score_fun = 'precision')
# pr_prec.score(dx, dy, score_fun = 'wracc')
# pr_wracc.score(dx, dy, score_fun = 'wracc')
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
# pr_new.score(dx, dy, score_fun = 'precision')
# 
# # HPO
# 
# from sklearn.utils.estimator_checks import check_estimator
# check_estimator(PRIM())
# 
# from sklearn.model_selection import GridSearchCV
# parameters = {'alpha':[0.01, 0.1, 0.45]}
# pr_new = PRIM()
# primcv = GridSearchCV(pr_new, parameters)
# primcv.fit(dx, dy)
# primcv.best_params_
# 
# # HPO WRAcc
# 
# parameters = {'alpha':[0.01, 0.1, 0.45]}
# 
# pr_w = PRIM(target = 'wracc')
# pr_001_w = PRIM(target = 'wracc', alpha = 0.01)
# pr_01_w = PRIM(target = 'wracc', alpha = 0.1)
# pr_045_w = PRIM(target = 'wracc', alpha = 0.45)
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




'''
import pandas as pd
from ema_workbench.analysis import prim as p, prim_util as pu

class PRIM:

    def __init__(self, threshold=1, mass_min=0.05, wracc: bool=False):
        self.threshold = threshold
        self.mass_min = mass_min
        self.wracc = wracc

    def find(self, X: pd.DataFrame, y: np.ndarray, regression=True):
        if regression:
            loc_mode = p.sdutil.RuleInductionType.REGRESSION
        else:
            loc_mode = p.sdutil.RuleInductionType.BINARY

        if self.wracc:
            obj_function = pu.PRIMObjectiveFunctions.WRACC
        else:
            obj_function = pu.PRIMObjectiveFunctions.ORIGINAL

        prim = p.PRIM(x=X, y=y, threshold=self.threshold, mode=loc_mode, obj_function=obj_function, mass_min=self.mass_min)
        box_pred = prim.find_box()

        return box_pred.box_lims
    

import time
np.random.seed(seed=1)
dx = np.random.random((1000,4))
dy = ((dx > 0.3).sum(axis = 1) == 4) - 0
dx = pd.DataFrame(dx, columns = ['x1', 'x2', 'x3' , 'x4']) 
pr = PRIM(threshold = 10)
start = time.time()
bp = pr.find(dx, dy)
end = time.time()
print(end - start) # ~2.8s

import pandas as pd
df = pd.read_csv("src\\main\\generators\\testdata.csv")
dy = df.iloc[:,6].copy().to_numpy()
dx = df.iloc[:,0:6].copy()
start = time.time()
bp = pr.find(dx,dy)
end = time.time()
print(end - start) # ~1.2s
'''