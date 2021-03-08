import numpy as np
import warnings
from sklearn.utils.validation import check_X_y, check_is_fitted


class BI:

    def __init__(self, depth = 5, beam_size = 1, add_iter = 50):
        self.beam_size = beam_size
        self.depth = depth
        self.add_iter = add_iter
        
    def get_params(self, deep=True):
        return {"beam_size": self.beam_size, "depth": self.depth, "add_iter": self.add_iter}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.N_ = len(y)
        self.Np_ = sum(y)
        dim = X.shape[1]
        
        if np.logical_or(y.min() < 0, y.max() > 1):
            warnings.warn("The target variable takes values from outside [0,1]")
        if self.depth > dim:
            warnings.warn("Restricting depth parameter to the number of atributes in data")
        depth = min(self.depth, dim)
        
        box_init = self._get_initial_restrictions(X)
        res_box = []
        res_tab = np.empty([0,3])
        
        for i in range(0, dim):
            tmp = self._refine(X, y, box_init, i, 0)
            res_box.append(tmp[0])
            res_tab = np.concatenate((res_tab, np.array([[tmp[1], tmp[2], i]])), axis = 0)
        
        if depth > 1:
            add_iter = depth + self.add_iter               
            while add_iter > 0:
                add_iter = add_iter - 1
                
                # first we get rid of some results;
                # - all boxes, that have lower quality than the quality of top(self.beam_size) boxes
                # - all duplicated boxes (in case more than self.beam_size are left from the previous step) -
                # ideally, this should come earlier, but with large beam size and large dimensionality, could become too expensive
                # - all boxes following the self.beam_size-th one 
                if res_tab.shape[0] > self.beam_size:
                    retain = res_tab[:,0] >= np.sort(res_tab[:,0])[::-1][self.beam_size - 1]
                    if sum(retain) < len(retain):
                            res_tab = res_tab[retain]
                            res_box = [res_box[i] for i in np.where(retain)[0]]
                    if len(res_box) > 1:
                        retain = self._get_dup_boxes(res_box)
                        if sum(retain) < len(retain):
                            res_tab = res_tab[retain]
                            res_box = [res_box[i] for i in np.where(retain)[0]]
                    if res_tab.shape[0] > self.beam_size:
                        _s_ind  = res_tab[:,0].argsort()[:self.beam_size]
                        res_tab = res_tab[_s_ind]
                        res_box = [res_box[i] for i in _s_ind]
                
                # if all remaining boxes are dead ends, stop from the next iteration
                if res_tab[:,1].sum() == 0:
                    add_iter = 0
                
                # refine all promissing boxes
                for k in range(0, len(res_tab)):
                    if res_tab[k, 1] == 1:
                        res_tab[k, 1] = 0
                        inds_r = np.where(np.equal(box_init, res_box[k]).sum(axis = 0) < 2)[0]                       
                        if len(inds_r) < depth:
                            inds_r = np.array(range(0,dim))
                        inds_r = inds_r[inds_r != res_tab[k, 2]]
                        for i in inds_r:
                            tmp = self._refine(X, y, res_box[k], i, res_tab[k, 0])
                            if tmp[2] == 1:
                                res_box.append(tmp[0])
                                res_tab = np.concatenate((res_tab, np.array([[tmp[1], tmp[2], i]])), axis = 0)                
        
        winner = np.where(res_tab[:,0] == max(res_tab[:,0]))[0][0] # just return one of the winning boxes
        self.box_ = res_box[winner]
        return self
    
    def score(self, X , y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Check is fit had been called
        check_is_fitted(self)
        
        ind_in_box = np.ones(len(y), dtype = bool)
        for i in range(0, self.box_.shape[1]):
            ind_in_box = np.logical_and(ind_in_box, np.logical_and(X[:,i] >= self.box_[0,i], X[:,i] <= self.box_[1,i]))

        res = (sum(ind_in_box)/len(y))*(sum(y[ind_in_box])/sum(ind_in_box) - sum(y)/len(y))                 
        return res    

    def _get_dup_boxes(self, boxes):
        inds = np.ones(len(boxes), dtype = 'bool')
        for i in range(0, len(boxes) - 1):
            for j in range(i + 1, len(boxes)):
                if inds[j]:
                    if np.array_equal(boxes[i], boxes[j]):
                        inds[j] = False
        return inds

    def _refine(self, X, y, box, ind, start_q):
        # below numbers correspond to the row numbers in the pseudo-code description
        # from "Efficient algorithms for finding richer subgroup descriptions in 
        # numeric and nominal data" (Algorithm 3)

        ind_in_box = np.ones(self.N_, dtype = bool)
        for i in range(0, X.shape[1]):
            if not i == ind:
                ind_in_box = np.logical_and(ind_in_box, np.logical_and(X[:,i] >= box[0,i], X[:,i] <= box[1,i]))
        in_box = np.vstack((X[ind_in_box,ind], y[ind_in_box])).T
        in_box = in_box[in_box[:,1].argsort()]

        t_m, h_m = -np.inf, -np.inf                         # 3-4
        l, r = box[0,ind], box[1,ind]                       # 1
        n = in_box.shape[0]
        npos = in_box[:,1].sum()
        wracc_m = start_q                                   # 2
        
        t = np.unique(in_box[:,0])                          # define T; does sorting automatically
        itert = len(t)
        for i in range(0,itert):                            # 5
            if i != 0:
                tmp = in_box[in_box[:,0] == t[i - 1]]
                n = n - tmp.shape[0]                        # 6
                npos = npos - tmp[:,1].sum()                # 6
            h = self._wracc(n, npos, self.N_, self.Np_)     # 7
            if h > h_m:                                     # 8
                h_m = h                                     # 9
                if i == 0:                                  # 10 
                    t_m = -np.inf
                else:
                    t_m = (t[i] + t[i - 1])/2                        
                                      
            tmp = in_box[np.logical_and(in_box[:,0] >= t_m, in_box[:,0] <= t[i])]
            n_i = tmp.shape[0]
            npos_i = tmp[:,1].sum()
            wracc_i = self._wracc(n_i, npos_i, self.N_, self.Np_)
            if wracc_i > wracc_m:                           # 11 
                l = t_m                                     # 12
                if i == (itert - 1):                        # 12 
                    r = np.inf
                else:
                    r = (t[i] + t[i + 1])/2                        
                wracc_m = wracc_i                           # 13 
        box_new = box.copy()
        box_new[:,ind] = [l,r]    
        return [box_new, wracc_m, int(not wracc_m == start_q)]   
    
    def _wracc(self, n, npos, N, Np):
        return (n/N)*(npos/n - Np/N)
    
    def _get_initial_restrictions(self, X):
        # maximum = X.max(axis=0)
        # minimum = X.min(axis=0)
        # return np.vstack((minimum, maximum))
        return np.vstack((np.full(X.shape[1],-np.inf), np.full(X.shape[1],np.inf)))



# =============================================================================
# # real data
# 
# import pandas as pd
# df = pd.read_csv("src\\main\\subgroup_discovery\\dsgc_sym.csv")
# df.head()
# dx = df.to_numpy()[9500:,0:12].copy()
# dy = df.to_numpy()[9500:,12].copy()
# bi = BI(depth = 12)
# bi.fit(dx,dy)  
# bi.score(df.to_numpy()[:9500,0:12], df.to_numpy()[:9500,12]) # 0.051309052631578915
# 
# 
# # HPO
# 
# from sklearn.utils.estimator_checks import check_estimator
# check_estimator(BI())
# 
# from sklearn.model_selection import GridSearchCV
# parameters = {'depth':[1,3,5,7,9,11]}
# bi = BI()
# reds = GridSearchCV(bi, parameters)
# reds.fit(dx, dy)
# reds.best_params_
# reds.score(df.to_numpy()[:9500,0:12], df.to_numpy()[:9500,12])
# 
# bi = BI(depth = 5)
# bi.fit(dx,dy)
# bi.score(df.to_numpy()[:9500,0:12], df.to_numpy()[:9500,12])
# 
# 
# # generated data 
# 
# np.random.seed(seed=1)
# dx = np.random.random((1000,4))
# dy = ((dx > 0.3).sum(axis = 1) == 4) - 0
# 
# # 1
# import time
# bi = BI()
# start = time.time()
# bi.fit(dx,dy)  
# end = time.time()
# print(end - start)   # ~ 0.36 s
# bi.score(dx, dy) # 0.186999
# 
# # 2
# dx[:,1] = dx[:,1]*2
# 
# bi = BI(depth = 4, beam_size = 1)
# bi.fit(dx, dy)
# bi.score(dx, dy)
# 
# bi = BI(depth = 4, beam_size = 4)
# bi.fit(dx, dy)
# bi.score(dx, dy)
# 
# bi = BI(depth = 3, beam_size = 1)
# bi.fit(dx, dy)
# bi.score(dx, dy)
# =============================================================================
