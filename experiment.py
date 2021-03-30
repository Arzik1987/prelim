from sklearn.tree import DecisionTreeClassifier
import wittgenstein as lw
from src.generators.gmm import Gen_gmmbic
from src.subgroup_discovery.PRIM import PRIM
from src.subgroup_discovery.BI import BI
from src.metamodels.rf import Meta_rf
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler #, MinMaxScaler
from src.utils.data_splitter import DataSplitter
import numpy as np
from multiprocessing import Pool

import pandas as pd
def nunique(a, axis):
    return (np.diff(np.sort(a,axis=axis),axis=axis)!=0).sum(axis=axis)+1


def opt_param(tmp, dim):
    fit_res = np.empty((0,dim))
    for key, value in tmp.items():
        if 'split' in key:
            fit_res = np.vstack((fit_res, value))
    return np.argmax(np.nanmean(fit_res, 0))


def experiment_BI(i):
    print(i)
    df = pd.read_csv('data/new/htru/HTRU_2.csv', delimiter = ",", header = None)
    y = (df[8] == 1).astype(int).to_numpy()
    X  = df.iloc[:, : 8].to_numpy()
    y = y[~np.isnan(X).any(axis = 1)]
    X = X[~np.isnan(X).any(axis = 1)]
    X = X[:, nunique(X, 0) > 19]

    nsets = 18                                                          # number of splits
    ds = DataSplitter()             
    ds.fit(X, y)                                                        # fit data splitter
    ds.configure(nsets, 200)                                            # configure data splitter
    gen = Gen_gmmbic()                                                  
    meta = Meta_rf()
    sd = BI()
    ss = StandardScaler()                                               # alternatively, MinMaxScaler()

    X, y = ds.get_train(i)                                              # get train data
    Xtest, ytest = ds.get_test(i)                                       # get test data
    sd.fit(X, y)                                                        # fit SD baseline
    old_tmp = sd.score(Xtest, ytest)                                    # add baseline score
    
    par_vals = [2,4,6,8]
    parameters = {'depth': par_vals}                                    # params for SD with HPO
    tmp = GridSearchCV(sd, parameters, refit = False).fit(X, y).cv_results_    # fit SD with HPO
    sdcv = BI(depth = par_vals[opt_param(tmp, len(par_vals))]).fit(X, y)
    cvres_tmp = sdcv.score(Xtest, ytest)                                 # add HPO score
    
    ss.fit(X)                                                           # fit scaler
    X = ss.transform(X)                                                 # scale
    gen.fit(X)                                                          # fit generator
    meta.fit(X, y)                                                      # fit metamodel          
    Xnew = gen.sample(10000)                                            # generate points                                         # lable examples
    ynew = meta.predict_proba(Xnew)                                     # lable examples
    Xnew = ss.inverse_transform(Xnew)                                   # scale back
    sd.fit(Xnew, ynew)                                                  # fit SD REDS+    
    new_tmp = sd.score(Xtest, ytest)                                    # add REDS+ score
    
    return [old_tmp, cvres_tmp, new_tmp]


def experiment_PRIM(i):
    print(i)
    d = np.genfromtxt('src/data/sylva.csv', delimiter=',')[1:,:]        # load data
    X = d[:,0:(d.shape[1] - 1)]
    y = d[:,d.shape[1] - 1]
    y[y == -1] = 0	

    nsets = 18                                                          # number of splits
    ds = DataSplitter()             
    ds.fit(X, y)                                                        # fit data splitter
    ds.configure(nsets, 200)                                            # configure data splitter
    gen = Gen_gmmbic()                                                  
    meta = Meta_rf()
    sd = PRIM()
    ss = StandardScaler()                                               # alternatively, MinMaxScaler()

    X, y = ds.get_train(i)                                              # get train data
    Xtest, ytest = ds.get_test(i)                                       # get test data
    sd.fit(X, y)                                                        # fit SD baseline
    old_tmp = sd.score(Xtest, ytest)                                    # add baseline score
    
    par_vals = [0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2]
    parameters = {'alpha': par_vals}                                    # params for SD with HPO
    tmp = GridSearchCV(sd, parameters, refit = False).fit(X, y).cv_results_    # fit SD with HPO
    sdcv = PRIM(alpha = par_vals[opt_param(tmp, len(par_vals))]).fit(X, y)
    cvres_tmp = sdcv.score(Xtest, ytest)                                 # add HPO score
    
    ss.fit(X)                                                           # fit scaler
    X = ss.transform(X)                                                 # scale
    gen.fit(X)                                                          # fit generator
    meta.fit(X, y)                                                      # fit metamodel
    Xnew = gen.sample(100000)                                           # generate points
    ynew = meta.predict(Xnew)                                           # lable examples
    Xnew = ss.inverse_transform(Xnew)                                   # scale back
    sd.fit(Xnew, ynew)                                                  # fit SD REDS+    
    new_tmp = sd.score(Xtest, ytest)                                    # add REDS+ score
    
    return [old_tmp, cvres_tmp, new_tmp]


def experiment_DT_comp(i):
    print(i)
    d = np.genfromtxt('src/data/dsgc_sym.csv', delimiter=',')[1:,:]        # load data
    X = d[:,0:(d.shape[1] - 1)]
    y = d[:,d.shape[1] - 1]
    y[y == -1] = 0	

    nsets = 18                                                          # number of splits
    ds = DataSplitter()             
    ds.fit(X, y)                                                        # fit data splitter
    ds.configure(nsets, 400)                                            # configure data splitter
    gen = Gen_gmmbic()                                                  
    meta = Meta_rf()
    sd_comp = DecisionTreeClassifier(max_depth = 3)
    sd = DecisionTreeClassifier()
    ss = StandardScaler()                                               # alternatively, MinMaxScaler()

    X, y = ds.get_train(i)                                              # get train data
    Xtest, ytest = ds.get_test(i)                                       # get test data
    sd_comp.fit(X, y)                                                   # fit SD baseline
    old_tmp_comp = sd_comp.score(Xtest, ytest)                          # add baseline score
    sd.fit(X, y)                                                        # fit SD baseline
    old_tmp = sd.score(Xtest, ytest)                                    # add baseline score
    
    par_vals = [3, 5, 7, 9, 11, 13, 15]
    parameters = {'max_depth': par_vals}                                # params for SD with HPO
    tmp = GridSearchCV(sd, parameters, refit = False).fit(X, y).cv_results_    # fit SD with HPO
    sdcv = DecisionTreeClassifier(max_depth = par_vals[opt_param(tmp, len(par_vals))]).fit(X, y)
    cvres_tmp = sdcv.score(Xtest, ytest)                                # add HPO score
    
    ss.fit(X)                                                           # fit scaler
    X = ss.transform(X)                                                 # scale
    gen.fit(X)                                                          # fit generator
    meta.fit(X, y)                                                      # fit metamodel
    Xnew = gen.sample(100000)                                           # generate points
    ynew = meta.predict(Xnew)                                           # lable examples
    Xnew = ss.inverse_transform(Xnew)                                   # scale back
    sd_comp.fit(Xnew, ynew)     
    sd.fit(Xnew, ynew)                                                  # fit SD REDS+    
    new_tmp_comp = sd_comp.score(Xtest, ytest)                                    # add REDS+ score
    new_tmp = sd.score(Xtest, ytest)                                    # add REDS+ score
    
    return [old_tmp_comp, new_tmp_comp, old_tmp, cvres_tmp, new_tmp]


def experiment_RIPPER_comp(i):
    print(i)
    d = np.genfromtxt('src/data/dsgc_sym.csv', delimiter=',')[1:,:]        # load data
    X = d[:,0:(d.shape[1] - 1)]
    y = d[:,d.shape[1] - 1]
    y[y == -1] = 0	

    nsets = 18                                                          # number of splits
    ds = DataSplitter()             
    ds.fit(X, y)                                                        # fit data splitter
    ds.configure(nsets, 400)                                            # configure data splitter
    gen = Gen_gmmbic()                                                  
    meta = Meta_rf()
    sd = lw.RIPPER(max_rules = 10, k = 10)
    ss = StandardScaler()                                               # alternatively, MinMaxScaler()

    X, y = ds.get_train(i)                                              # get train data
    Xtest, ytest = ds.get_test(i)                                       # get test data                       # add baseline score
    sd.fit(X, y.astype(int))                                            # fit SD baseline
    old_tmp = sd.score(Xtest, ytest)                                    # add baseline score
    
    # par_vals = [2, 4, 6, 8, 10]
    # parameters = {'k': par_vals}                                        # params for SD with HPO
    # tmp = GridSearchCV(lw.RIPPER(), parameters, refit = False).fit(X, y.astype(int)).cv_results_    # fit SD with HPO
    # sdcv = lw.RIPPER(k = par_vals[opt_param(tmp, len(par_vals))]).fit(X, y.astype(int))
    # cvres_tmp = sdcv.score(Xtest, ytest)                                # add HPO score
    
    ss.fit(X)                                                           # fit scaler
    X = ss.transform(X)                                                 # scale
    gen.fit(X)                                                          # fit generator
    meta.fit(X, y)                                                      # fit metamodel
    Xnew = gen.sample(10000)                                           # generate points
    ynew = meta.predict(Xnew)                                           # lable examples
    Xnew = ss.inverse_transform(Xnew)                                   # scale back    
    sd.fit(trainset = Xnew, y = ynew.astype(int))                       # fit SD REDS+    
    new_tmp = sd.score(Xtest, ytest)                                    # add REDS+ score
    
    return [old_tmp, new_tmp]


def experiment_IREP_comp(i):
    print(i)
    d = np.genfromtxt('src/data/dsgc_sym.csv', delimiter=',')[1:,:]        # load data
    X = d[:,0:(d.shape[1] - 1)]
    y = d[:,d.shape[1] - 1]
    y[y == -1] = 0	

    nsets = 18                                                          # number of splits
    ds = DataSplitter()             
    ds.fit(X, y)                                                        # fit data splitter
    ds.configure(nsets, 400)                                            # configure data splitter
    gen = Gen_gmmbic()                                                  
    meta = Meta_rf()
    sd = lw.IREP(max_rules = 10)
    ss = StandardScaler()                                               # alternatively, MinMaxScaler()

    X, y = ds.get_train(i)                                              # get train data
    Xtest, ytest = ds.get_test(i)                                       # get test data                       # add baseline score
    sd.fit(X, y.astype(int))                                            # fit SD baseline
    old_tmp = sd.score(Xtest, ytest)                                    # add baseline score
    
    ss.fit(X)                                                           # fit scaler
    X = ss.transform(X)                                                 # scale
    gen.fit(X)                                                          # fit generator
    meta.fit(X, y)                                                      # fit metamodel
    Xnew = gen.sample(10000)                                           # generate points
    ynew = meta.predict(Xnew)                                           # lable examples
    Xnew = ss.inverse_transform(Xnew)                                   # scale back    
    sd.fit(trainset = Xnew, y = ynew.astype(int))                       # fit SD REDS+    
    new_tmp = sd.score(Xtest, ytest)                                    # add REDS+ score
    
    return [old_tmp, new_tmp]



def exp_parallel():
  pool = Pool(3)
  result = pool.map(experiment_BI, range(0,18))
  pool.close()
  pool.join()
  return(result)


if __name__ == "__main__":
  res = np.asarray(exp_parallel())
  np.savetxt('src/data/res_BI.csv', res, delimiter = ",")



