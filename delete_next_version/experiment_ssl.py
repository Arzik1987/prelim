import os

os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

from src.metamodels.rf import Meta_rf
from src.metamodels.xgb import Meta_xgb

from sklearn.tree import DecisionTreeClassifier, _tree
import src.classification_rules.wittgenstein as lw
from src.subgroup_discovery.BI import BI
from src.subgroup_discovery.PRIM import PRIM

from src.utils.data_splitter import DataSplitter
from src.utils.data_loader import load_data

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from itertools import product
import time
import copy
import sys
  
# =============================================================================

NSETS = 25
SPLITNS = list(range(0, NSETS))
DNAMES = ["occupancy", "higgs7", "electricity", "htru", "shuttle", "avila",
          "cc", "ees", "pendata", "ring", "sylva", "higgs21",
          "jm1", "saac2", "stocks", 
          "sensorless", "bankruptcy", "nomao",
          "gas", "clean2", "seizure", "gt",
          "ccpp", "seoul", "turbine", "wine", "parkinson", "dry", "anuran", "ml"]
DSIZES = [100, 400]


def n_leaves(tree):
    tree = copy.deepcopy(tree)
    dat = tree.tree_
    nodes = range(0, dat.node_count)
    ls = dat.children_left
    rs = dat.children_right
    classes = [[list(e).index(max(e)) for e in v] for v in dat.value]
    leaves = [(ls[i] == rs[i]) for i in nodes]
    n = np.sum([1 if i == _tree.TREE_UNDEFINED else 0 for i in dat.feature])

    LEAF = -1
    for i in reversed(nodes):
        if leaves[i]:
            continue
        if leaves[ls[i]] and leaves[rs[i]] and classes[ls[i]] == classes[rs[i]]:
            ls[i] = rs[i] = LEAF
            leaves[i] = True
            n = n - 1
            
    return n

def get_new_test(Xtest, ytest, dsize):
    n = min(int(np.floor(len(ytest)/2)), 10000 - dsize)
    ytest = ytest[n:]
    Xnew = Xtest[:n,:].copy()
    Xtest = Xtest[n:,:]
    return Xtest, ytest, Xnew 

def experiment_ssl(splitn, dname, dsize):                                                                              
                                                
    metarf = Meta_rf()
    metaxgb = Meta_xgb()
    
    dtres = pd.read_csv("registrydt/%s_%s_%s.csv" % (dname, splitn, dsize), delimiter = ",", header = None)
    dtres.columns = ['alg', 'gen', 'met', 'tra', 'new', 'tes', 'nle', 'tme', 'fid']
    mln = dtres[dtres['alg'].isin(['dtcv2'])]['nle'].iloc[0]
    dt = DecisionTreeClassifier(min_samples_split = 10)
    dtcomp2 = DecisionTreeClassifier(max_leaf_nodes = 8)
    dtcv2 = DecisionTreeClassifier(max_leaf_nodes = max(mln, 2))
    
    ripper = lw.RIPPER(max_rules = 8)
    irep = lw.IREP(max_rules = 8)
    
    if splitn < 20:
        sdres = pd.read_csv("registrysd/%s_%s_%s.csv" % (dname, splitn, dsize), delimiter = ",", header = None)
        sdres.columns = ['alg', 'gen', 'met', 'tra', 'new', 'tes', 'nle', 'tme']
        dpt = sdres[sdres['alg'].isin(['bicv'])]['nle'].iloc[0]
        bicv = BI(depth = dpt)
        
        sdres = pd.read_csv("registrysd/%s_%s_%s_times.csv" % (dname, splitn, dsize), delimiter = ",", header = None)
        sdres.columns = ['alg', 'val']
        alph = sdres[sdres['alg'].isin(['primcvopt'])]['val'].iloc[0]
        primcv = PRIM(alpha = alph)
    
    # get datasets
    X, y = load_data(dname)    
    ds = DataSplitter()                                                 
    ds.fit(X, y)                                                    
    ds.configure(NSETS, dsize)                                         
    X, y = ds.get_train(splitn)       
    if y.sum() == 0:
        return                                    
    Xtest, ytest = ds.get_test(splitn) 
    Xtest = Xtest[:,(X.max(axis=0) != X.min(axis=0))]
    X = X[:,(X.max(axis=0) != X.min(axis=0))]
    
    ss = StandardScaler()                                               
    ss.fit(X)                                                       
    X = ss.transform(X) 
    Xtest = ss.transform(Xtest)
    
    Xtest, ytest, Xnew = get_new_test(Xtest, ytest, dsize)
   
    # prelim
       
    for j in [metarf, metaxgb]: 
        j.fit(X, y)

        ynew = j.predict(Xnew)
        Xnew = np.concatenate([Xnew, X])
        ynew = np.concatenate([ynew, y])
        ynewp = j.predict_proba(Xnew)
        ypredtest = j.predict(Xtest)
        
        for k, names in zip([dt, dtcomp2, dtcv2], ["dt", "dtcomp2", "dtcv2"]):
            fileres = open("registrydt/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
            start = time.time()
            k.fit(Xnew, ynew)
            end = time.time()                                     
            sctrain = k.score(X, y)
            scnew = k.score(Xnew, ynew)
            sctest = k.score(Xtest, ytest)
            fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
            fileres.write(names +",%s,%s,%s,nan,%s,%s,%s,%s\n" % ('ssl', j.my_name(), sctrain, sctest, n_leaves(k), (end-start), fidel)) 
            fileres.close()  
        
        for k, names in zip([ripper, irep],["ripper", "irep"]):
            fileres = open("registryrules/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
            start = time.time()
            k.fit(Xnew, ynew)
            end = time.time()                                     
            sctrain = k.score(X, y)
            scnew = k.score(Xnew, ynew)
            sctest = k.score(Xtest, ytest)
            fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
            fileres.write(names +",%s,%s,%s,nan,%s,%s,%s,%s\n" % ('ssl', j.my_name(), sctrain, sctest, len(k.ruleset_), (end-start), fidel)) 
            fileres.close()     
        
        if splitn < 20:
            for k, names in zip([bicv, primcv], ["bicv", "primcv"]):
                fileres = open("registrysd/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
                start = time.time()
                k.fit(Xnew, ynewp)
                end = time.time()                                     
                sctrain = k.score(X, y)
                sctest = k.score(Xtest, ytest)
                fileres.write(names +",%s,%s,%s,nan,%s,%s,%s\n" % ('ssl', j.my_name(), sctrain, sctest, k.get_nrestr(), (end-start))) 
                fileres.close() 


def exp_parallel():
    WHERE = 'registryrules/' 
    _, _, filenames = next(os.walk(WHERE))
    k = 0
    ndone = []
    for i in filenames:
        k = k + 1
        sys.stdout.write('\r' + "Loading" + "." + str(k))
        if not "times" in i and not "zeros" in i:
            tmp = pd.read_csv(WHERE + i, delimiter = ",", header = None)
            if len(tmp) != 120 and len(tmp) != 116:
                raise Exception("found an error at \%s" % (i))
            if len(tmp) == 116:
                extra = i.split(".")[0].split("_")
                ndone.append([extra[1], extra[0], extra[2]])
    ndone = pd.DataFrame(ndone, columns=['splitn','dname', 'dsize'])
    ndone = ndone.astype({'splitn': 'int32', 'dname' : 'string', 'dsize' : 'int32'})
    pool = Pool(32)
    pool.starmap_async(experiment_ssl, ndone.itertuples(index = False, name = None))
    pool.close()
    pool.join()


if __name__ == "__main__":
    exp_parallel()
    
# experiment_ssl(14, "anuran", 100)
