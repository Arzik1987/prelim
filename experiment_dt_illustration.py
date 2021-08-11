import os

os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

from src.metamodels.rf import Meta_rf
from src.generators.kde import Gen_kdebw
from src.generators.rand import Gen_randu
from sklearn.tree import DecisionTreeClassifier, _tree
from src.utils.data_splitter import DataSplitter
from src.utils.data_loader import load_data
from sklearn.preprocessing import StandardScaler

import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import product
import time
import copy

if not os.path.exists('registryill'):
    os.makedirs('registryill')
    
# =============================================================================

NSETS = 25
SPLITNS = list(range(0, NSETS))
DNAMES = ["anuran", "ring", "turbine"]
DSIZES = [800]


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


def experiment_dt(splitn, dname, dsize):                                                                              
    genkde = Gen_kdebw()
    genrandu = Gen_randu()
    metarf = Meta_rf()

    dt = DecisionTreeClassifier(min_samples_split = 10)
    
    # get datasets
    X, y = load_data(dname)    
    ds = DataSplitter()                                                 
    ds.fit(X, y)                                                    
    ds.configure(NSETS, dsize)                                         
    X, y = ds.get_train(splitn)       
    if y.sum() == 0:
        fileres = open("registryill/%s_%s_%s_zeros.csv" % (dname, splitn, dsize), "a")
        fileres.close()
        return                                    
    Xtest, ytest = ds.get_test(splitn) 
    Xtest = Xtest[:,(X.max(axis=0) != X.min(axis=0))]
    X = X[:,(X.max(axis=0) != X.min(axis=0))]
    
    defprec = 1 if y.mean() >= 0.5 else 0
    testprec = ytest.mean() if defprec == 1 else 1 - ytest.mean()
    trainprec = y.mean() if defprec == 1 else 1 - y.mean()
    filetme = open("registryill/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
    filetme.write("testprec,%s\n" % testprec) 
    filetme.write("trainprec,%s\n" % trainprec) 
    filetme.close()        

    #===== trees                         

    fileres = open("registryill/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
    # (1) model (2) gen (3) met (4) sctr (5) scnew (6) sctest (7) nleaves (8) time (9) ngen (10) fid1 (11) fid2
    start = time.time()
    dt.fit(X, y)
    end = time.time()                                                  
    sctrain = dt.score(X, y)
    sctest = dt.score(Xtest, ytest)
    fileres.write("dt,na,na,%s,nan,%s,%s,%s,na,na,na" % (sctrain, sctest, n_leaves(dt), (end-start))) 
    fileres.close()    
    
    # prelim
    ss = StandardScaler()                                               
    ss.fit(X)                                                       
    Xs = ss.transform(X) 
                                                
    for i in [genkde, genrandu]:
        filetme = open("registryill/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                      
        start = time.time()
        i.fit(Xs, y)
        end = time.time()
        filetme.write(i.my_name() + ",%s\n" % (end-start)) 
        filetme.close()
        
    for j in [metarf]: 
        filetme = open("registryill/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                     
        start = time.time()
        j.fit(Xs, y)
        end = time.time()
        filetme.write(j.my_name() + ",%s\n" % (end-start)) 
        filetme.write(j.my_name() + "acc,%s\n" % j.fit_score()) 
        filetme.close()
        
    for i, j in product([genkde, genrandu], [metarf]):        
        for k in [2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1024000, 2048000, 4096000]:
            filetme = open("registryill/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
            start = time.time()
            Xnew = i.sample(k - dsize)                                                                      
            ynew = j.predict(Xnew)
            Xnew = ss.inverse_transform(Xnew)  
            Xnew = np.concatenate([Xnew, X])
            ynew = np.concatenate([ynew, y])
            end = time.time()
            filetme.write(i.my_name() + j.my_name() + ",%s\n" % (end-start))  
            Xfid1 = genrandu.sample(10000) 
            yfid1 = j.predict(Xfid1)
            Xfid1 = ss.inverse_transform(Xfid1) 
            yfid2 = j.predict(ss.transform(Xtest))
            # filetme.write(i.my_name() + j.my_name() + "prec,%s\n" % (ynew.mean() if defprec == 1 else 1 - ynew.mean()))
            filetme.close()                             
        
            fileres = open("registryill/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
            start = time.time()
            dt.fit(Xnew, ynew)
            end = time.time()                                     
            sctrain = dt.score(X, y)
            scnew = dt.score(Xnew, ynew)
            sctest = dt.score(Xtest, ytest)
            fid1 = dt.score(Xfid1, yfid1)
            fid2 = dt.score(Xtest, yfid2)
            fileres.write("\ndt,%s,%s,%s,nan,%s,%s,%s,%s,%s,%s" % (i.my_name(), j.my_name(), sctrain, sctest, n_leaves(dt), (end-start), k, fid1, fid2)) 
            
            fileres.close()      
        

def exp_parallel():
    # pool = Pool(4)
    pool = Pool(cpu_count())
    pool.starmap_async(experiment_dt, product(SPLITNS, DNAMES, DSIZES))
    pool.close()
    pool.join()

if __name__ == "__main__":
    exp_parallel()

