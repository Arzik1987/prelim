import os

os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

# from src.metamodels.kriging import Meta_kriging
# from src.metamodels.nb import Meta_nb
# from src.metamodels.svm import Meta_svm
from src.metamodels.rf import Meta_rf
from src.metamodels.xgb import Meta_xgb

from src.generators.gmm import Gen_gmmbic
from src.generators.kde import Gen_kdebw
from src.generators.kdem import Gen_kdebwm
from src.generators.munge import Gen_munge
from src.generators.noise import Gen_noise
from src.generators.rand import Gen_randn, Gen_randu
from src.generators.dummy import Gen_dummy
from src.generators.smote import Gen_smote
from src.generators.adasyn import Gen_adasyn
from src.generators.rfdens import Gen_rfdens
from src.generators.vva import Gen_vva

from sklearn.tree import DecisionTreeClassifier, _tree

from src.utils.data_splitter import DataSplitter
from src.utils.data_loader import load_data

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import product
import time
import copy

if not os.path.exists('registrydt'):
    os.makedirs('registrydt')
    
# =============================================================================

# NSETS = 2
NSETS = 25
SPLITNS = list(range(0, NSETS))
# DNAMES = ["avila", "higgs7"]
DNAMES = ["occupancy", "higgs7", "electricity", "htru", "shuttle", "avila",
          "cc", "ees", "pendata", "ring", "sylva", "higgs21",
          "jm1", "saac2", "stocks", 
          "sensorless", "bankruptcy", "nomao",
          "gas", "clean2", "seizure",
          "ccpp", "seoul", "turbine", "wine", "parkinson", "dry", "anuran", "ml"]
# DSIZES = [100]
DSIZES = [100, 200, 400, 800]


def opt_param(cvres, nval):
    fit_res = np.empty((0, nval))
    for key, value in cvres.items():
        if 'split' in key:
            fit_res = np.vstack((fit_res, value))
    tmp = np.nanmean(fit_res, 0)
    return tmp

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
    gengmmbic = Gen_gmmbic() 
    genkde = Gen_kdebw()
    genkdem = Gen_kdebwm()
    genmunge = Gen_munge()
    genrandu = Gen_randu()
    genrandn = Gen_randn()
    gendummy = Gen_dummy()
    gennoise = Gen_noise()
    genadasyn = Gen_adasyn()
    gensmote = Gen_smote()
    genrfdens = Gen_rfdens()
    genvva = Gen_vva()
                                               
    metarf = Meta_rf()
    metaxgb = Meta_xgb()
    
    dt = DecisionTreeClassifier(min_samples_split = 10)
    dtcomp = DecisionTreeClassifier(max_depth = 3)
    dtcomp2 = DecisionTreeClassifier(max_leaf_nodes = 8)
    dtcv = DecisionTreeClassifier()
    dtcv2 = DecisionTreeClassifier()
    
    # get datasets
    X, y = load_data(dname)    
    ds = DataSplitter()                                                 
    ds.fit(X, y)                                                    
    ds.configure(NSETS, dsize)                                         
    X, y = ds.get_train(splitn)       
    if y.sum() == 0:
        fileres = open("registrydt/%s_%s_%s_zeros.csv" % (dname, splitn, dsize), "a")
        fileres.close()
        return                                    
    Xtest, ytest = ds.get_test(splitn) 
    Xtest = Xtest[:,(X.max(axis=0) != X.min(axis=0))]
    X = X[:,(X.max(axis=0) != X.min(axis=0))]
    
    ss = StandardScaler()                                               
    ss.fit(X)                                                       
    X = ss.transform(X) 
    Xtest = ss.transform(Xtest)
    
    defprec = 1 if y.mean() >= 0.5 else 0
    testprec = ytest.mean() if defprec == 1 else 1 - ytest.mean()
    trainprec = y.mean() if defprec == 1 else 1 - y.mean()
    filetme = open("registrydt/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
    filetme.write("testprec,%s\n" % testprec) 
    filetme.write("trainprec,%s\n" % trainprec) 
    filetme.close()        

    #===== trees                         

    fileres = open("registrydt/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
    # (1) model (2) gen (3) met (4) sctr (5) scnew (6) sctest (7) nleaves (8) time
    for k, names in zip([dt, dtcomp, dtcomp2],["dt", "dtcomp", "dtcomp2"]):
        start = time.time()
        k.fit(X, y)
        end = time.time()                                                  
        sctrain = k.score(X, y)
        sctest = k.score(Xtest, ytest)
        fileres.write(names + ",na,na,%s,nan,%s,%s,%s\n" % (sctrain, sctest, n_leaves(k), (end-start))) 
    
    # DT HPO
    par_vals = [1,2,3,4,5,6,7]
    parameters = {'max_depth': par_vals}   
    start = time.time()                           
    tmp = GridSearchCV(dtcv, parameters, refit = False).fit(X, y).cv_results_ 
    tmp = opt_param(tmp, len(par_vals))
    dtcv = DecisionTreeClassifier(max_depth = par_vals[np.argmax(tmp)])                                            
    dtcv.fit(X, y)
    end = time.time()
    sctrain = dtcv.score(X, y)
    sctest = dtcv.score(Xtest, ytest) 
    fileres.write("dtcv,na,na,%s,nan,%s,%s,%s\n" % (sctrain, sctest, n_leaves(dtcv), (end-start))) 
    filetme = open("registrydt/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
    filetme.write("dtcvsc,%s\n" % tmp[np.argmax(tmp)])
    filetme.write("dtcompsc,%s\n" % tmp[par_vals.index(3)])
    filetme.close()  
    dtcv = DecisionTreeClassifier(max_depth = dtcv.get_depth())
    
    par_vals = [2**number for number in par_vals]
    parameters = {'max_leaf_nodes': par_vals}   
    start = time.time()                           
    tmp = GridSearchCV(dtcv2, parameters, refit = False).fit(X, y).cv_results_ 
    tmp = opt_param(tmp, len(par_vals))
    dtcv2 = DecisionTreeClassifier(max_leaf_nodes = par_vals[np.argmax(tmp)])                                            
    dtcv2.fit(X, y)
    end = time.time()
    sctrain = dtcv2.score(X, y)
    sctest = dtcv2.score(Xtest, ytest) 
    fileres.write("dtcv2,na,na,%s,nan,%s,%s,%s\n" % (sctrain, sctest, n_leaves(dtcv2), (end-start)))
    filetme = open("registrydt/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
    filetme.write("dtcv2sc,%s\n" % tmp[np.argmax(tmp)])
    filetme.write("dtcomp2sc,%s\n" % tmp[par_vals.index(8)])
    filetme.close() 
    fileres.close()  
    dtcv2 = DecisionTreeClassifier(max_leaf_nodes = max(n_leaves(dtcv2),2))
    
    
    # prelim
                                                
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
              gennoise, gensmote, genadasyn, genrfdens, genkdem]:
        filetme = open("registrydt/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                      
        start = time.time()
        i.fit(X, y)
        end = time.time()
        filetme.write(i.my_name() + ",%s\n" % (end-start)) 
        filetme.close()
        
    for j in [metarf, metaxgb]: 
        filetme = open("registrydt/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                     
        start = time.time()
        j.fit(X, y)
        end = time.time()
        filetme.write(j.my_name() + ",%s\n" % (end-start)) 
        filetme.write(j.my_name() + "acc,%s\n" % j.fit_score()) 
        filetme.close()
        
    for j in [metarf, metaxgb]:
        ntrain = int(np.ceil(X.shape[0]*2/3))
        Xstrain = X[:ntrain,:].copy()
        Xstest = X[ntrain:,:].copy()
        ystrain = y[:ntrain].copy()
        ystest = y[ntrain:].copy()
        start = time.time() 
        genvva.fit(Xstrain, j)
        end = time.time()
        filetme = open("registrydt/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
        filetme.write(j.my_name() + "vva,%s\n" % (end-start))
        filetme.close()
        
        for k, names in zip([dt, dtcomp, dtcomp2, dtcv, dtcv2], ["dt", "dtcomp", "dtcomp2", "dtcv", "dtcv2"]):
            start = time.time()            
            k.fit(Xstrain, ystrain)
            sctest0 = k.score(Xstest, ystest)
            ropt = 0
            
            if genvva.will_generate():
                for r in np.linspace(0.5, 2.5, num = 5):
                    Xnew = genvva.sample(r)    
                    ynew = np.concatenate([j.predict(Xnew), ystrain]) 
                    k.fit(np.concatenate([Xnew, Xstrain]), ynew)                                    
                    sctest = k.score(Xstest, ystest)
                    if sctest > sctest0:
                        sctest0 = sctest
                        ropt = r
               
            end = time.time()
            filetme = open("registrydt/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
            filetme.write(names + j.my_name() + "vvaopt,%s\n" % (end-start))
            filetme.write(names + j.my_name() + "ropt,%s\n" % ropt)
            
            start = time.time()
            if ropt > 0:
                Xnew = Gen_vva().fit(X, j).sample(ropt)  
                ynew = j.predict(Xnew) 
                Xnew = np.concatenate([Xnew, X])
                ynew = np.concatenate([ynew, y])
            else:
                Xnew = X.copy()
                ynew = y.copy()
                
            end = time.time()
            filetme.write(names + j.my_name() + "vvagen,%s\n" % (end-start))            
            filetme.close() 
                              
            fileres = open("registrydt/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
            start = time.time()
            k.fit(Xnew, ynew)
            end = time.time()                                     
            sctrain = k.score(X, y)
            scnew = k.score(Xnew, ynew)
            sctest = k.score(Xtest, ytest)
            fileres.write(names + ",vva,%s,%s,nan,%s,%s,%s\n" % (j.my_name(), sctrain, sctest, n_leaves(k), (end-start))) 
            fileres.close()      
                
    
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
                         gennoise, gensmote, genadasyn, genrfdens, genkdem]:
        
        filetme = open("registrydt/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
        start = time.time()
        Xgen = i.sample(100000 - dsize)  
        end = time.time()
        filetme.write(i.my_name() + "gen,%s\n" % (end-start))
        filetme.close()   
        
        for j in [metarf, metaxgb]:
            filetme = open("registrydt/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
            start = time.time()
            Xnew = Xgen.copy()  
            ynew = j.predict(Xnew)
            Xnew = np.concatenate([Xnew, X])
            ynew = np.concatenate([ynew, y])
            end = time.time()
            filetme.write(i.my_name() + j.my_name() + ",%s\n" % (end-start))  
            filetme.write(i.my_name() + j.my_name() + "prec,%s\n" % (ynew.mean() if defprec == 1 else 1 - ynew.mean()))
            filetme.close()
            
            for k, names in zip([dt, dtcomp, dtcomp2, dtcv, dtcv2], ["dt", "dtcomp", "dtcomp2", "dtcv", "dtcv2"]):
                fileres = open("registrydt/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
                start = time.time()
                k.fit(Xnew, ynew)
                end = time.time()                                     
                sctrain = k.score(X, y)
                scnew = k.score(Xnew, ynew)
                sctest = k.score(Xtest, ytest)
                fileres.write(names +",%s,%s,%s,nan,%s,%s,%s\n" % (i.my_name(), j.my_name(), sctrain, sctest, n_leaves(k), (end-start))) 
                fileres.close()      
        
        

def exp_parallel():
    # pool = Pool(4)
    pool = Pool(cpu_count())
    pool.starmap_async(experiment_dt, product(SPLITNS, DNAMES, DSIZES))
    pool.close()
    pool.join()

if __name__ == "__main__":
    exp_parallel()


# experiment_dt(2, "avila", 200)
