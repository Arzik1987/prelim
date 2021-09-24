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
from src.generators.munge import Gen_munge
from src.generators.noise import Gen_noise
from src.generators.rand import Gen_randn, Gen_randu
from src.generators.dummy import Gen_dummy
from src.generators.smote import Gen_smote
from src.generators.adasyn import Gen_adasyn
from src.generators.rfdens import Gen_rfdens

# import wittgenstein as lw
from src.subgroup_discovery.BI import BI

from src.utils.data_splitter import DataSplitter
from src.utils.data_loader import load_data

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import product
import time
import copy

if not os.path.exists('registrybi'):
    os.makedirs('registrybi')
    
# =============================================================================

# NSETS = 2
NSETS = 25
SPLITNS = list(range(0, NSETS))
# DNAMES = ["avila", "higgs7"]
DNAMES = ["occupancy", "higgs7", "electricity", "htru", "shuttle", "avila",
          "cc", "ees", "pendata", "ring", "sylva", "higgs21",
          "jm1", "saac2", "stocks", 
          "sensorless", "bankruptcy", 
          # "gas", "clean2", "seizure", "nomao",
          "ccpp", "seoul", "turbine", "wine", "parkinson", "dry", "anuran", "ml"]
# DSIZES = [100]
DSIZES = [100, 200, 400, 800]

def get_bi_param(nval, nattr):
    a = [ -x for x in range(-nattr, 0, np.ceil(nattr/nval).astype(int))]
    b = [ -x for x in range(-nattr, min(-nattr + nval, 0), 1)]
    res = a if len(a) > nval/2 + 1 else b
    return np.flip(res)

def opt_param(cvres, nval):
    fit_res = np.empty((0, nval))
    for key, value in cvres.items():
        if 'split' in key:
            fit_res = np.vstack((fit_res, value))
    tmp = np.nanmean(fit_res, 0)
    return tmp

def experiment_rules(splitn, dname, dsize):                                                                              
    gengmmbic = Gen_gmmbic() 
    genkde = Gen_kdebw()
    genmunge = Gen_munge()
    genrandu = Gen_randu()
    genrandn = Gen_randn()
    gendummy = Gen_dummy()
    gennoise = Gen_noise()
    genadasyn = Gen_adasyn()
    gensmote = Gen_smote()
    genrfdens = Gen_rfdens()
                                               
    metarf = Meta_rf()
    metaxgb = Meta_xgb()
    
    bi = BI()
    
    # get datasets
    X, y = load_data(dname)    
    ds = DataSplitter()                                                 
    ds.fit(X, y)                                                    
    ds.configure(NSETS, dsize)                                         
    X, y = ds.get_train(splitn)       
    if y.sum() == 0:
        fileres = open("registrybi/%s_%s_%s_zeros.csv" % (dname, splitn, dsize), "a")
        fileres.close()
        return                                    
    Xtest, ytest = ds.get_test(splitn) 
    Xtest = Xtest[:,(X.max(axis=0) != X.min(axis=0))]
    X = X[:,(X.max(axis=0) != X.min(axis=0))]
    
    defprec = 1 if y.mean() >= 0.5 else 0
    testprec = ytest.mean() if defprec == 1 else 1 - ytest.mean()
    trainprec = y.mean() if defprec == 1 else 1 - y.mean()
    filetme = open("registrybi/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
    filetme.write("testprec,%s\n" % testprec) 
    filetme.write("trainprec,%s\n" % trainprec) 
    filetme.close()        

    #===== SD                         

    fileres = open("registrybi/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
    # (1) model (2) gen (3) met (4) sctr (5) scnew (6) sctest (7) nrestr (8) time
    start = time.time()
    bi.fit(X, y)
    end = time.time()                                                   
    sctrain = bi.score(X, y)
    sctest = bi.score(Xtest, ytest)
    fileres.write("bi,na,na,%s,nan,%s,nan,%s" % (sctrain, sctest, (end-start))) 

    # BI HPO
    parsbi = get_bi_param(5, X.shape[1])
    parameters = {'depth': parsbi}                                      # params for SD with HPO
    start = time.time() 
    tmp = GridSearchCV(bi, parameters, refit = False).fit(X, y).cv_results_ 
    tmp = opt_param(tmp, len(parsbi))
    bicv = BI(depth = parsbi[np.argmax(tmp)])
    bicv.fit(X, y)
    end = time.time()
    sctrain = bicv.score(X, y)
    sctest = bicv.score(Xtest, ytest)     
    fileres.write("\nbicv,na,na,%s,nan,%s,%s,%s" % (sctrain, sctest, parsbi[np.argmax(tmp)], (end-start))) 
    fileres.close()
    
    filetme = open("registrybi/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
    filetme.write("bicvsc,%s\n" % tmp[np.argmax(tmp)])
    filetme.write("bisc,%s\n" % tmp[-1])
    # filetme.write("dtcompsc,%s\n" % tmp[par_vals.index(3)])
    filetme.close()  
    
    
    # prelim
    ss = StandardScaler()                                               
    ss.fit(X)                                                       
    Xs = ss.transform(X) 
                                                
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
              gennoise, gensmote, genadasyn, genrfdens]:
        filetme = open("registrybi/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                      
        start = time.time()
        i.fit(Xs, y)
        end = time.time()
        filetme.write(i.my_name() + ",%s\n" % (end-start)) 
        filetme.close()
        
    for j in [metarf, metaxgb]: 
        filetme = open("registrybi/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                     
        start = time.time()
        j.fit(Xs, y)
        end = time.time()
        filetme.write(j.my_name() + ",%s\n" % (end-start)) 
        filetme.write(j.my_name() + "acc,%s\n" % j.fit_score()) 
        filetme.close()
        
    for i, j in product([gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
                         gennoise, gensmote, genadasyn, genrfdens], [metarf, metaxgb]):
        filetme = open("registrybi/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")              

        start = time.time()
        Xnew = i.sample(10000 - dsize)                                                                      
        ynew = j.predict(Xnew)
        ynewp = j.predict_proba(Xnew)
        Xnew = ss.inverse_transform(Xnew)
        Xnewp = Xnew.copy()
        Xnew = np.concatenate([Xnew, X])
        ynew = np.concatenate([ynew, y])
        end = time.time()
        filetme.write(i.my_name() + j.my_name() + ",%s\n" % (end-start))  
        filetme.write(i.my_name() + j.my_name() + "prec,%s\n" % (ynew.mean() if defprec == 1 else 1 - ynew.mean()))
        filetme.close()                             
        
        fileres = open("registrybi/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
        
        start = time.time()
        bi.fit(Xnew, ynew)
        end = time.time()                                     
        sctrain = bi.score(X, y)
        scnew = bi.score(Xnew, ynew)
        sctest = bi.score(Xtest, ytest)
        fileres.write("\nbi,%s,%s,%s,%s,%s,nan,%s" % (i.my_name(), j.my_name(), sctrain, scnew, sctest, (end-start))) 
        
        start = time.time()
        bi.fit(Xnewp, ynewp)
        end = time.time()                                     
        sctrain = bi.score(X, y)
        scnew = bi.score(Xnewp, ynewp)
        sctest = bi.score(Xtest, ytest)
        fileres.write("\nbip,%s,%s,%s,%s,%s,nan,%s" % (i.my_name(), j.my_name(), sctrain, scnew, sctest, (end-start))) 
        
        start = time.time()
        bicv.fit(Xnew, ynew) 
        end = time.time()                                    
        sctrain = bicv.score(X, y)
        scnew = bicv.score(Xnew, ynew)
        sctest = bicv.score(Xtest, ytest)                                       
        fileres.write("\nbicv,%s,%s,%s,%s,%s,%s,%s" % (i.my_name(), j.my_name(), sctrain, scnew, sctest, parsbi[np.argmax(tmp)], (end-start)))
        
        start = time.time()
        bicv.fit(Xnewp, ynewp) 
        end = time.time()                                    
        sctrain = bicv.score(X, y)
        scnew = bicv.score(Xnewp, ynewp)
        sctest = bicv.score(Xtest, ytest)                                       
        fileres.write("\nbicvp,%s,%s,%s,%s,%s,%s,%s" % (i.my_name(), j.my_name(), sctrain, scnew, sctest, parsbi[np.argmax(tmp)], (end-start)))
        
        fileres.close()      
        # TODO: experiment with probabilities?
        

def exp_parallel():
    # pool = Pool(4)
    pool = Pool(cpu_count())
    pool.starmap_async(experiment_rules, product(SPLITNS, DNAMES, DSIZES))
    pool.close()
    pool.join()

if __name__ == "__main__":
    exp_parallel()

