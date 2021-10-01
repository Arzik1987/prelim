import os

os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)


from src.metamodels.rf import Meta_rf
from src.metamodels.xgb import Meta_xgb

from src.generators.gmm import Gen_gmmbic, Gen_gmmbical
from src.generators.kde import Gen_kdebw
from src.generators.kdem import Gen_kdebwm
from src.generators.munge import Gen_munge
from src.generators.kdeb import Gen_kdeb
# from src.generators.noise import Gen_noise
from src.generators.rand import Gen_randn, Gen_randu
from src.generators.dummy import Gen_dummy
from src.generators.smote import Gen_smote
from src.generators.adasyn import Gen_adasyn
from src.generators.rfdens import Gen_rfdens
from src.generators.vva import Gen_vva
from src.generators.rerx import Gen_rerx

from src.subgroup_discovery.BI import BI
from src.subgroup_discovery.PRIM import PRIM

from src.utils.data_splitter import DataSplitter
from src.utils.data_loader import load_data

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import product
import time
# import copy

dirnme = 'registrysd'
if not os.path.exists(dirnme):
    os.makedirs(dirnme)
    
# =============================================================================

# NSETS = 2
NSETS = 25
SPLITNS = list(range(0, 20))
# DNAMES = ["avila", "higgs7"]
DNAMES = ["occupancy", "higgs7", "electricity", "htru", "shuttle", "avila",
          "cc", "ees", "pendata", "ring", "sylva", "higgs21",
          "jm1", "saac2", "stocks", 
          "sensorless", "bankruptcy", "gt", 
           "gas", "clean2", "seizure", "nomao",
          "ccpp", "seoul", "turbine", "wine", "parkinson", "dry", "anuran", "ml"]
# DSIZES = [100]
DSIZES = [100, 400]


def opt_param(cvres, nval):
    fit_res = np.empty((0, nval))
    for key, value in cvres.items():
        if 'split' in key:
            fit_res = np.vstack((fit_res, value))
    tmp = np.nanmean(fit_res, 0)
    return tmp

def get_bi_param(nval, nattr):
    nattr = min(15, nattr)
    a = [ -x for x in range(-nattr, 0, np.ceil(nattr/nval).astype(int))]
    b = [ -x for x in range(-nattr, min(-nattr + nval, 0), 1)]
    res = a if len(a) > nval/2 + 1 else b
    # if np.min(res) > 5:
    #     res = res[1:]
    #     res.append(5)
    return np.flip(res)


def experiment_bi(splitn, dname, dsize):                                                                              
    gengmmbic = Gen_gmmbic() 
    genkde = Gen_kdebw()
    genkdeb = Gen_kdeb()
    genkdem = Gen_kdebwm()
    genmunge = Gen_munge()
    genrandu = Gen_randu()
    genrandn = Gen_randn()
    gendummy = Gen_dummy()
    genadasyn = Gen_adasyn()
    gensmote = Gen_smote()
    genrfdens = Gen_rfdens()
    genvva = Gen_vva()
    gengmmbical = Gen_gmmbical()
    genrerx = Gen_rerx()
                                               
    metarf = Meta_rf()
    metaxgb = Meta_xgb()
    
    bi = BI()
    prim = PRIM()
    
    # get datasets
    X, y = load_data(dname)    
    ds = DataSplitter()                                                 
    ds.fit(X, y)                                                    
    ds.configure(NSETS, dsize)                                         
    X, y = ds.get_train(splitn)       
    if y.sum() == 0:
        fileres = open(dirnme + "/%s_%s_%s_zeros.csv" % (dname, splitn, dsize), "a")
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
    filetme = open(dirnme + "/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
    filetme.write("testprec,%s\n" % testprec) 
    filetme.write("trainprec,%s\n" % trainprec) 
    filetme.close()        
   
    # BI HPO
    fileres = open(dirnme + "/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
    parsbi = get_bi_param(5, X.shape[1])
    parameters = {'depth': parsbi}                                      # params for SD with HPO
    start = time.time() 
    tmp = GridSearchCV(BI(), parameters, refit = False).fit(X, y).cv_results_ 
    tmp = opt_param(tmp, len(parsbi))
    bicv = BI(depth = parsbi[np.argmax(tmp)])
    bicv.fit(X, y)
    end = time.time()
    sctrain = bicv.score(X, y)
    sctest = bicv.score(Xtest, ytest)     
    fileres.write("bicv,na,na,%s,nan,%s,%s,%s\n" % (sctrain, sctest, bicv.get_nrestr(), (end-start))) 
    filetme = open(dirnme + "/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
    filetme.write("bicvopt,%s\n" % parsbi[np.argmax(tmp)])
    filetme.close()  
    bicv= BI(depth = bicv.get_nrestr())
    
    # PRIM HPO
    par_vals = [0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2]
    parameters = {'alpha': par_vals}                                      # params for SD with HPO
    start = time.time() 
    tmp = GridSearchCV(PRIM(), parameters, refit = False).fit(X, y).cv_results_ 
    tmp = opt_param(tmp, len(par_vals))
    primcv = PRIM(alpha = par_vals[np.argmax(tmp)])
    primcv.fit(X, y)
    end = time.time()
    sctrain = primcv.score(X, y)
    sctest = primcv.score(Xtest, ytest)     
    fileres.write("primcv,na,na,%s,nan,%s,%s,%s\n" % (sctrain, sctest, primcv.get_nrestr(), (end-start))) 
    fileres.close()
    filetme = open(dirnme + "/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
    filetme.write("primcvopt,%s\n" % par_vals[np.argmax(tmp)])
    filetme.close() 
    
    
    # prelim
                                                
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
              gengmmbical, gensmote, genadasyn, genrfdens, genkdem, genkdeb]:
        filetme = open(dirnme + "/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                      
        start = time.time()
        i.fit(X, y)
        end = time.time()
        filetme.write(i.my_name() + ",%s\n" % (end-start)) 
        filetme.close()
        
    for j in [metarf, metaxgb]: 
        filetme = open(dirnme + "/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                     
        start = time.time()
        j.fit(X, y)
        end = time.time()
        # ypredtest = j.predict(Xtest)
        filetme.write(j.my_name() + ",%s\n" % (end-start)) 
        filetme.write(j.my_name() + "acc,%s\n" % j.fit_score()) 
        filetme.close()
        
        # rerx generator
        genrerx.fit(X, y, j)
        Xnew = genrerx.sample()  
        # ynew = j.predict(Xnew)
        ynewp = j.predict_proba(Xnew)
        filetme = open(dirnme + "/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                      
        filetme.write("rerxndel,%s\n" % (dsize - len(y))) 
        filetme.close()
        
        for k, names in zip([bicv, primcv], ["bicv", "primcv"]):
            
            fileres = open(dirnme + "/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
            start = time.time()
            k.fit(Xnew, ynewp)
            end = time.time()                                     
            sctrain = k.score(X, y)
            # scnew = k.score(Xnew, ynewp)
            sctest = k.score(Xtest, ytest)
            fileres.write(names +",%s,%s,%s,nan,%s,%s,%s\n" % (genrerx.my_name(), j.my_name(), sctrain, sctest, k.get_nrestr(), (end-start))) 
            fileres.close()  
        
        # vva generator
        ntrain = int(np.ceil(X.shape[0]*2/3))
        Xstrain = X[:ntrain,:].copy()
        Xstest = X[ntrain:,:].copy()
        ystrain = y[:ntrain].copy()
        ystest = y[ntrain:].copy()
        start = time.time() 
        genvva.fit(Xstrain, j)
        end = time.time()
        filetme = open(dirnme + "/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
        filetme.write(j.my_name() + "vva,%s\n" % (end-start))
        filetme.close()
        
        for k, names in zip([bicv, primcv], ["bicv", "primcv"]):
            start = time.time()            
            k.fit(Xstrain, j.predict_proba(Xstrain))
            sctest0 = k.score(Xstest, ystest)
            ropt = 0
            
            if genvva.will_generate():
                for r in np.linspace(0.5, 2.5, num = 5):
                    Xnew = genvva.sample(r)    
                    Xnew = np.concatenate([Xnew, Xstrain])
                    ynew = j.predict_proba(Xnew) 
                    k.fit(Xnew, ynew)                                    
                    sctest = k.score(Xstest, ystest)
                    if sctest > sctest0:
                        sctest0 = sctest
                        ropt = r
               
            end = time.time()
            filetme = open(dirnme + "/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
            filetme.write(names + j.my_name() + "vvaopt,%s\n" % (end-start))
            filetme.write(names + j.my_name() + "ropt,%s\n" % ropt)
            
            start = time.time()
            if ropt > 0:
                Xnew = Gen_vva().fit(X, j).sample(ropt)   
                Xnew = np.concatenate([Xnew, X])
            else:
                Xnew = X.copy()
            
            end = time.time()
            filetme.write(names + j.my_name() + "vvagen,%s\n" % (end-start))            
            filetme.close() 
            
            fileres = open(dirnme + "/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
            start = time.time()
            k.fit(Xnew, j.predict_proba(Xnew))
            end = time.time()                                     
            sctrain = k.score(X, y)
            # scnew = k.score(Xnew, ynew)
            sctest = k.score(Xtest, ytest)
            fileres.write(names + ",vva,%s,%s,nan,%s,%s,%s\n" % (j.my_name(), sctrain, sctest, k.get_nrestr(), (end-start))) 
            fileres.close()    
                
    
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
                         gengmmbical, gensmote, genadasyn, genrfdens, genkdem, genkdeb]:
        
        filetme = open(dirnme + "/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
        start = time.time()
        Xgen = i.sample(10000 - dsize)  
        end = time.time()
        filetme.write(i.my_name() + "gen,%s\n" % (end-start))
        filetme.close()   
        
        for j in [metarf, metaxgb]:
            filetme = open(dirnme + "/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
            start = time.time()
            Xnew = Xgen.copy()  
            # ynew = j.predict(Xnew)
            Xnew = np.concatenate([Xnew, X])
            # ynew = np.concatenate([ynew, y])
            ynewp = j.predict_proba(Xnew)
            end = time.time()
            filetme.write(i.my_name() + j.my_name() + ",%s\n" % (end-start))  
            filetme.write(i.my_name() + j.my_name() + "prec,%s\n" % (ynew.mean() if defprec == 1 else 1 - ynew.mean()))
            filetme.close()
            
            for k, names in zip([bicv, primcv], ["bicv", "primcv"]):
                
                fileres = open(dirnme + "/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
                start = time.time()
                k.fit(Xnew, ynewp)
                end = time.time()                                     
                sctrain = k.score(X, y)
                # scnew = k.score(Xnew, ynew)
                sctest = k.score(Xtest, ytest)
                fileres.write(names +",%s,%s,%s,nan,%s,%s,%s\n" % (i.my_name(), j.my_name(), sctrain, sctest, k.get_nrestr(), (end-start))) 
                fileres.close() 
        
        

def exp_parallel():
    # pool = Pool(4)
    pool = Pool(cpu_count())
    pool.starmap_async(experiment_bi, product(SPLITNS, DNAMES, DSIZES))
    pool.close()
    pool.join()

if __name__ == "__main__":
    exp_parallel()


# experiment_bi(0, "avila", 100)
