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

from sklearn.linear_model import LogisticRegression

from src.utils.data_splitter import DataSplitter
from src.utils.data_loader import load_data

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import product
import time
import copy

if not os.path.exists('registrylr'):
    os.makedirs('registrylr')
    
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

def get_coef(model):
    tmp = np.array2string(model.coef_, precision = 3, separator='_', sign = '-',  max_line_width = 10)
    tmp = tmp.replace("\n", "")
    tmp = tmp.replace(" ", "")
    tmp = tmp.replace("[", "")
    tmp = tmp.replace("]", "")
    return tmp

def experiment_lr(splitn, dname, dsize):                                                                              
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
    
    lr = LogisticRegression(penalty='l1', solver='saga', max_iter = 5000)
    lrcv = LogisticRegression(penalty='l1', solver='saga', max_iter = 5000)
    
    # get datasets
    X, y = load_data(dname)    
    ds = DataSplitter()                                                 
    ds.fit(X, y)                                                    
    ds.configure(NSETS, dsize)                                         
    X, y = ds.get_train(splitn)       
    if y.sum() == 0:
        fileres = open("registrylr/%s_%s_%s_zeros.csv" % (dname, splitn, dsize), "a")
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
    lr.fit(Xtest, ytest)
    filetme = open("registrylr/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
    filetme.write("testprec,%s\n" % testprec) 
    filetme.write("trainprec,%s\n" % trainprec) 
    filetme.write("testcoef,%s\n" % get_coef(lr)) 
    filetme.close()        

    #===== trees                         

    fileres = open("registrylr/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
    # (1) model (2) gen (3) met (4) sctr (5) scnew (6) sctest (7) coef (8) time
    for k, names in zip([lr],["lr"]):
        start = time.time()
        k.fit(X, y)
        end = time.time()                                                  
        sctrain = k.score(X, y)
        sctest = k.score(Xtest, ytest)
        fileres.write(names + ",na,na,%s,nan,%s,%s,%s\n" % (sctrain, sctest, get_coef(k), (end-start))) 
    
    # DT HPO
    par_vals = [0.01, 0.1, 0.25, 0.5, 1, 1.5, 10]
    parameters = {'C': par_vals}   
    start = time.time()                           
    tmp = GridSearchCV(lrcv, parameters, refit = False).fit(X, y).cv_results_ 
    tmp = opt_param(tmp, len(par_vals))
    lrcv = LogisticRegression(penalty='l1', solver='saga', C = par_vals[np.argmax(tmp)])                                            
    lrcv.fit(X, y)
    end = time.time()
    sctrain = lrcv.score(X, y)
    sctest = lrcv.score(Xtest, ytest) 
    fileres.write("lrcv,na,na,%s,nan,%s,%s,%s\n" % (sctrain, sctest, get_coef(lrcv), (end-start))) 
    filetme = open("registrylr/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
    filetme.write("lrcvsc,%s\n" % tmp[np.argmax(tmp)])
    filetme.close()    
    
    # prelim
                                                
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
              gennoise, gensmote, genadasyn, genrfdens, genkdem]:
        filetme = open("registrylr/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                      
        start = time.time()
        i.fit(X, y)
        end = time.time()
        filetme.write(i.my_name() + ",%s\n" % (end-start)) 
        filetme.close()
        
    for j in [metarf, metaxgb]: 
        filetme = open("registrylr/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                     
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
        filetme = open("registrylr/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
        filetme.write(j.my_name() + "vva,%s\n" % (end-start))
        filetme.close()
        
        for k, names in zip([lr, lrcv], ["lr", "lrcv"]):
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
            filetme = open("registrylr/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
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
                              
            fileres = open("registrylr/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
            start = time.time()
            k.fit(Xnew, ynew)
            end = time.time()                                     
            sctrain = k.score(X, y)
            scnew = k.score(Xnew, ynew)
            sctest = k.score(Xtest, ytest)
            fileres.write(names + ",vva,%s,%s,nan,%s,%s,%s\n" % (j.my_name(), sctrain, sctest, get_coef(k), (end-start))) 
            fileres.close()      
                
    
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
                         gennoise, gensmote, genadasyn, genrfdens, genkdem]:
        
        filetme = open("registrylr/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
        start = time.time()
        Xgen = i.sample(100000 - dsize)  
        end = time.time()
        filetme.write(i.my_name() + "gen,%s\n" % (end-start))
        filetme.close()   
        
        for j in [metarf, metaxgb]:
            filetme = open("registrylr/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
            start = time.time()
            Xnew = Xgen.copy()  
            ynew = j.predict(Xnew)
            Xnew = np.concatenate([Xnew, X])
            ynew = np.concatenate([ynew, y])
            end = time.time()
            filetme.write(i.my_name() + j.my_name() + ",%s\n" % (end-start))  
            filetme.write(i.my_name() + j.my_name() + "prec,%s\n" % (ynew.mean() if defprec == 1 else 1 - ynew.mean()))
            filetme.close()
            
            for k, names in zip([lr, lrcv], ["lr", "lrcv"]):
                fileres = open("registrylr/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
                start = time.time()
                k.fit(Xnew, ynew)
                end = time.time()                                     
                sctrain = k.score(X, y)
                scnew = k.score(Xnew, ynew)
                sctest = k.score(Xtest, ytest)
                fileres.write(names +",%s,%s,%s,nan,%s,%s,%s\n" % (i.my_name(), j.my_name(), sctrain, sctest, get_coef(k), (end-start))) 
                fileres.close()      
        
        

def exp_parallel():
    # pool = Pool(4)
    pool = Pool(cpu_count())
    pool.starmap_async(experiment_lr, product(SPLITNS, DNAMES, DSIZES))
    pool.close()
    pool.join()

if __name__ == "__main__":
    exp_parallel()


experiment_lr(2, "avila", 200)
