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

import wittgenstein as lw

from src.utils.data_splitter import DataSplitter
from src.utils.data_loader import load_data

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import product
import time
import copy

if not os.path.exists('registryrules'):
    os.makedirs('registryrules')
    
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
    
    ripper = lw.RIPPER(max_rules = 8)
    irep = lw.IREP(max_rules = 8)
    
    # get datasets
    X, y = load_data(dname)    
    ds = DataSplitter()                                                 
    ds.fit(X, y)                                                    
    ds.configure(NSETS, dsize)                                         
    X, y = ds.get_train(splitn)       
    if y.sum() == 0:
        fileres = open("registryrules/%s_%s_%s_zeros.csv" % (dname, splitn, dsize), "a")
        fileres.close()
        return                                    
    Xtest, ytest = ds.get_test(splitn) 
    Xtest = Xtest[:,(X.max(axis=0) != X.min(axis=0))]
    X = X[:,(X.max(axis=0) != X.min(axis=0))]
    
    defprec = 1 if y.mean() >= 0.5 else 0
    testprec = ytest.mean() if defprec == 1 else 1 - ytest.mean()
    trainprec = y.mean() if defprec == 1 else 1 - y.mean()
    filetme = open("registryrules/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
    filetme.write("testprec,%s\n" % testprec) 
    filetme.write("trainprec,%s\n" % trainprec) 
    filetme.close()        

    #===== rules                         

    fileres = open("registryrules/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
    # (1) model (2) gen (3) met (4) sctr (5) scnew (6) sctest (7) nleaves (8) time
    start = time.time()
    ripper.fit(X, y)
    end = time.time()                                                   
    sctrain = ripper.score(X, y)
    sctest = ripper.score(Xtest, ytest)
    fileres.write("ripper,na,na,%s,nan,%s,nan,%s" % (sctrain, sctest, (end-start))) 

    start = time.time()
    irep.fit(X, y)
    end = time.time()                                                   
    sctrain = irep.score(X, y)
    sctest = irep.score(Xtest, ytest)
    fileres.write("\nirep,na,na,%s,nan,%s,nan,%s" % (sctrain, sctest, (end-start))) 
    
    
    # prelim
    ss = StandardScaler()                                               
    ss.fit(X)                                                       
    Xs = ss.transform(X) 
                                                
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
              gennoise, gensmote, genadasyn, genrfdens]:
        filetme = open("registryrules/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                      
        start = time.time()
        i.fit(Xs, y)
        end = time.time()
        filetme.write(i.my_name() + ",%s\n" % (end-start)) 
        filetme.close()
        
    for j in [metarf, metaxgb]: 
        filetme = open("registryrules/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                     
        start = time.time()
        j.fit(Xs, y)
        end = time.time()
        filetme.write(j.my_name() + ",%s\n" % (end-start)) 
        filetme.write(j.my_name() + "acc,%s\n" % j.fit_score()) 
        filetme.close()
        
    for i, j in product([gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
                         gennoise, gensmote, genadasyn, genrfdens], [metarf, metaxgb]):
        filetme = open("registryrules/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")              

        start = time.time()
        Xnew = i.sample(10000 - dsize)                                                                      
        ynew = j.predict(Xnew)
        Xnew = ss.inverse_transform(Xnew)  
        Xnew = np.concatenate([Xnew, X])
        ynew = np.concatenate([ynew, y])
        end = time.time()
        filetme.write(i.my_name() + j.my_name() + ",%s\n" % (end-start))  
        filetme.write(i.my_name() + j.my_name() + "prec,%s\n" % (ynew.mean() if defprec == 1 else 1 - ynew.mean()))
        filetme.close()                             
        
        fileres = open("registryrules/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
        start = time.time()
        ripper.fit(Xnew, ynew)
        end = time.time()                                     
        sctrain = ripper.score(X, y)
        scnew = ripper.score(Xnew, ynew)
        sctest = ripper.score(Xtest, ytest)
        fileres.write("\nripper,%s,%s,%s,%s,%s,nan,%s" % (i.my_name(), j.my_name(), sctrain, scnew, sctest, (end-start))) 
        
        start = time.time()
        irep.fit(Xnew, ynew) 
        end = time.time()                                    
        sctrain = irep.score(X, y)
        scnew = irep.score(Xnew, ynew)
        sctest = irep.score(Xtest, ytest)                                       
        fileres.write("\nirep,%s,%s,%s,%s,%s,nan,%s" % (i.my_name(), j.my_name(), sctrain, scnew, sctest, (end-start)))
        
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

