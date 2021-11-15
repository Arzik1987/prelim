import os

os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import src.classification_rules.wittgenstein as lw

from src.metamodels.rf import Meta_rf
from src.metamodels.xgb import Meta_xgb

from src.utils.data_splitter import DataSplitter
from src.utils.data_loader import load_data

from sklearn.preprocessing import StandardScaler

import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import product
import time
import copy

if not os.path.exists('registryrules'):
    os.makedirs('registryrules')
    
# =============================================================================

NSETS = 25
SPLITNS = list(range(0, NSETS))
DNAMES = ["nomao", "gas", "clean2", "seizure",
          "occupancy", "higgs7", "electricity", "htru", "shuttle", "avila",
          "cc", "ees", "pendata", "ring", "sylva", "higgs21",
          "jm1", "saac2", "stocks", 
          "sensorless", "bankruptcy", "gt",
          "ccpp", "seoul", "turbine", "wine", "parkinson", "dry", "anuran", "ml"]

DSIZES = [400, 100]


def experiment_rules(splitn, dname, dsize):                                                                              

    ripper = lw.RIPPER(max_rules = 8)
    irep = lw.IREP(max_rules = 8)
    
    metarf = Meta_rf()
    metaxgb = Meta_xgb()
    
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
    
    ss = StandardScaler()                                               
    ss.fit(X)                                                       
    X = ss.transform(X) 
    Xtest = ss.transform(Xtest)  
    
    Xold = X.copy()
    yold = y.copy()
    tms = int(np.ceil(10000/X.shape[0]))
    X = np.tile(X,[tms,1])
    y = np.tile(y,tms)

    #===== rules                         

    fileres = open("registryrules/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
    # (1) model (2) gen (3) met (4) sctr (5) scnew (6) sctest (7) nleaves (8) time (9) fidelity
    for k, names in zip([ripper, irep],["ripper", "irep"]):
        start = time.time()
        k.fit(X, y)
        end = time.time()                                                  
        sctrain = k.score(X, y)
        sctest = k.score(Xtest, ytest)
        fileres.write(names + "ext,na,na,%s,nan,%s,%s,%s,na\n" % (sctrain, sctest, len(k.ruleset_), (end-start))) 
    fileres.close()
    
    #===== fidelity   
        
    for j in [metarf, metaxgb]:                                    
      j.fit(Xold, yold)
      ypredtest = j.predict(Xtest)
      filetme = open("registryrules/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")  
      for k, names in zip([ripper, irep],["ripper", "irep"]):
          fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
          filetme.write(j.my_name() + names + "extfid,%s\n" % (fidel))
      filetme.close()
      

def exp_parallel():
    pool = Pool(cpu_count())
    pool.starmap_async(experiment_rules, product(SPLITNS, DNAMES, DSIZES))
    pool.close()
    pool.join()

if __name__ == "__main__":
    exp_parallel()
    
# experiment_rules(0, 'avila', 100)
