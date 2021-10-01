import os

os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

from src.metamodels.rf import Meta_rf
from src.metamodels.xgb import Meta_xgb

from src.utils.data_splitter import DataSplitter
from src.utils.data_loader import load_data

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import product
import time
import copy

if not os.path.exists('registryfid'):
    os.makedirs('registryfid')
    
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

def experiment_fid(splitn, dname, dsize):                                                                              
                                               
    metarf = Meta_rf()
    metaxgb = Meta_xgb()
      
    # get datasets
    X, y = load_data(dname)    
    ds = DataSplitter()                                                 
    ds.fit(X, y)                                                    
    ds.configure(NSETS, dsize)                                         
    X, y = ds.get_train(splitn)       
    if y.sum() == 0:
        fileres = open("registryfid/%s_%s_%s_zeros.csv" % (dname, splitn, dsize), "a")
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
    ydeftest = np.ones(len(ytest))*defprec
  
    for j in [metarf, metaxgb]: 
        filetme = open("registryfid/%s_%s_%s.csv" % (dname, splitn, dsize), "a")                                
        j.fit(X, y)
        ypredtest = j.predict(Xtest)
        fidel = np.count_nonzero(ypredtest == ydeftest)/len(ypredtest)
        filetme.write(j.my_name() + "fid,%s\n" % (fidel))
        filetme.close()


def exp_parallel():
    # pool = Pool(4)
    pool = Pool(cpu_count())
    pool.starmap_async(experiment_fid, product(SPLITNS, DNAMES, DSIZES))
    pool.close()
    pool.join()

if __name__ == "__main__":
    exp_parallel()


# experiment_fid(0, "avila", 100)
