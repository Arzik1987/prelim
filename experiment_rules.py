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
from src.generators.rand import Gen_randn, Gen_randu
from src.generators.dummy import Gen_dummy
from src.generators.smote import Gen_smote
from src.generators.adasyn import Gen_adasyn
from src.generators.rfdens import Gen_rfdens
from src.generators.vva import Gen_vva
from src.generators.rerx import Gen_rerx

import src.classification_rules.wittgenstein as lw

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
DNAMES = ["nomao", "gas", "clean2", "seizure",
          "occupancy", "higgs7", "electricity", "htru", "shuttle", "avila",
          "cc", "ees", "pendata", "ring", "sylva", "higgs21",
          "jm1", "saac2", "stocks", 
          "sensorless", "bankruptcy", "gt",
          "ccpp", "seoul", "turbine", "wine", "parkinson", "dry", "anuran", "ml"]
# DSIZES = [100]
DSIZES = [400, 100]


def experiment_rules(splitn, dname, dsize):                                                                              
    gengmmbic = Gen_gmmbic() 
    genkde = Gen_kdebw()
    genkdeb = Gen_kdeb()
    genkdem = Gen_kdebwm()
    genmunge = Gen_munge()
    genrandu = Gen_randu()
    genrandn = Gen_randn()
    gendummy = Gen_dummy()
    # gennoise = Gen_noise()
    genadasyn = Gen_adasyn()
    gensmote = Gen_smote()
    genrfdens = Gen_rfdens()
    genvva = Gen_vva()
    gengmmbical = Gen_gmmbical()
    genrerx = Gen_rerx()
                                               
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
    
    ss = StandardScaler()                                               
    ss.fit(X)                                                       
    X = ss.transform(X) 
    Xtest = ss.transform(Xtest)
    
    defprec = 1 if y.mean() >= 0.5 else 0
    testprec = ytest.mean() if defprec == 1 else 1 - ytest.mean()
    trainprec = y.mean() if defprec == 1 else 1 - y.mean()
    filetme = open("registryrules/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
    filetme.write("testprec,%s\n" % testprec) 
    filetme.write("trainprec,%s\n" % trainprec) 
    filetme.close()        

    #===== rules                         

    fileres = open("registryrules/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
    # (1) model (2) gen (3) met (4) sctr (5) scnew (6) sctest (7) nleaves (8) time (9) fidelity
    for k, names in zip([ripper, irep],["ripper", "irep"]):
        start = time.time()
        k.fit(X, y)
        end = time.time()                                                  
        sctrain = k.score(X, y)
        sctest = k.score(Xtest, ytest)
        fileres.write(names + ",na,na,%s,nan,%s,%s,%s,na\n" % (sctrain, sctest, len(k.ruleset_), (end-start))) 
        
    ripperc = lw.RIPPER(max_rules = len(ripper.ruleset_))
    irepc = lw.IREP(max_rules = len(irep.ruleset_))
    
    # prelim
                                                
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
              gengmmbical, gensmote, genadasyn, genrfdens, genkdem, genkdeb]:
        filetme = open("registryrules/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                      
        start = time.time()
        i.fit(X, y)
        end = time.time()
        filetme.write(i.my_name() + ",%s\n" % (end-start)) 
        filetme.close()
        
    for j in [metarf, metaxgb]: 
        filetme = open("registryrules/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                     
        start = time.time()
        j.fit(X, y)
        end = time.time()
        ypredtest = j.predict(Xtest)
        filetme.write(j.my_name() + ",%s\n" % (end-start)) 
        filetme.write(j.my_name() + "acc,%s\n" % j.fit_score()) 
        for k, names in zip([ripper, irep],["ripper", "irep"]):
            fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
            filetme.write(j.my_name() + names + "fid,%s\n" % (fidel))
        filetme.close()
        
        # rerx generator
        genrerx.fit(X, y, j)
        Xnew = genrerx.sample()  
        ynew = j.predict(Xnew)
        filetme = open("registryrules/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")                                      
        filetme.write("rerxndel,%s\n" % (dsize - len(y))) 
        filetme.close()
        
        for k, names in zip([ripper, irep, ripperc, irepc],["ripper", "irep", "ripperc", "irepc"]):
            fileres = open("registryrules/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
            start = time.time()
            k.fit(Xnew, ynew)
            end = time.time()                                     
            sctrain = k.score(X, y)
            scnew = k.score(Xnew, ynew)
            sctest = k.score(Xtest, ytest)
            fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
            fileres.write(names +",%s,%s,%s,nan,%s,%s,%s,%s\n" % (genrerx.my_name(), j.my_name(), sctrain, sctest, len(k.ruleset_), (end-start), fidel)) 
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
        filetme = open("registryrules/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
        filetme.write(j.my_name() + "vva,%s\n" % (end-start))
        filetme.close()
        
        for k, names in zip([ripper, irep, ripperc, irepc],["ripper", "irep", "ripperc", "irepc"]):
            start = time.time()            
            k.fit(Xstrain, ystrain)
            sctest0 = k.score(Xstest, ystest)
            ropt = 0
            
            if genvva.will_generate():
                for r in np.linspace(0.5, 2.5, num = 5):
                    Xnew = genvva.sample(r)    
                    ynew = np.concatenate([j.predict(Xnew), ystrain])
                    try:
                        k.fit(np.concatenate([Xnew, Xstrain]), ynew)
                    except:
                        pass                                    
                    sctest = k.score(Xstest, ystest)
                    if sctest > sctest0:
                        sctest0 = sctest
                        ropt = r
               
            end = time.time()
            filetme = open("registryrules/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
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
                              
            fileres = open("registryrules/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
            start = time.time()
            try:
                k.fit(Xnew, ynew)
            except:
                k.fit(X, y)
            end = time.time()                                     
            sctrain = k.score(X, y)
            scnew = k.score(Xnew, ynew)
            sctest = k.score(Xtest, ytest)
            fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
            fileres.write(names + ",vva,%s,%s,nan,%s,%s,%s,%s\n" % (j.my_name(), sctrain, sctest, len(k.ruleset_), (end-start), fidel)) 
            fileres.close()      
                
    
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
                         gengmmbical, gensmote, genadasyn, genrfdens, genkdem, genkdeb]:
        
        filetme = open("registryrules/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
        start = time.time()
        Xgen = i.sample(10000 - dsize)  
        end = time.time()
        filetme.write(i.my_name() + "gen,%s\n" % (end-start))
        filetme.close()   
        
        for j in [metarf, metaxgb]:
            filetme = open("registryrules/%s_%s_%s_times.csv" % (dname, splitn, dsize), "a")
            start = time.time()
            Xnew = Xgen.copy()  
            ynew = j.predict(Xnew)
            Xnew = np.concatenate([Xnew, X])
            ynew = np.concatenate([ynew, y])
            end = time.time()
            filetme.write(i.my_name() + j.my_name() + ",%s\n" % (end-start))  
            filetme.write(i.my_name() + j.my_name() + "prec,%s\n" % (ynew.mean() if defprec == 1 else 1 - ynew.mean()))
            filetme.close()
            ypredtest = j.predict(Xtest) # not too efficient to calculate it here...
            
            for k, names in zip([ripper, irep, ripperc, irepc],["ripper", "irep", "ripperc", "irepc"]):
                fileres = open("registryrules/%s_%s_%s.csv" % (dname, splitn, dsize), "a")
                start = time.time()
                k.fit(Xnew, ynew)
                end = time.time()                                     
                sctrain = k.score(X, y)
                scnew = k.score(Xnew, ynew)
                sctest = k.score(Xtest, ytest)
                fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
                fileres.write(names +",%s,%s,%s,nan,%s,%s,%s,%s\n" % (i.my_name(), j.my_name(), sctrain, sctest, len(k.ruleset_), (end-start), fidel)) 
                fileres.close()      
        
        

def exp_parallel():
    # pool = Pool(4)
    pool = Pool(cpu_count())
    pool.starmap_async(experiment_rules, product(SPLITNS, DNAMES, DSIZES))
    pool.close()
    pool.join()

if __name__ == "__main__":
    exp_parallel()


# experiment_dt(0, "avila", 100)
