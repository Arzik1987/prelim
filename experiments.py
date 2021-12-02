
# Prevent numpy multithreading: https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

# Black-box models
from src.metamodels.rf import Meta_rf
from src.metamodels.xgb import Meta_xgb

# Generators
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

# White-box models
from sklearn.tree import DecisionTreeClassifier
import wittgenstein as lw
from src.subgroup_discovery.BI import BI
from src.subgroup_discovery.PRIM import PRIM

# Other
from src.utils.data_splitter import DataSplitter
from src.utils.data_loader import load_data
from src.utils.helpers import opt_param, n_leaves, get_bi_param, get_new_test
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from itertools import product
import time
import copy
import logging
import traceback

# Directory for storing the results
dirnme = 'registry'
if not os.path.exists(dirnme):
    os.makedirs(dirnme)


# ==============================    Experiment description      ===================================

def experiment(splitn, dname, dsize):  
    s_t = time.time()
    # Generators                                                                            
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
    
    # Black-box models
    metarf = Meta_rf()
    metaxgb = Meta_xgb()
    
    # White-box models
    dt = DecisionTreeClassifier(min_samples_split = 10)
    # one could restrict depth instead. Results will be worse, but 
    # ranking of generator's will not generally change (still kde is the best)
    dtc = DecisionTreeClassifier(max_leaf_nodes = 8)    
    dtval = DecisionTreeClassifier()
    ripper = lw.RIPPER(max_rules = 8)
    irep = lw.IREP(max_rules = 8)
    
    # get datasets
    X, y = load_data(dname)
    # seed here is very important to have consistent splits in different experiments    
    ds = DataSplitter(seed = 2020)                                                
    ds.fit(X, y)                                                    
    ds.configure(NSETS, dsize)                                         
    X, y = ds.get_train(splitn)       
    if y.sum() == 0:    # only one class is in the dataset
        fileres = open("registry/%s_%s_%s_zeros.csv" % (dname, splitn, dsize), "a")
        fileres.close()
        return                                    
    Xtest, ytest = ds.get_test(splitn) 
    # if the attribute takes a single value in X (train) - filter it out!
    Xtest = Xtest[:,(X.max(axis=0) != X.min(axis=0))] 
    X = X[:,(X.max(axis=0) != X.min(axis=0))]
    
    # scale data to unit variance
    ss = StandardScaler()                                               
    ss.fit(X)                                                       
    X = ss.transform(X) 
    Xtest = ss.transform(Xtest)
    # enlarge dataset for rules generation (leads to a stronger baseline for IREP)
    tms = int(np.ceil(10000/X.shape[0]))
    Xr = np.tile(X, [tms,1])
    yr = np.tile(y, tms)
    
    # Here we write results
    # structure: (1) model (2) gen (3) met (4) sctr (5) sctest (6) nleaves/rules (7) time (8) fidelity
    fileres = open(dirnme + '/%s_%s_%s.csv' % (dname, splitn, dsize), 'a')
    # structure: (1) variable (2) value
    filetme = open(dirnme + '/%s_%s_%s_meta.csv' % (dname, splitn, dsize), 'a')
    
    # accuracy of the naive (= default class) classifier on train and test
    defprec = 1 if y.mean() >= 0.5 else 0
    filetme.write("testprec,%s\n" % (ytest.mean() if defprec == 1 else 1 - ytest.mean())) 
    filetme.write("trainprec,%s\n" % (y.mean() if defprec == 1 else 1 - y.mean()))
    ydeftest = np.ones(len(ytest))*defprec  # for fidelity of naive model
      
    # WB models, no HPO                          
    for k, names in zip([dt, dtc],['dt', 'dtc']):
        start = time.time()
        k.fit(X, y)
        end = time.time()                                                  
        sctrain = k.score(X, y)
        sctest = k.score(Xtest, ytest)
        fileres.write(names + ',na,na,%s,%s,%s,%s,na\n' % (sctrain, sctest, n_leaves(k), (end-start))) 
        
    for k, names in zip([ripper, irep],['ripper', 'irep']):
        start = time.time()
        k.fit(Xr, yr)
        end = time.time()                                                  
        sctrain = k.score(Xr, yr)
        sctest = k.score(Xtest, ytest)
        fileres.write(names + ',na,na,%s,%s,%s,%s,na\n' % (sctrain, sctest, len(k.ruleset_), (end-start))) 
    del Xr, yr
    
    # WB models, HPO - optimize the number of leaves using grid search 
    # decision tree
    par_vals = [2**number for number in [1,2,3,4,5,6,7]]
    parameters = {'max_leaf_nodes': par_vals}   
    start = time.time()                          
    tmp = GridSearchCV(dtval, parameters, refit = False).fit(X, y).cv_results_ 
    tmp = opt_param(tmp, len(par_vals))
    dtval = DecisionTreeClassifier(max_leaf_nodes = par_vals[np.argmax(tmp)])                                            
    dtval.fit(X, y)
    end = time.time()
    sctrain = dtval.score(X, y)
    sctest = dtval.score(Xtest, ytest) 
    fileres.write("dtval,na,na,%s,%s,%s,%s,na\n" % (sctrain, sctest, n_leaves(dtval), (end-start)))
  
    # for the following fidelity estimation
    dtvalold = copy.deepcopy(dtval) 
    # restricting the number of leaves in the following attempts
    dtval = DecisionTreeClassifier(max_leaf_nodes = max(n_leaves(dtval),2)) 
    
    # BI 
    parsbi = get_bi_param(5, X.shape[1]) # hyperparameters for BI with HPO
    parameters = {'depth': parsbi}                                      
    start = time.time() 
    tmp = GridSearchCV(BI(), parameters, refit = False).fit(X, y).cv_results_ 
    tmp = opt_param(tmp, len(parsbi))
    bicv = BI(depth = parsbi[np.argmax(tmp)])
    bicv.fit(X, y)
    end = time.time()
    sctrain = bicv.score(X, y)
    sctest = bicv.score(Xtest, ytest)     
    fileres.write("bicv,na,na,%s,%s,%s,%s,na\n" % (sctrain, sctest, bicv.get_nrestr(), (end-start))) 
    # limiting the number of restricted dimensions in the following attempts
    bicv = BI(depth = bicv.get_nrestr())
    
    # PRIM
    par_vals = [0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2]
    parameters = {'alpha': par_vals}                                     
    start = time.time() 
    tmp = GridSearchCV(PRIM(), parameters, refit = False).fit(X, y).cv_results_ 
    tmp = opt_param(tmp, len(par_vals))
    primcv = PRIM(alpha = par_vals[np.argmax(tmp)])
    primcv.fit(X, y)
    end = time.time()
    sctrain = primcv.score(X, y)
    sctest = primcv.score(Xtest, ytest)     
    fileres.write("primcv,na,na,%s,%s,%s,%s,na\n" % (sctrain, sctest, primcv.get_nrestr(), (end-start))) 

    ################
    #### PRELIM ####
    ################
    
    # Fitting generators
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
              gengmmbical, gensmote, genadasyn, genrfdens, genkdem, genkdeb]:                             
        start = time.time()
        i.fit(X, y)
        end = time.time()
        filetme.write(i.my_name() + "time,%s\n" % (end-start)) 
        
    # Fitting black-box models
    for j in [metarf, metaxgb]: 
        start = time.time()
        j.fit(X, y)
        end = time.time()
        filetme.write(j.my_name() + "time,%s\n" % (end-start)) 
        filetme.write(j.my_name() + "acc,%s\n" % j.fit_score()) 
        
        # fidelity of the naive classifier
        ypredtest = j.predict(Xtest)
        fidel = np.count_nonzero(ypredtest == ydeftest)/len(ypredtest)
        filetme.write(j.my_name() + "fid,%s\n" % (fidel))
        
        # fidelity of white-box models trained from original data
        for k, names in zip([dt, dtc, dtvalold, ripper, irep], ['dt', 'dtc', 'dtval', 'ripper', 'irep']):
            fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
            filetme.write(j.my_name() + names + "fid,%s\n" % (fidel))

        
    # re-rx generator
    # consider it separately since it requires a metamodel as input
    for j in [metarf, metaxgb]: 
        genrerx.fit(X, y, j)
        Xnew = genrerx.sample()  
        ynew = j.predict(Xnew)   
                                 
        ypredtest = j.predict(Xtest)
        for k, names in zip([dt, dtc, dtval, ripper, irep], ['dt', 'dtc', 'dtval', 'ripper', 'irep']):
            start = time.time()
            k.fit(Xnew, ynew)
            end = time.time()                                     
            sctrain = k.score(X, y)
            sctest = k.score(Xtest, ytest)
            fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
            if names in ['ripper', 'irep']:
                nlr = len(k.ruleset_)
            else:
                nlr = n_leaves(k)
            fileres.write(names +",rerx,%s,%s,%s,%s,%s,%s\n" % (j.my_name(), sctrain, sctest, nlr, (end-start), fidel)) 
        
        ynew = j.predict_proba(Xnew) 
        for k, names in zip([primcv, bicv],['primcv', 'bicv']):
            start = time.time()
            k.fit(Xnew, ynew)
            end = time.time()                                     
            sctrain = k.score(X, y)
            sctest = k.score(Xtest, ytest)
            fileres.write(names +",rerx,%s,%s,%s,%s,%s,na\n" % (j.my_name(), sctrain, sctest, k.get_nrestr(), (end-start))) 

    # vva generator
    # consider it separately since it requires white-box model and metamodel
    for j in [metarf, metaxgb]:
        # split train data into train and validation
        ntrain = int(np.ceil(X.shape[0]*2/3))
        Xtrain = X[:ntrain,:].copy()
        Xval = X[ntrain:,:].copy()
        ytrain = y[:ntrain].copy()
        yval = y[ntrain:].copy()
        start = time.time() 
        genvva.fit(Xtrain, j)
        end = time.time()
        filetme.write(j.my_name() + 'vva,%s\n' % (end-start))
        
        ypredtest = j.predict(Xtest)
        # optimize the number of generated points for each white-box model separately
        for k, names in zip([dt, dtc, dtval, ripper, irep, primcv, bicv],\
                            ['dt', 'dtc', 'dtval', 'ripper', 'irep', 'primcv', 'bicv']):
            start = time.time()      
            if names in ['primcv', 'bicv']:
                k.fit(Xtrain, j.predict_proba(Xtrain))
            else:
                k.fit(Xtrain, ytrain)
            sctest0 = k.score(Xval, yval)
            ropt = 0
            
            if genvva.will_generate():  # black-box model does not predict a single class
                for r in np.linspace(0.5, 2.5, num = 5):
                    Xnew = genvva.sample(r)    
                    ynew = j.predict(Xnew) 
                    Xnew = np.concatenate([Xnew, Xtrain])
                    ynew = np.concatenate([ynew, ytrain])
                    if names in ['primcv', 'bicv']:
                        k.fit(Xnew, j.predict_proba(Xnew))
                    else:
                        k.fit(Xnew, ynew)
                    sctest = k.score(Xval, yval)
                    if sctest > sctest0:
                        sctest0 = sctest
                        ropt = r
               
            end = time.time()
            filetme.write(names + j.my_name() + "vvaopt,%s\n" % (end-start))
            filetme.write(names + j.my_name() + "ropt,%s\n" % ropt)
            
            # generate points from fitted vva with optimal hyperparameter
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
                                        
            # train white-box model with vva
            start = time.time()
            if names in ['primcv', 'bicv']:
                k.fit(Xnew, j.predict_proba(Xnew))
            else:
                k.fit(Xnew, ynew)
            end = time.time()                                     
            sctrain = k.score(X, y)
            sctest = k.score(Xtest, ytest)
            fidel = 'na' if names in ['primcv', 'bicv'] else\
                np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
            if names in ['primcv', 'bicv']:
                nlr = k.get_nrestr()
            elif names in ['ripper', 'irep']:
                nlr = len(k.ruleset_)
            else:
                nlr = n_leaves(k)
            fileres.write(names + ",vva,%s,%s,%s,%s,%s,%s\n" % (j.my_name(), sctrain, sctest, nlr, (end-start), fidel)) 
    
    # All remaining generators
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
              gengmmbical, gensmote, genadasyn, genrfdens, genkdem, genkdeb]:
        
        start = time.time()
        Xgen = i.sample(100000 - dsize)  
        end = time.time()
        filetme.write(i.my_name() + "gen,%s\n" % (end-start))
        
        for j in [metarf, metaxgb]:
            ypredtest = j.predict(Xtest)
            
            start = time.time()
            Xnew = Xgen.copy()  
            ynew = j.predict(Xnew)
            Xnew = np.concatenate([X, Xnew])
            ynew = np.concatenate([y, ynew])
            end = time.time()
            filetme.write(i.my_name() + j.my_name() + ",%s\n" % (end-start))  
             
            for k, names in zip([dt, dtc, dtval], ['dt', 'dtc', 'dtval']):
                start = time.time()
                k.fit(Xnew, ynew)
                end = time.time()                                     
                sctrain = k.score(X, y)
                sctest = k.score(Xtest, ytest)
                fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
                fileres.write(names +",%s,%s,%s,%s,%s,%s,%s\n" % (i.my_name(), j.my_name(), sctrain, sctest, n_leaves(k), (end-start), fidel)) 
    
            # smaller data for rules
            Xnew = Xnew[:10000,:]
            ynew = ynew[:10000]   
            
            for k, names in zip([ripper, irep],['ripper', 'irep']):
                start = time.time()
                k.fit(Xnew, ynew)
                end = time.time()                                     
                sctrain = k.score(X, y)
                sctest = k.score(Xtest, ytest)
                fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
                fileres.write(names +",%s,%s,%s,%s,%s,%s,%s\n" % (i.my_name(), j.my_name(), sctrain, sctest, len(k.ruleset_), (end-start), fidel)) 
            
            # probabilities for subgroup discovery
            ynew = j.predict_proba(Xnew)
            for k, names in zip([primcv, bicv], ['primcv', 'bicv']):
                start = time.time()
                k.fit(Xnew, ynew)
                end = time.time()                                     
                sctrain = k.score(X, y)
                sctest = k.score(Xtest, ytest)
                fileres.write(names +",%s,%s,%s,%s,%s,%s,na\n" % (i.my_name(), j.my_name(), sctrain, sctest, k.get_nrestr(), (end-start))) 
    
    # semi-supervised learning testing
    Xtest, ytest, Xgen = get_new_test(Xtest, ytest, dsize)
    for j in [metarf, metaxgb]: 
        ypredtest = j.predict(Xtest)
        
        ynew = np.concatenate([j.predict(Xgen), y])
        Xnew = np.concatenate([Xgen, X])
        
        for k, names in zip([dt, dtc, dtval, ripper, irep], ['dt', 'dtc', 'dtval', 'ripper', 'irep']):
            start = time.time()
            k.fit(Xnew, ynew)
            end = time.time()                                     
            sctrain = k.score(X, y)
            sctest = k.score(Xtest, ytest)
            fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
            if names in ['ripper', 'irep']:
                nlr = len(k.ruleset_)
            else:
                nlr = n_leaves(k)
            fileres.write(names +",ssl,%s,%s,%s,%s,%s,%s\n" % (j.my_name(), sctrain, sctest, nlr, (end-start), fidel)) 
                
        ynew = j.predict_proba(Xnew)
        for k, names in zip([primcv, bicv], ['primcv', 'bicv']):
            start = time.time()
            k.fit(Xnew, ynew)
            end = time.time()                                     
            sctrain = k.score(X, y)
            sctest = k.score(Xtest, ytest)
            fileres.write(names +",ssl,%s,%s,%s,%s,%s,na\n" % (j.my_name(), sctrain, sctest, k.get_nrestr(), (end-start))) 

    fileres.close()
    e_t = time.time()
    filetme.write('overall,%s\n' %(e_t-s_t))
    filetme.close()


# ==============================            Logging             ===================================


def non_interrupting_experiment(splitn, dname, dsize):
    logger = logging.getLogger("error")

    succesful = False
    stacktrace = None
    try:
        experiment(splitn, dname, dsize)
        succesful = True
    except Exception as e:
        logger.log(logging.ERROR, f"Error occured in Experiment with: splits={splitn}, dataset=${dname}, Size={dsize})")
        logger.log(logging.ERROR, traceback.format_exc())
        stacktrace = traceback.format_exc()

    return succesful, splitn, dname, dsize, stacktrace


# ==============================    Configuration & Execution      ===================================


NSETS = 25      # number of experiments with each dataset
SPLITNS = list(range(0, NSETS))         # list of experiment numbers for each dataset
DNAMES = ['anuran', 'avila', 'bankruptcy', 'ccpp', 'cc', 'clean2', 'dry',
          'ees', 'electricity', 'gas', 'gt', 'higgs21', 'higgs7', 'htru', 'jm1',
          'ml', 'nomao', 'occupancy', 'parkinson', 'pendata', 'ring',
          'saac2', 'seizure', 'sensorless', 'seoul', 'shuttle', 'stocks',
          'sylva', 'turbine', 'wine']       #  datasets' names
DSIZES = [100, 400]         # datasets' sizes used in experiments


# run experiments on all available cores
def exp_parallel():
    args = product(SPLITNS, DNAMES, DSIZES)
    result_list = Parallel(n_jobs=cpu_count(), verbose=100)(delayed(non_interrupting_experiment)(*a) for a in args)
    print(np.asarray(result_list))


if __name__ == "__main__":
    exp_parallel()

