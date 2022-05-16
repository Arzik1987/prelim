
# Prevent numpy multithreading: https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

# Black-box models
from metamodels.rf import Meta_rf
from metamodels.xgb import Meta_xgb
from metamodels.xgbb import Meta_xgb_bal
from metamodels.rfb import Meta_rf_bal

# Generators
from prelim.generators.gmm import Gen_gmmbic, Gen_gmmbical
from prelim.generators.kde import Gen_kdebw
from prelim.generators.kdem import Gen_kdebwm
from prelim.generators.munge import Gen_munge
from prelim.generators.kdeb import Gen_kdeb
from prelim.generators.rand import Gen_randn, Gen_randu
from prelim.generators.dummy import Gen_dummy
from prelim.generators.smote import Gen_smote
from prelim.generators.adasyn import Gen_adasyn
from prelim.generators.rfdens import Gen_rfdens
from prelim.generators.vva import Gen_vva
from prelim.generators.rerx import Gen_rerx

# White-box models
from sklearn.tree import DecisionTreeClassifier
import wittgenstein as lw
from prelim.sd.BI import BI
from prelim.sd.PRIM import PRIM

# Other
from utils.data_splitter import DataSplitter
from utils.data_loader import load_data
from utils.helpers import opt_param, n_leaves, get_bi_param, get_new_test
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from itertools import product
import time
import copy
import logging
import traceback

# Directory for storing the results
FILEPATH = os.path.dirname(os.path.abspath(__file__))
dirnme = FILEPATH + '/registry'
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
    metarfb = Meta_rf_bal()
    metaxgbb = Meta_xgb_bal()
    
    # White-box models
    dt = DecisionTreeClassifier(min_samples_split = 10)
    dtb = DecisionTreeClassifier(min_samples_split = 10, class_weight = 'balanced')
    # one could restrict depth instead. Results will be worse, but 
    # ranking of generator's will not generally change (still kde is the best)
    dtc = DecisionTreeClassifier(max_leaf_nodes = 8)    
    dtval = DecisionTreeClassifier()
    dtcb = DecisionTreeClassifier(max_leaf_nodes = 8, class_weight = 'balanced') 
    dtvalb = DecisionTreeClassifier(class_weight = 'balanced')
    # classification rules
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
        fileres = open(dirnme + '/%s_%s_%s_zeros.csv' % (dname, splitn, dsize), 'a')
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
    # structure: (1) model (2) gen (3) met (4) sctr (5) sctest (6) nleaves/rules (7) time (8) fidelity (9) balanced accuracy
    fileres = open(dirnme + '/%s_%s_%s.csv' % (dname, splitn, dsize), 'a')
    # structure: (1) variable (2) value
    filetme = open(dirnme + '/%s_%s_%s_meta.csv' % (dname, splitn, dsize), 'a')
    
    # accuracy of the naive (= default class) classifier on train and test
    # note that balanced accuracy for such classifier is always 0.5
    defprec = 1 if y.mean() >= 0.5 else 0
    filetme.write('testprec,%s\n' % (ytest.mean() if defprec == 1 else 1 - ytest.mean())) 
    filetme.write('trainprec,%s\n' % (y.mean() if defprec == 1 else 1 - y.mean()))
    ydeftest = np.ones(len(ytest))*defprec  # for fidelity of naive model
      
    # WB models, no HPO                          
    for k, names in zip([dt, dtc, dtb, dtcb],['dt', 'dtc', 'dtb', 'dtcb']):
        start = time.time()
        k.fit(X, y)
        end = time.time()                                                  
        sctrain = k.score(X, y)
        sctest = k.score(Xtest, ytest)
        bactest = balanced_accuracy_score(ytest, k.predict(Xtest))
        fileres.write(names + ',na,na,%s,%s,%s,%s,na,%s\n' % (sctrain, sctest, n_leaves(k), (end-start), bactest)) 
        
    for k, names in zip([ripper, irep],['ripper', 'irep']):
        start = time.time()
        k.fit(Xr, yr)
        end = time.time()                                                  
        sctrain = k.score(Xr, yr)
        sctest = k.score(Xtest, ytest)
        bactest = balanced_accuracy_score(ytest, k.predict(Xtest))
        fileres.write(names + ',na,na,%s,%s,%s,%s,na,%s\n' % (sctrain, sctest, len(k.ruleset_), (end-start), bactest)) 
    del Xr, yr
    
    # WB models, HPO - optimize the number of leaves using grid search 
    par_vals = [2**number for number in [1,2,3,4,5,6,7]]
    parameters = {'max_leaf_nodes': par_vals}   
    
    # decision tree
    start = time.time()                          
    tmp = GridSearchCV(dtval, parameters, refit = False).fit(X, y).cv_results_ 
    tmp = opt_param(tmp, len(par_vals))
    dtval = DecisionTreeClassifier(max_leaf_nodes = par_vals[np.argmax(tmp)])                                            
    dtval.fit(X, y)
    end = time.time()
    sctrain = dtval.score(X, y)
    sctest = dtval.score(Xtest, ytest) 
    bactest = balanced_accuracy_score(ytest, dtval.predict(Xtest))
    fileres.write('dtval,na,na,%s,%s,%s,%s,na,%s\n' % (sctrain, sctest, n_leaves(dtval), (end-start), bactest))
    
    # balanced decision tree
    start = time.time()                          
    tmp = GridSearchCV(dtvalb, parameters, refit = False, scoring = 'balanced_accuracy').fit(X, y).cv_results_ 
    tmp = opt_param(tmp, len(par_vals))
    dtvalb = DecisionTreeClassifier(max_leaf_nodes = par_vals[np.argmax(tmp)], class_weight='balanced')                                            
    dtvalb.fit(X, y)
    end = time.time()
    sctrain = dtvalb.score(X, y)
    sctest = dtvalb.score(Xtest, ytest) 
    bactest = balanced_accuracy_score(ytest, dtvalb.predict(Xtest))
    fileres.write('dtvalb,na,na,%s,%s,%s,%s,na,%s\n' % (sctrain, sctest, n_leaves(dtvalb), (end-start), bactest))
  
    # for the following fidelity estimation
    dtvalold = copy.deepcopy(dtval) 
    dtvalbold = copy.deepcopy(dtvalb) 
    # restricting the number of leaves in the following attempts
    dtval = DecisionTreeClassifier(max_leaf_nodes = max(n_leaves(dtval),2)) 
    dtvalb = DecisionTreeClassifier(max_leaf_nodes = max(n_leaves(dtvalb),2), class_weight='balanced')
    
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
    fileres.write('bicv,na,na,%s,%s,%s,%s,na,na\n' % (sctrain, sctest, bicv.get_nrestr(), (end-start))) 
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
    fileres.write('primcv,na,na,%s,%s,%s,%s,na,na\n' % (sctrain, sctest, primcv.get_nrestr(), (end-start))) 

    ################
    #### PRELIM ####
    ################
    
    # Fitting generators
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
              gengmmbical, gensmote, genadasyn, genrfdens, genkdem, genkdeb]:                             
        start = time.time()
        i.fit(X, y)
        end = time.time()
        filetme.write(i.my_name() + 'time,%s\n' % (end-start)) 
        
    # Fitting black-box models
    for j in [metarf, metaxgb, metarfb, metaxgbb]: 
        start = time.time()
        j.fit(X, y)
        end = time.time()
        filetme.write(j.my_name() + 'time,%s\n' % (end-start)) 
        filetme.write(j.my_name() + 'acccv,%s\n' % j.fit_score()) 
        
        # fidelity of the naive classifier
        ypredtest = j.predict(Xtest)
        fidel = np.count_nonzero(ypredtest == ydeftest)/len(ypredtest)
        filetme.write(j.my_name() + 'fid,%s\n' % (fidel))
        # out-of-sample accuracy and balanced accuracy
        filetme.write(j.my_name() + 'acc,%s\n' % accuracy_score(ytest, ypredtest))
        filetme.write(j.my_name() + 'bac,%s\n' % balanced_accuracy_score(ytest, ypredtest))
        
        # fidelity of white-box models trained from original data 
        if j in [metarf, metaxgb]:
            smodels = zip([dt, dtc, dtvalold, ripper, irep], ['dt', 'dtc', 'dtval', 'ripper', 'irep'])
        if j in [metarfb, metaxgbb]:
            smodels = zip([dtb, dtcb, dtvalbold], ['dtb', 'dtcb', 'dtvalb'])
        for k, names in smodels:   
            fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
            filetme.write(j.my_name() + names + 'fid,%s\n' % (fidel))

    # re-rx generator
    # consider it separately since it requires a metamodel as input
    for j in [metarf, metaxgb, metarfb, metaxgbb]: 
        genrerx.fit(X, y, j)
        ypredtest = j.predict(Xtest)
        Xnew = genrerx.sample()  
        ynew = j.predict(Xnew)
        
        if j in [metarf, metaxgb]:
            smodels = zip([dt, dtc, dtval, ripper, irep], ['dt', 'dtc', 'dtval', 'ripper', 'irep'])
        if j in [metarfb, metaxgbb]:
            smodels = zip([dtb, dtcb, dtvalb], ['dtb', 'dtcb', 'dtvalb'])
            
        for k, names in smodels:
            start = time.time()
            k.fit(Xnew, ynew)
            end = time.time()                                     
            sctrain = k.score(X, y)
            sctest = k.score(Xtest, ytest)
            bactest = balanced_accuracy_score(ytest, k.predict(Xtest))
            fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
            if names in ['ripper', 'irep']:
                nlr = len(k.ruleset_)
            else:
                nlr = n_leaves(k)
            fileres.write(names +',rerx,%s,%s,%s,%s,%s,%s,%s\n' % (j.my_name(), sctrain, sctest, nlr, (end-start), fidel, bactest)) 
        
        if j in [metarf, metaxgb]:
            ynew = j.predict_proba(Xnew) 
            for k, names in zip([primcv, bicv],['primcv', 'bicv']):
                start = time.time()
                k.fit(Xnew, ynew)
                end = time.time()                                     
                sctrain = k.score(X, y)
                sctest = k.score(Xtest, ytest)
                fileres.write(names +',rerx,%s,%s,%s,%s,%s,na,na\n' % (j.my_name(), sctrain, sctest, k.get_nrestr(), (end-start))) 

    # vva generator
    # consider it separately since it requires white-box model and metamodel
    for j in [metarf, metaxgb, metarfb, metaxgbb]:
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
        
        if j in [metarf, metaxgb]:
            smodels = zip([dt, dtc, dtval, ripper, irep, primcv, bicv],\
                          ['dt', 'dtc', 'dtval', 'ripper', 'irep', 'primcv', 'bicv'])
        if j in [metarfb, metaxgbb]:
            smodels = zip([dtb, dtcb, dtvalb], ['dtb', 'dtcb', 'dtvalb'])
        
        # optimize the number of generated points for each white-box model separately
        for k, names in smodels:
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
            filetme.write(names + j.my_name() + 'vvaopt,%s\n' % (end-start))
            filetme.write(names + j.my_name() + 'ropt,%s\n' % ropt)
            
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
            filetme.write(names + j.my_name() + 'vvagen,%s\n' % (end-start))            
                                        
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
                fidel = 'na'
                bactest = 'na'
            else:
                fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
                bactest = balanced_accuracy_score(ytest, k.predict(Xtest))
                if names in ['ripper', 'irep']:
                    nlr = len(k.ruleset_)
                else:
                    nlr = n_leaves(k)
            fileres.write(names + ',vva,%s,%s,%s,%s,%s,%s,%s\n' % (j.my_name(), sctrain, sctest, nlr, (end-start), fidel, bactest)) 

    # All remaining generators
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
              gengmmbical, gensmote, genadasyn, genrfdens, genkdem, genkdeb]:
        
        start = time.time()
        Xgen = i.sample(100000)  
        end = time.time()
        filetme.write(i.my_name() + 'gen,%s\n' % (end-start))
        
        for j in [metarf, metaxgb]:
            ypredtest = j.predict(Xtest)
            
            start = time.time()
            Xnew = Xgen.copy()  
            ynew = j.predict(Xnew)
            end = time.time()
            filetme.write(i.my_name() + j.my_name() + ',%s\n' % (end-start))  
            
            for k, names in zip([dt, dtc, dtval], ['dtp', 'dtcp', 'dtvalp']):
                start = time.time()
                k.fit(Xnew, ynew)
                end = time.time()                                     
                sctrain = k.score(X, y)
                sctest = k.score(Xtest, ytest)
                bactest = balanced_accuracy_score(ytest, k.predict(Xtest))
                fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
                fileres.write(names +',%s,%s,%s,%s,%s,%s,%s,%s\n' % (i.my_name(), j.my_name(), sctrain, sctest, n_leaves(k), (end-start), fidel, bactest)) 

            # with using original data as part of new
            Xnew = Xnew[:100000 - dsize,:]
            ynew = ynew[:100000 - dsize]
            Xnew = np.concatenate([X, Xnew])
            ynew = np.concatenate([y, ynew])            
             
            for k, names in zip([dt, dtc, dtval], ['dt', 'dtc', 'dtval']):
                start = time.time()
                k.fit(Xnew, ynew)
                end = time.time()                                     
                sctrain = k.score(X, y)
                sctest = k.score(Xtest, ytest)
                bactest = balanced_accuracy_score(ytest, k.predict(Xtest))
                fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
                fileres.write(names +',%s,%s,%s,%s,%s,%s,%s,%s\n' % (i.my_name(), j.my_name(), sctrain, sctest, n_leaves(k), (end-start), fidel, bactest)) 
    
            # smaller data for rules
            Xnew = Xnew[:10000,:]
            ynew = ynew[:10000]   
            
            for k, names in zip([ripper, irep],['ripper', 'irep']):
                start = time.time()
                k.fit(Xnew, ynew)
                end = time.time()                                     
                sctrain = k.score(X, y)
                sctest = k.score(Xtest, ytest)
                bactest = balanced_accuracy_score(ytest, k.predict(Xtest))
                fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
                fileres.write(names +',%s,%s,%s,%s,%s,%s,%s,%s\n' % (i.my_name(), j.my_name(), sctrain, sctest, len(k.ruleset_), (end-start), fidel, bactest)) 
            
            # probabilities for subgroup discovery
            ynew = j.predict_proba(Xnew)
            for k, names in zip([primcv, bicv], ['primcv', 'bicv']):
                start = time.time()
                k.fit(Xnew, ynew)
                end = time.time()                                     
                sctrain = k.score(X, y)
                sctest = k.score(Xtest, ytest)
                fileres.write(names +',%s,%s,%s,%s,%s,%s,na,na\n' % (i.my_name(), j.my_name(), sctrain, sctest, k.get_nrestr(), (end-start))) 
                
        for j in [metarfb, metaxgbb]:
            ypredtest = j.predict(Xtest)
            
            start = time.time()
            Xnew = Xgen[:100000 - dsize,:].copy()  
            ynew = j.predict(Xnew)
            Xnew = np.concatenate([X, Xnew])
            ynew = np.concatenate([y, ynew]) 
            end = time.time()
            filetme.write(i.my_name() + j.my_name() + ',%s\n' % (end-start))  
            
            for k, names in zip([dtb, dtcb, dtvalb], ['dtb', 'dtcb', 'dtvalb']):
                start = time.time()
                k.fit(Xnew, ynew)
                end = time.time()                                     
                sctrain = k.score(X, y)
                sctest = k.score(Xtest, ytest)
                bactest = balanced_accuracy_score(ytest, k.predict(Xtest))
                fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
                fileres.write(names +',%s,%s,%s,%s,%s,%s,%s,%s\n' % (i.my_name(), j.my_name(), sctrain, sctest, n_leaves(k), (end-start), fidel, bactest)) 
   
    # semi-supervised learning testing
    Xtest, ytest, Xgen = get_new_test(Xtest, ytest, dsize)
    for j in [metarf, metaxgb, metarfb, metaxgbb]: 
        ypredtest = j.predict(Xtest)
        
        ynew = np.concatenate([j.predict(Xgen), y])
        Xnew = np.concatenate([Xgen, X])
        
        if j in [metarf, metaxgb]:
            smodels = zip([dt, dtc, dtval, ripper, irep], ['dt', 'dtc', 'dtval', 'ripper', 'irep'])
        if j in [metarfb, metaxgbb]:
            smodels = zip([dtb, dtcb, dtvalb], ['dtb', 'dtcb', 'dtvalb'])
        
        for k, names in smodels:
            start = time.time()
            k.fit(Xnew, ynew)
            end = time.time()                                     
            sctrain = k.score(X, y)
            sctest = k.score(Xtest, ytest)
            bactest = balanced_accuracy_score(ytest, k.predict(Xtest))
            fidel = np.count_nonzero(k.predict(Xtest) == ypredtest)/len(ypredtest)
            if names in ['ripper', 'irep']:
                nlr = len(k.ruleset_)
            else:
                nlr = n_leaves(k)
            fileres.write(names +',ssl,%s,%s,%s,%s,%s,%s,%s\n' % (j.my_name(), sctrain, sctest, nlr, (end-start), fidel, bactest)) 
                
        if j in [metarf, metaxgb]:
            ynew = j.predict_proba(Xnew)
            for k, names in zip([primcv, bicv], ['primcv', 'bicv']):
                start = time.time()
                k.fit(Xnew, ynew)
                end = time.time()                                     
                sctrain = k.score(X, y)
                sctest = k.score(Xtest, ytest)
                fileres.write(names +',ssl,%s,%s,%s,%s,%s,na,na\n' % (j.my_name(), sctrain, sctest, k.get_nrestr(), (end-start))) 

    fileres.close()
    e_t = time.time()
    filetme.write('overall,%s\n' %(e_t-s_t))
    filetme.close()


# ==============================            Logging             ===================================


def non_interrupting_experiment(dname, dsize, splitn):
    logger = logging.getLogger('error')

    succesful = False
    stacktrace = None
    try:
        experiment(splitn, dname, dsize)
        succesful = True
    except Exception as e:
        logger.log(logging.ERROR, f'Error occured in Experiment with: splits={splitn}, dataset=${dname}, Size={dsize})')
        logger.log(logging.ERROR, traceback.format_exc())
        stacktrace = traceback.format_exc()

    return succesful, splitn, dname, dsize, stacktrace


# ==============================    Configuration & Execution      ===================================


NSETS = 25      # number of experiments with each dataset
SPLITNS = list(range(0, NSETS))         # list of experiment numbers for each dataset
DNAMES = ['clean2', 'seizure', 'gas', 'nomao', 'bankruptcy', 'anuran', 'avila', 
          'ccpp', 'cc', 'dry', 'ees', 'electricity', 'gt', 'higgs21', 'higgs7', 
          'htru', 'jm1', 'ml', 'occupancy', 'parkinson', 'pendata', 'ring',
          'saac2', 'sensorless', 'seoul', 'shuttle', 'stocks',
          'sylva', 'turbine', 'wine']       #  datasets' names
DSIZES = [400,100]         # datasets' sizes used in experiments


# run experiments on all available cores
def exp_parallel():
    args = product(DNAMES, DSIZES, SPLITNS)
    result_list = Parallel(n_jobs=cpu_count(), verbose=100)(delayed(non_interrupting_experiment)(*a) for a in args)
    print(np.asarray(result_list))


if __name__ == '__main__':
    exp_parallel()

