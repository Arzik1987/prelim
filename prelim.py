from sklearn.utils.validation import check_X_y, check_is_fitted

def prelim(X, y, bb_model, wb_model, gen_name, new_size, proba = False, verbose = True):
    # X - np aray of feature values
    # y - binary class label from {0,1}
    # bb_model trained or not trained complex classifier
    # 

    X, y = check_X_y(X, y)
    if not hasattr(bb_model, "classes_"):
        if verbose == True:
            print("fitting bb_model to data")
        bb_model.fit(X, y)
    
    if not hasattr(bb_model, 'predict_proba'):
        if gen_name in ['vva']:
            # TODO add an exceptions below
            print('vva needs method "predict_proba()" method of bb_model')
        if proba == True:
            print('bb_model does not have "predict_proba()" method.\
                  Set "proba = False" or use another bb_model')    
                  
    if gen_name == gmm:
        from src.generators.gmm import Gen_gmmbic
        gen = Gen_gmmbic()
    elif gen_name == gmmal:
        from src.generators.gmm import Gen_gmmbical
        gen = Gen_gmmbical()
            
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
    genrerx = Gen_rerx()
    

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
    
    if gen == 'vva':
        ntrain = int(np.ceil(X.shape[0]*2/3))
        Xtrain = X[:ntrain,:].copy()
        Xval = X[ntrain:,:].copy()
        ytrain = y[:ntrain].copy()
        yval = y[ntrain:].copy()
        genvva.fit(Xtrain, j)


        
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
    
    
    
        
    


        

import numpy as np
from sklearn.ensemble import RandomForestClassifier

mean = [0, 0]
cov = [[1, 0], [0, 1]]
X = np.random.multivariate_normal(mean, cov, 500)
mean = [3,3]
X = np.vstack((X, np.random.multivariate_normal(mean, cov, 500)))
y = np.hstack((np.zeros(500), np.ones(500))).astype(int)
rf = RandomForestClassifier()
rf.fit(X, y)

if not hasattr(rf, 'predict_proba'):
    print(name)
    
isinstance(rf, RandomForestClassifier)
