from sklearn.utils.validation import check_X_y#, check_is_fitted
import numpy as np

def prelim(X, y, bb_model, wb_model, gen_name, new_size, proba = False, verbose = True):
    # X - np aray of feature values
    # y - binary class label from {0,1}
    # bb_model trained or not trained black-box model
    # wb_model white-box model to be fitted
    # gen_name - name of generator to use
    # new_size - size of the the dataset to be used for wb_model. Ignored for 'rerx', 'vva', 'dummy' generators
    # proba - if the black-box model should output class probabilities
    # verbose - if the function should print messages

    X, y = check_X_y(X, y)
    if not hasattr(bb_model, 'classes_'):
        if verbose == True:
            print('fitting bb_model to data')
        bb_model.fit(X, y)
    
    if not hasattr(bb_model, 'predict_proba'):
        if gen_name in ['vva']:
            # TODO add exceptions below
            print('vva needs method "predict_proba()" method of bb_model')
        if proba == True:
            print('bb_model does not have "predict_proba()" method.\
                  Set "proba = False" or use another bb_model')    
    # Load proper generator      
    if gen_name == 'adasyn':
        from .generators.adasyn import Gen_adasyn
        gen = Gen_adasyn()
    if gen_name == 'cmm':
        from .generators.rfdens import Gen_rfdens
        gen = Gen_rfdens()
    if gen_name == 'dummy':
        from .generators.dummy import Gen_dummy
        gen = Gen_dummy()          
    if gen_name == 'gmm':
        from .generators.gmm import Gen_gmmbic
        gen = Gen_gmmbic()
    elif gen_name == 'gmmal':
        from .generators.gmm import Gen_gmmbical
        gen = Gen_gmmbical()
    if gen_name == 'kde':
        from .generators.kde import Gen_kdebw
        gen = Gen_kdebw() 
    if gen_name == 'kdeb':
        from .generators.kdeb import Gen_kdeb
        gen = Gen_kdeb() 
    if gen_name == 'kdem':
        from .generators.kdem import Gen_kdebwm
        gen = Gen_kdebwm()   
    if gen_name == 'munge':
        from .generators.munge import Gen_munge
        gen = Gen_munge()                
    if gen_name == 'norm':
        from .generators.rand import Gen_randn
        gen = Gen_randn()  
    if gen_name == 'rerx':
        from .generators.rerx import Gen_rerx
        gen = Gen_rerx()  
    if gen_name == 'smote':
        from .generators.smote import Gen_smote
        gen = Gen_smote()  
    if gen_name == 'unif':
        from .generators.rand import Gen_randu
        gen = Gen_randu()
    if gen_name == 'vva':
        from .generators.vva_p import Gen_vva
        gen = Gen_vva()

    #### vva generator
    if gen_name == 'vva':
        ntrain = int(np.ceil(X.shape[0]*2/3))
        Xtrain = X[:ntrain,:].copy()
        Xval = X[ntrain:,:].copy()
        ytrain = y[:ntrain].copy()
        yval = y[ntrain:].copy()
        gen.fit(Xtrain, metamodel = bb_model)
        
        #### optimizing the share of generated points r
        # score for no generation
        if proba:
            wb_model.fit(Xtrain, bb_model.predict_proba(Xtrain)[:, int(np.where(bb_model.classes_ == 1)[0])])
        else:
            wb_model.fit(Xtrain, ytrain)
        sctest0 = wb_model.score(Xval, yval)
        ropt = 0
        
        if gen.will_generate():  # black-box model does not predict a single class
            for r in np.linspace(0.5, 2.5, num = 5):
                Xnew = gen.sample(r)    
                if proba:
                    Xnew = np.concatenate([Xnew, Xtrain])
                    ynew = bb_model.predict_proba(Xnew)[:, int(np.where(bb_model.classes_ == 1)[0])]
                else:
                    ynew = bb_model.predict(Xnew) 
                    Xnew = np.concatenate([Xnew, Xtrain])
                    ynew = np.concatenate([ynew, ytrain])
                    
                wb_model.fit(Xnew, ynew)
                sctest = wb_model.score(Xval, yval)
                if sctest > sctest0:
                    sctest0 = sctest
                    ropt = r
        
        #### generate points from fitted vva with optimal r
        if ropt > 0:
            Xnew = gen.fit(X, metamodel = bb_model).sample(ropt)
            if proba:
                Xnew = np.concatenate([Xnew, X])
                ynew = bb_model.predict_proba(Xnew)[:, int(np.where(bb_model.classes_ == 1)[0])]
            else:
                ynew = bb_model.predict(Xnew) 
                Xnew = np.concatenate([Xnew, X])
                ynew = np.concatenate([ynew, y])
                
            wb_model.fit(Xnew, ynew)
        else: # no points to generate
            wb_model.fit(X, y)
    
    else: # not vva generator
        gen.fit(X, y, metamodel = bb_model)
        Xnew = gen.sample(new_size - len(y))
        if proba:
            if gen_name != 'rerx':
                Xnew = np.concatenate([Xnew, X])
            ynew = bb_model.predict_proba(Xnew)[:, int(np.where(bb_model.classes_ == 1)[0])]
        else:
            ynew = bb_model.predict(Xnew) 
            if gen_name != 'rerx':
                Xnew = np.concatenate([Xnew, X])
                ynew = np.concatenate([ynew, y])
                
        wb_model.fit(Xnew, ynew)
      
    return wb_model
                                    
     
# =============================================================================
# #### test
# 
# npt = 50
# cov = [[1, 0], [0, 1]]
# X = np.vstack((np.random.multivariate_normal([0, 0], cov, npt), np.random.multivariate_normal([1, 1], cov, npt)))
# y = np.hstack((np.zeros(npt), np.ones(npt))).astype(int)
# 
# Xtest = np.vstack((np.random.multivariate_normal([0, 0], cov, 100*npt), np.random.multivariate_normal([1, 1], cov, 100*npt)))
# ytest = np.hstack((np.zeros(100*npt), np.ones(100*npt))).astype(int)
# 
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# 
# wb_model = prelim(X, y, RandomForestClassifier(), DecisionTreeClassifier(max_leaf_nodes = 8),\
#                   'kde', new_size = 2000, proba = False, verbose = True)
# print('prelim_kde_score = %s' % wb_model.score(Xtest, ytest))
# 
# wb_model = DecisionTreeClassifier(max_leaf_nodes = 8).fit(X,y)
# print('orig_score = %s' % wb_model.score(Xtest, ytest))
# =============================================================================




