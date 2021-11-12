from sklearn.utils.validation import check_X_y, check_is_fitted

def prelim(X, y, bb_model, wb_model, gen, new_size, proba = False, verbose = True):
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
        if gen in ['vva']:
            # TODO add an exceptions below
            print('vva needs method "predict_proba()" method of bb_model')
        if proba == True:
            print('bb_model does not have "predict_proba()" method.\
                  Set "proba = False" or use another bb_model')    
    
    if gen == 'vva':
        
    
    
    
        
    


        

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
