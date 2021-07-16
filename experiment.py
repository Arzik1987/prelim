import os

os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

# from src.metamodels.kriging import Meta_kriging
# from src.metamodels.nb import Meta_nb
from src.metamodels.rf import Meta_rf
# from src.metamodels.svm import Meta_svm
from src.metamodels.xgb import Meta_xgb

from src.generators.gmm import Gen_gmmbic#, Gen_gmm
from src.generators.kde import Gen_kdebw#, Gen_kdebwhl
from src.generators.munge import Gen_munge
from src.generators.noise import Gen_noise
from src.generators.rand import Gen_randn, Gen_randu
from src.generators.dummy import Gen_dummy
from src.generators.smote import Gen_smote
from src.generators.adasyn import Gen_adasyn

# from src.subgroup_discovery.BI import BI
from src.subgroup_discovery.PRIM import PRIM
from sklearn.tree import DecisionTreeClassifier
# import wittgenstein as lw

from src.utils.data_splitter import DataSplitter
from src.utils.data_loader import load_data
# from src.utils.my_transform import My_transform

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import numpy as np
from multiprocessing import Pool
from itertools import product
import time

if not os.path.exists('registry'):
    os.makedirs('registry')

NSETS = 25                                                               # number of splits
SPLITNS = list(range(0, NSETS))
# DNAMES = ["occupancy"]
DNAMES = ["occupancy", "higgs7", "electricity", "htru", "shuttle", "avila",
          "cc", "ees", "pendata", "ring", "sylva", "higgs21",
          "jm1", "saac2", "stocks", 
          "sensorless", "bankruptcy", "nomao",
          "ccpp", "seoul", "turbine", "wine", "parkinson", "dry", "anuran", "ml"]
# DSIZES = [100]
DSIZES = [100, 200, 800]


def opt_param(cvres, nval):
    fit_res = np.empty((0, nval))
    for key, value in cvres.items():
        if 'split' in key:
            fit_res = np.vstack((fit_res, value))
    tmp = np.nanmean(fit_res, 0)
    return np.argmax(tmp), tmp[np.argmax(tmp)]

def get_bi_param(nval, nattr):
    a = [ -x for x in range(-nattr, 0, np.ceil(nattr/nval).astype(int))]
    b = [ -x for x in range(-nattr, min(-nattr + nval, 0), 1)]
    res = a if len(a) > nval/2 + 1 else b
    return res


def experiment_class(splitn, dname, dsize):                                                                              
    gengmmbic = Gen_gmmbic() 
    genkde = Gen_kdebw()
    # genkdehl = Gen_kdebwhl()
    genmunge = Gen_munge()
    genrandu = Gen_randu()
    genrandn = Gen_randn()
    gendummy = Gen_dummy()
    gennoise = Gen_noise()
    genadasyn = Gen_adasyn()
    gensmote = Gen_smote()
                                               
    metarf = Meta_rf()
    metaxgb = Meta_xgb()
    # metanb = Meta_nb()
    
    dt = DecisionTreeClassifier()
    dt_comp = DecisionTreeClassifier(max_depth = 3)
    # ripper = lw.RIPPER(max_rules = 8)
    # irep = lw.IREP(max_rules = 8)
    prim = PRIM()
    
    # get datasets
    X, y = load_data(dname)
    ds = DataSplitter()                                                 
    ds.fit(X, y)                                                    
    ds.configure(NSETS, dsize)                                         
    X, y = ds.get_train(splitn)                                           
    Xtest, ytest = ds.get_test(splitn) 
    filetme = open("registry/" + dname + "_" + "%s" % splitn + "_" + "%s" % dsize + "_times.csv", "a")
    filetme.write("testprec" + "," + "%s" % max(ytest.mean(), 1-ytest.mean()) + "\n") 
    filetme.write("trainprec" + "," + "%s" % max(y.mean(), 1-y.mean()) + "\n") 
    filetme.close()                                 

    fileres = open("registry/" + dname + "_" + "%s" % splitn + "_" + "%s" % dsize + ".csv", "a")
    start = time.time()
    dt.fit(X, y)
    end = time.time()                                                  
    sctrain = dt.score(X, y)
    sctest = dt.score(Xtest, ytest)
    # (1) model (2) gen (3) met (4) sctr (5) scnew (6) sctest (7) time
    fileres.write("dt" + ",na,na," + "%s" % sctrain + ",nan," + "%s" % sctest + "," + "%s" % (end-start)) 
    # 
    start = time.time()
    dt_comp.fit(X, y)  
    end = time.time()                                                 
    sctrain = dt_comp.score(X, y)
    sctest = dt_comp.score(Xtest, ytest)
    fileres.write("\n" + "dt_comp" + ",na,na," + "%s" % sctrain + ",nan," + "%s" % sctest + "," + "%s" % (end-start)) 
    # 
    # start = time.time()
    # ripper.fit(X, y)
    # end = time.time()                                                   
    # sctrain = ripper.score(X, y)
    # sctest = ripper.score(Xtest, ytest)
    # fileres.write("\n" + "ripper" + ",na,na," + "%s" % sctrain + ",nan," + "%s" % sctest + "," + "%s" % (end-start)) 
    # # 
    # start = time.time()
    # irep.fit(X, y)
    # end = time.time()                                                   
    # sctrain = irep.score(X, y)
    # sctest = irep.score(Xtest, ytest)
    # fileres.write("\n" + "irep" + ",na,na," + "%s" % sctrain + ",nan," + "%s" % sctest + "," + "%s" % (end-start)) 
    fileres.close() 
    
    # DT HPO
    fileres = open("registry/" + dname + "_" + "%s" % splitn + "_" + "%s" % dsize + ".csv", "a")
    par_vals = [1,2,3,4,5,6,7,None]
    parameters = {'max_depth': par_vals}   
    start = time.time()                           
    tmp = GridSearchCV(dt, parameters, refit = False).fit(X, y).cv_results_ 
    tmp = opt_param(tmp, len(par_vals))
    dtcv = DecisionTreeClassifier(max_depth = par_vals[tmp[0]])                                            
    dtcv.fit(X, y)
    end = time.time()
    sctrain = dtcv.score(X, y)
    sctest = dtcv.score(Xtest, ytest)     
    fileres.write("\n" + "dtcv" + ",na,na," + "%s" % sctrain + ",nan," + "%s" % sctest + "," + "%s" % (end-start)) 
    fileres.close()
    
    filetme = open("registry/" + dname + "_" + "%s" % splitn + "_" + "%s" % dsize + "_times.csv", "a")
    filetme.write("dtcvsc" + "," + "%s" % tmp[1] + "\n") 
    filetme.close()
    
    # PRIM 
    fileprim = open("registry/" + dname + "_" + "%s" % splitn + "_" + "%s" % dsize + "_prim.csv", "a")
    start = time.time()
    prim.fit(X, y)
    end = time.time()                                                  
    sctrain = prim.score(X, y)
    pr_train = prim.get_pr()
    sctest = prim.score(Xtest, ytest)
    pr_test = prim.get_pr()
    # (1) model (2) gen (3) met (4) sctr (5) scnew (6) sctest (7) time (8-10) prec (11-13) rec
    fileprim.write("prim" + ",na,na," + "%s" % sctrain + ",nan," + "%s" % sctest + "," 
                   + "%s" % (end-start) + "," + pr_train[0] + ",nan," + pr_test[0] + "," +
                   pr_train[1] + ",nan," + pr_test[1]) 
    fileprim.close()
    
    # PRIM HPO
    fileprim = open("registry/" + dname + "_" + "%s" % splitn + "_" + "%s" % dsize + "_prim.csv", "a")
    par_vals = [0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2]
    parameters = {'alpha': par_vals}
    start = time.time()     
    tmp = GridSearchCV(prim, parameters, refit = False).fit(X, y).cv_results_ 
    tmp = opt_param(tmp, len(par_vals))
    primcv = PRIM(alpha = par_vals[tmp[0]])                                          
    primcv.fit(X, y)
    end = time.time()
    sctrain = primcv.score(X, y)
    pr_train = primcv.get_pr()
    sctest = primcv.score(Xtest, ytest)
    pr_test = primcv.get_pr()  
    fileprim.write("\n" + "primcv" + ",na,na," + "%s" % sctrain + ",nan," + "%s" % sctest + "," 
                   + "%s" % (end-start) + "," + pr_train[0] + ",nan," + pr_test[0] + "," +
                   pr_train[1] + ",nan," + pr_test[1]) 
    fileprim.close()
    
    
    # prelim
    ss = StandardScaler()                                               
    ss.fit(X)                                                       
    Xs = ss.transform(X) 
                                                
    for i in [gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
              gennoise, gensmote, genadasyn]:
        filetme = open("registry/" + dname + "_" + "%s" % splitn + "_" + "%s" % dsize + "_times.csv", "a")                                      
        start = time.time()
        i.fit(Xs)
        end = time.time()
        filetme.write(i.my_name() + "," + "%s" % (end-start) + "\n") 
        filetme.close()
        
    for j in [metarf, metaxgb]: 
        filetme = open("registry/" + dname + "_" + "%s" % splitn + "_" + "%s" % dsize + "_times.csv", "a")                                     
        start = time.time()
        j.fit(Xs, y)
        end = time.time()
        filetme.write(j.my_name() + "," + "%s" % (end-start) + "\n") 
        filetme.write(j.my_name() + "acc," + "%s" % j.fit_score() + "\n") 
        filetme.close()
        
    for i, j in product([gengmmbic, genkde, genmunge, genrandu, genrandn, gendummy,\
                         gennoise, gensmote, genadasyn], [metarf, metaxgb]):
        filetme = open("registry/" + dname + "_" + "%s" % splitn + "_" + "%s" % dsize + "_times.csv", "a")              

        start = time.time()
        Xnew = i.sample(100000)                                                                      
        ynew = j.predict(Xnew)
        Xnew = ss.inverse_transform(Xnew)      
        # Xnews = i.sample(10000) # RIPPER and IREP                                                            
        # ynews = j.predict(Xnews)
        # Xnews = ss.inverse_transform(Xnews)  
        end = time.time()
        filetme.write(i.my_name() + j.my_name() + "," + "%s" % (end-start) + "\n")  
        filetme.write(i.my_name() + j.my_name() + "prec" + "," + "%s" % max(ynew.mean(), 1-ynew.mean()) + "\n")
        filetme.close()                             
        
        fileres = open("registry/" + dname + "_" + "%s" % splitn + "_" + "%s" % dsize + ".csv", "a")
        start = time.time()
        dt.fit(Xnew, ynew)
        end = time.time()                                     
        sctrain = dt.score(X, y)
        scnew = dt.score(Xnew, ynew)
        sctest = dt.score(Xtest, ytest)                                       
        fileres.write("\n" + "dt" + "," + i.my_name() + "," + j.my_name() + 
                      "," + "%s" % sctrain + "," + "%s" % scnew + "," + "%s" % sctest + "," + "%s" % (end-start)) 
        
        start = time.time()
        dt_comp.fit(Xnew, ynew) 
        end = time.time()                                    
        sctrain = dt_comp.score(X, y)
        scnew = dt_comp.score(Xnew, ynew)
        sctest = dt_comp.score(Xtest, ytest)                                       
        fileres.write("\n" + "dt_comp" + "," + i.my_name() + "," + j.my_name() + 
                      "," + "%s" % sctrain + "," + "%s" % scnew + "," + "%s" % sctest + "," + "%s" % (end-start)) 
        
        # start = time.time()
        # ripper.fit(Xnews, ynews)
        # end = time.time()                                     
        # sctrain = ripper.score(X, y)
        # scnew = ripper.score(Xnews, ynews)
        # sctest = ripper.score(Xtest, ytest)                                       
        # fileres.write("\n" + "ripper" + "," + i.my_name() + "," + j.my_name() + 
        #               "," + "%s" % sctrain + "," + "%s" % scnew + "," + "%s" % sctest + "," + "%s" % (end-start)) 
        
        # start = time.time()
        # irep.fit(Xnews, ynews)    
        # end = time.time()                                 
        # sctrain = irep.score(X, y)
        # scnew = irep.score(Xnews, ynews)
        # sctest = irep.score(Xtest, ytest)                                       
        # fileres.write("\n" + "irep" + "," + i.my_name() + "," + j.my_name() + 
        #               "," + "%s" % sctrain + "," + "%s" % scnew + "," + "%s" % sctest + "," + "%s" % (end-start)) 
        
        start = time.time()
        dtcv.fit(Xnew, ynew)    
        end = time.time()                                 
        sctrain = dtcv.score(X, y)
        scnew = dtcv.score(Xnew, ynew)
        sctest = dtcv.score(Xtest, ytest)                                       
        fileres.write("\n" + "dtcv" + "," + i.my_name() + "," + j.my_name() + 
                      "," + "%s" % sctrain + "," + "%s" % scnew + "," + "%s" % sctest + "," + "%s" % (end-start))         
        
        par_vals = [1,2,3,4,5,6,7,None]
        parameters = {'max_depth': par_vals}   
        start = time.time()                           
        tmp = GridSearchCV(dt, parameters, refit = False).fit(Xnew, ynew).cv_results_ 
        tmp = opt_param(tmp, len(par_vals))
        dtcv2 = DecisionTreeClassifier(max_depth = par_vals[tmp[0]])                                            
        dtcv2.fit(Xnew, ynew)
        end = time.time()
        sctrain = dtcv2.score(X, y)
        scnew = dtcv2.score(Xnew, ynew)
        sctest = dtcv2.score(Xtest, ytest)                                       
        fileres.write("\n" + "dtcv2" + "," + i.my_name() + "," + j.my_name() + 
                      "," + "%s" % sctrain + "," + "%s" % scnew + "," + "%s" % sctest + "," + "%s" % (end-start)) 
        fileres.close()      
        
        
        fileprim = open("registry/" + dname + "_" + "%s" % splitn + "_" + "%s" % dsize + "_prim.csv", "a")
        start = time.time()
        prim.fit(Xnew, ynew)
        end = time.time()                                                  
        sctrain = prim.score(X, y)
        pr_train = prim.get_pr()
        scnew = prim.score(Xnew, ynew)
        pr_new = prim.get_pr()
        sctest = prim.score(Xtest, ytest)
        pr_test = prim.get_pr()
        fileprim.write("\n" + "prim" + "," + i.my_name() + "," + j.my_name() + "," + "%s" % sctrain + ",nan," + "%s" % sctest + "," 
                       + "%s" % (end-start) + "," + pr_train[0] + "," + pr_new[0] + "," + pr_test[0] + "," +
                       pr_train[1] + "," + pr_new[1] + "," + pr_test[1]) 
      
        start = time.time()
        primcv.fit(Xnew, ynew)
        end = time.time()                                                  
        sctrain = primcv.score(X, y)
        pr_train = primcv.get_pr()
        scnew = primcv.score(Xnew, ynew)
        pr_new = primcv.get_pr()
        sctest = primcv.score(Xtest, ytest)
        pr_test = primcv.get_pr()
        fileprim.write("\n" + "primcv" + "," + i.my_name() + "," + j.my_name() + "," + "%s" % sctrain + ",nan," + "%s" % sctest + "," 
                       + "%s" % (end-start) + "," + pr_train[0] + "," + pr_new[0] + "," + pr_test[0] + "," +
                       pr_train[1] + "," + pr_new[1] + "," + pr_test[1])     
        fileprim.close()
        

def exp_parallel():
    # pool = Pool(2)
    pool = Pool(32)
    pool.starmap(experiment_class, list(product(SPLITNS, DNAMES, DSIZES)))
    pool.close()
    pool.join()

if __name__ == "__main__":
    exp_parallel()

