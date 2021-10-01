import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

WHERE = 'registrydt/' 
if os.path.exists(WHERE + "res.csv"):
    os.remove(WHERE + "res.csv")
    os.remove(WHERE + "res_met.csv")

_, _, filenames = next(os.walk(WHERE))

# find errors
k = 0
for i in filenames:
    k = k + 1
    sys.stdout.write('\r' + "Loading" + "." + str(k))
    if not "times" in i and not "zeros" in i:
        tmp = pd.read_csv(WHERE + i, delimiter = ",", header = None)
        if len(tmp) != 145:
            print(i)
            print(len(tmp))

# accuracy


def get_met_acc_increase(tmp_times):
    precdefault = tmp_times[tmp_times['alg'].isin(['testprec'])]['val'].iloc[0]
    data = []
    for i in range(0,len(tmp_times)):
        if 'acc' in tmp_times['alg'].iloc[i]:
            tmp_times['val'].iloc[i] = tmp_times['val'].iloc[i] - precdefault
            tmp_times['alg'].iloc[i] = tmp_times['alg'].iloc[i][:-3]
            data.append(pd.DataFrame([tmp_times.iloc[i]]))
            
    return pd.concat(data)


#### calculate accuracy increase with respect to naive prediction
#### for a single experiment, separately for each metamodel

def get_result(fname):
    tmp_times = pd.read_csv(WHERE + fname.split(".")[0] + '.csv',\
                            delimiter = ",", header = None)
    tmp_times.columns = ['alg', 'val']
    tmp = get_met_acc_increase(tmp_times)
            
    extra = fname.split(".")[0].split("_")
    tmp['dat'] = [extra[0]]*tmp.shape[0]
    tmp['itr'] = [extra[1]]*tmp.shape[0]
    tmp['npt'] = [extra[2]]*tmp.shape[0]
    return tmp

#### read and process data from all experiments with decision trees

res = []
k = 0
for i in filenames:
    k = k + 1
    sys.stdout.write('\r' + "Loading" + "." + str(k))
    try:
        if "times" in i:
            res.append(get_result(i))
    except:
        print("error at " + i)
res = pd.concat(res)
res.to_csv(WHERE + 'res_met.csv')


#### accuracy increase
# res = pd.read_csv(WHERE + 'res.csv', delimiter = ",")

a = res.copy()   
a['npt'] = pd.to_numeric(a['npt'])    
a = a[['alg', 'npt', 'val']].groupby(['alg', 'npt']).mean()
a.to_csv(WHERE + 'a.csv')
a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")
os.remove(WHERE + "a.csv")

