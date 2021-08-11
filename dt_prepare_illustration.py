import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

WHERE = 'registrydt/'
if os.path.exists(WHERE + "a.csv"):
    os.remove(WHERE + "a.csv")
else:
    print("The file does not exist")
    
_, _, filenames = next(os.walk(WHERE))


def get_acc_increase(d, tmp_times):
    d['tes'] = d['tes'] - tmp_times[tmp_times['alg'].isin(['rfacc'])]['val'].iloc[0]
    return d

#### calculate accuracy increase with respect to naive prediction
#### for a single experiment, separately for each metamodel

def get_result(fname):
    tmp_times = pd.read_csv(WHERE + fname.split(".")[0] + "_times" + '.csv',\
                            delimiter = ",", header = None)
    tmp_times.columns = ['alg', 'val']
    tmp = pd.read_csv(WHERE + fname, delimiter = ",", header = None)
    tmp.columns = ['alg', 'gen', 'met', 'tra', 'new', 'tes', 'nle','tme']
    tmp = tmp[(tmp['alg'] == 'dt')]
    tmp = get_acc_increase(tmp, tmp_times)
            
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
        if not "times" in i and not "zeros" in i:
            res.append(get_result(i))
    except:
        print("error at " + i)
res = pd.concat(res)
res.loc[res['gen'] == 'adasyns','gen'] = 'adasyn'


for i in res['dat'].unique(): 
    res1 = res.loc[res['dat'] == i]
    res1['how'] = res1['alg'] + res1['gen'] + res1['met']
    res1 = res1[res1['how'].isin(['dtnana', 'dtrandurf', 'dtkdebwrf'])][['how', 'tes', 'npt']]
    sns.boxplot(y = 'tes', x = 'npt', 
                      data = res1, 
                      palette = "colorblind",
                      hue = 'how').set_title(i)
    plt.show()


