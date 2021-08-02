import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

WHERE = 'registryrules/' 
_, _, filenames = next(os.walk(WHERE))

#### check if all jobs have terminated

terminated = []
for i in filenames:
    if not "times" in i:
        extra = i.split(".")[0].split("_")
        terminated.append([extra[1], extra[0], extra[2]])

terminated = pd.DataFrame(terminated, columns=['splitn','dname', 'dsize'])
stats = pd.pivot_table(terminated, values = "splitn", index = ['dname'],
                    columns=['dsize'], aggfunc = pd.Series.count)

sum((stats != 25).to_numpy()) # should be the array of zeros

# rdgn = sns.diverging_palette(h_neg = 130, h_pos = 10, s = 99, l = 55, sep = 3, as_cmap = True)
# sns.heatmap(stats, cmap = rdgn, center = 25, annot = True)

#### which datasets take the most time:

def get_result(fname):
    tmp = pd.read_csv(WHERE + fname, delimiter = ",", header = None)
    tmp.columns = ['alg', 'gen', 'met', 'tra', 'new', 'tes', 'nle','tme']
    tmp = sum(tmp['tme'])
    extra = fname.split(".")[0].split("_")
    
    return pd.DataFrame(data = {'dat': [extra[0]], 'itr': [extra[1]], 'npt': [extra[2]], 'tme': [tmp]})

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

a = res.groupby(['dat']).mean()
a.to_csv(WHERE + 'a.csv')
a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")
os.remove(WHERE + "a.csv")

a = a.sort_values('tme',  ascending = False)
tmp = np.cumsum(a['tme']).to_numpy()
tmp/tmp[-1] # so 4 datasets require 77% of time (Close to Pareto principle!)

tcf = a.head(4)['dat']


#### calculate accuracy increase with respect to naive prediction
#### for a single experiment

def get_acc_increase(d, tmp_times):
    d['tes'] = d['tes'] - tmp_times[tmp_times['alg'].isin(['testprec'])]['val'].iloc[0]
    algnames = d['alg'].unique()
    for i in algnames:
        orig_acc = d[(d['alg'] == i) & (d['gen'] =='na')]['tes']
        d = d[~((d['alg'] == i) & (d['gen'] =='na'))]
        d.loc[d['alg'] == i,'ora'] = [orig_acc]*d.loc[d['alg'] == i].shape[0]
    d['npr'] = d['gen'] + d['met'] + 'prec'
    trainprec = tmp_times[tmp_times['alg'].isin(['trainprec'])]['val'].iloc[0]
    d['tpr'] = [trainprec]*d.shape[0]
    for i in d['npr'].unique():
        newprec = tmp_times[tmp_times['alg'].isin([i])]['val'].iloc[0]
        d.loc[d['npr'] == i, 'npr'] = [newprec]*d.loc[d['npr'] == i].shape[0]
    return d.drop(columns = 'nle')

#### calculate accuracy increase with respect to naive prediction
#### for a single experiment, separately for each metamodel

def get_result(fname):
    tmp_times = pd.read_csv(WHERE + fname.split(".")[0] + "_times" + '.csv',\
                            delimiter = ",", header = None)
    tmp_times.columns = ['alg', 'val']
    tmp = pd.read_csv(WHERE + fname, delimiter = ",", header = None)
    tmp.columns = ['alg', 'gen', 'met', 'tra', 'new', 'tes', 'nle','tme']
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

#### if we exclude time-consuming experiments, will it affect much?

res = res[~res['dat'].isin(tcf)]


#### accuracy increase

a = res[['alg', 'gen', 'met', 'npt', 'tes', 'ora']].groupby(['alg', 'gen', 'met', 'npt']).mean()
a.to_csv(WHERE + 'a.csv')
a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")
os.remove(WHERE + "a.csv")
tmp1 = a.drop(columns = 'tes')
tmp1['gen'] = 'No'
tmp1 = tmp1.rename(columns={"ora": "tes"})
tmp1 = tmp1.groupby(['alg', 'gen', 'met', 'npt']).mean()
tmp1.to_csv(WHERE + 'tmp1.csv')
tmp1 = pd.read_csv(WHERE + 'tmp1.csv', delimiter = ",")
os.remove(WHERE + "tmp1.csv")
tmp2 = a.drop(columns = 'ora')
a = pd.concat([tmp1, tmp2])
a['am'] = a['alg'] + a['met']
a = a.drop(columns = ['alg', 'met'])

def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    center = data[data['gen'] == 'No']['tes'].iloc[0]
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, center = center, **kwargs)

rdgn = sns.diverging_palette(h_neg = 10, h_pos = 130, s = 99, l = 55, sep = 3, as_cmap = True)
# fg = sns.FacetGrid(a, col = 'alg', row = 'npt', margin_titles=True, despine=False)
fg = sns.FacetGrid(a, col = 'am', row = 'npt', margin_titles=True, despine=False)
fg.map_dataframe(draw_heatmap, 'am', 'gen', 'tes', cbar = False, cmap = rdgn, annot = True, fmt ='.1%')
fg.set_axis_labels("metamodel", "generator")
fg.set_titles(col_template="{col_name}", row_template="{row_name}")
fg.tight_layout()
fg.savefig("results/rules_accuracy_heatmap_short.png")

#### check values:
# df = a[(a['npt'] == 800) & (a['alg'] == 'dtcomp')]
# dfna = df.pivot('gen', 'met', 'tes')
# dfoa = df.pivot('gen', 'met', 'ora')
# (dfna - dfoa)/dfoa



# an interesting fact is that although the relative improvement drops with 'npt',
# this improvement is more consistent over the experiments:
a = res.copy()
a['dif'] = np.sign(a['tes'] - a['ora'])
a = a[['alg', 'gen', 'met', 'npt', 'dif']]
a = a.groupby(['alg', 'gen', 'met', 'npt']).dif.value_counts().unstack()
