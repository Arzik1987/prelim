import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

WHERE = 'registrydt/' 
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


#### calculate accuracy increase with respect to naive prediction
#### for a single experiment

def get_acc_increase(d, tmp_times):
    d['tes'] = d['tes'] - tmp_times[tmp_times['alg'].isin(['testprec'])]['val'].iloc[0]
    algnames = d['alg'].unique()
    for i in algnames:
        orig_acc = d[(d['alg'] == i) & (d['gen'] =='na')]['tes']
        orig_nle = d[(d['alg'] == i) & (d['gen'] =='na')]['nle']
        d = d[~((d['alg'] == i) & (d['gen'] =='na'))]
        d.loc[d['alg'] == i,'ora'] = [orig_acc]*d.loc[d['alg'] == i].shape[0]
        d.loc[d['alg'] == i,'orn'] = [orig_nle]*d.loc[d['alg'] == i].shape[0]
    return d

#### calculate accuracy increase with respect to naive prediction
#### for a single experiment, separately for each metamodel

def get_result(fname):
    tmp_times = pd.read_csv(WHERE + fname.split(".")[0] + "_times" + '.csv',\
                            delimiter = ",", header = None)
    tmp_times.columns = ['alg', 'val']
    tmp = pd.read_csv(WHERE + fname, delimiter = ",", header = None)
    tmp.columns = ['alg', 'gen', 'met', 'tra', 'new', 'tes', 'nle','tme']
    
    cv_acc = tmp_times['val'][tmp_times['alg'] == 'dtcvsc'].iloc[0]
    rf_acc = tmp_times['val'][tmp_times['alg'] == 'rfacc'].iloc[0]
    xgb_acc = tmp_times['val'][tmp_times['alg'] == 'xgbacc'].iloc[0]
    tmp = get_acc_increase(tmp, tmp_times)
     
    if cv_acc/rf_acc > 0.95:    # in case rf does not improve over dtcv much
        tmp = tmp[(tmp['alg'] != 'dtcv') | (tmp['met'] != 'rf')]
        tmp = tmp[(tmp['alg'] != 'dtcomp') | (tmp['met'] != 'rf')]
        tmp = tmp[(tmp['alg'] != 'dtcomp2') | (tmp['met'] != 'rf')]
        
    if cv_acc/xgb_acc > 0.95:   # in case xgb does not improve over dtcv much
        tmp = tmp[(tmp['alg'] != 'dtcv') | (tmp['met'] != 'xgb')]
        tmp = tmp[(tmp['alg'] != 'dtcomp') | (tmp['met'] != 'xgb')]
        tmp = tmp[(tmp['alg'] != 'dtcomp2') | (tmp['met'] != 'rf')]
            
    extra = fname.split(".")[0].split("_")
    tmp['dat'] = [extra[0]]*tmp.shape[0]
    tmp['itr'] = [extra[1]]*tmp.shape[0]
    tmp['npt'] = [extra[2]]*tmp.shape[0]
    return tmp

#### read and process data from all experiments with decision trees

res = []
for i in filenames:
    try:
        if not "times" in i and not "zeros" in i:
            res.append(get_result(i))
    except:
        print("error at " + i)
res = pd.concat(res)
res.loc[res['gen'] == 'adasyns','gen'] = 'adasyn'


#### accuracy increase

a = res[['alg', 'gen', 'met', 'npt', 'tes', 'nle', 'ora', 'orn']].groupby(['alg', 'gen', 'met', 'npt']).mean()
a.to_csv(WHERE + 'a.csv')
a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")
os.remove(WHERE + "a.csv")
a['inc'] = (a['tes'] - a['ora'])/a['ora']

def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)

rdgn = sns.diverging_palette(h_neg = 10, h_pos = 130, s = 99, l = 55, sep = 3, as_cmap = True)
fg = sns.FacetGrid(a, col = 'alg', row = 'npt', margin_titles=True, despine=False)
fg.map_dataframe(draw_heatmap, 'met', 'gen', 'inc', cbar = False, cmap = rdgn, center = 0.0, annot = True, fmt ='.0%')
fg.set_axis_labels("metamodel", "generator")
fg.set_titles(col_template="{col_name}", row_template="{row_name}")
fg.tight_layout()
fg.savefig("results/dt_accuracy_heatmap.png")

#### check values:
# df = a[(a['npt'] == 800) & (a['alg'] == 'dtcomp')]
# dfna = df.pivot('gen', 'met', 'tes')
# dfoa = df.pivot('gen', 'met', 'ora')
# (dfna - dfoa)/dfoa

#### number of rules decrease

a['dec'] = (-a['nle'] + a['orn'])/a['nle']

rdgn = sns.diverging_palette(h_neg = 10, h_pos = 130, s = 99, l = 55, sep = 3, as_cmap = True)
fg = sns.FacetGrid(a, col = 'alg', row = 'npt', margin_titles = True, despine = False)
fg.map_dataframe(draw_heatmap, 'met', 'gen', 'dec', cbar = False, cmap = rdgn, center = 0.0, annot = True, fmt ='.0%')
fg.set_axis_labels("metamodel", "generator")
fg.set_titles(col_template="{col_name}", row_template="{row_name}")
fg.tight_layout()
fg.savefig("results/dt_complexity_heatmap.png")


# an interesting fact is that although the relative improvement drops with 'npt',
# this improvement is more consistent over the experiments:
a = res.copy()
a['dif'] = np.sign(a['tes'] - a['ora'])
a = a[['alg', 'gen', 'met', 'npt', 'dif']]
a = a.groupby(['alg', 'gen', 'met', 'npt']).dif.value_counts().unstack()
