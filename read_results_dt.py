import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

WHERE = 'registrydt/' 
if os.path.exists(WHERE + "res.csv"):
    os.remove(WHERE + "res.csv")

_, _, filenames = next(os.walk(WHERE))

#### check if all jobs have terminated

# terminated = []
# for i in filenames:
#     if not "times" in i:
#         extra = i.split(".")[0].split("_")
#         terminated.append([extra[1], extra[0], extra[2]])

# terminated = pd.DataFrame(terminated, columns=['splitn','dname', 'dsize'])
# stats = pd.pivot_table(terminated, values = "splitn", index = ['dname'],
#                     columns=['dsize'], aggfunc = pd.Series.count)

# sum((stats != 25).to_numpy()) # should be the array of zeros

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
        
    # d['npr'] = d['gen'] + d['met'] + 'prec'
    # for i in d['npr'].unique():
    #     newprec = tmp_times[tmp_times['alg'].isin([i])]['val'].iloc[0]
    #     d.loc[d['npr'] == i, 'npr'] = [newprec]*d.loc[d['npr'] == i].shape[0]
        
    d['tpr'] = [tmp_times[tmp_times['alg'].isin(['trainprec'])]['val'].iloc[0]]*d.shape[0]
    d['cva'] = [tmp_times[tmp_times['alg'].isin(['dtcvsc'])]['val'].iloc[0]]*d.shape[0]
    rf_acc = tmp_times['val'][tmp_times['alg'] == 'rfacc'].iloc[0]
    xgb_acc = tmp_times['val'][tmp_times['alg'] == 'xgbacc'].iloc[0]
    d.loc[d['met'] == 'rf', 'mac'] = [rf_acc]*d.loc[d['met'] == 'rf'].shape[0]
    d.loc[d['met'] == 'xgb', 'mac'] = [xgb_acc]*d.loc[d['met'] == 'xgb'].shape[0]
    
    return d.drop(columns = 'new')



#### calculate accuracy increase with respect to naive prediction
#### for a single experiment, separately for each metamodel

def get_result(fname):
    tmp_times = pd.read_csv(WHERE + fname.split(".")[0] + "_times" + '.csv',\
                            delimiter = ",", header = None)
    tmp_times.columns = ['alg', 'val']
    tmp = pd.read_csv(WHERE + fname, delimiter = ",", header = None)
    tmp.columns = ['alg', 'gen', 'met', 'tra', 'new', 'tes', 'nle', 'tme']
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
res.to_csv(WHERE + 'res.csv')


#### accuracy increase

res = pd.read_csv(WHERE + 'res.csv', delimiter = ",")

def draw_heatmap(*args, **kwargs):
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)
    
def draw_heatmap_c(*args, **kwargs):
    data = kwargs.pop('data')
    center = data[data['gen'] == 'No']['tes'].iloc[0]
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, center = center, **kwargs)

def draw_with_thr(thrtype = 'acc', thrvals = [0, 1, 1.05]):
    for thr in thrvals:
        a = res.copy()
        if thrtype == 'acc':
            a.loc[a['mac']/a['cva'] < thr, 'tes'] = a.loc[a['mac']/a['cva'] < thr, 'ora']
        elif thrtype == 'prec':
            a.loc[(a['npr']/a['tpr'] > (1 + thr)) | (a['npr']/a['tpr'] < (1 - thr)), 'tes']=\
                a.loc[(a['npr']/a['tpr'] > (1 + thr)) | (a['npr']/a['tpr'] < (1 - thr)), 'ora']
        else:
            a = a[(a['ora'] > thr) & ((1 - a['cva']) > thr)]
        
        a = a[['alg', 'gen', 'met', 'npt', 'tes', 'ora']].groupby(['alg', 'gen', 'met', 'npt']).mean()
        a.to_csv(WHERE + 'a.csv')
        a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")
        os.remove(WHERE + "a.csv")
        rdgn = sns.diverging_palette(h_neg = 10, h_pos = 130, s = 99, l = 55, sep = 3, as_cmap = True)
        
        # a['inc'] = (a['tes'] - a['ora'])/a['ora']
        # rdgn = sns.diverging_palette(h_neg = 10, h_pos = 130, s = 99, l = 55, sep = 3, as_cmap = True)
        # fg = sns.FacetGrid(a, col = 'alg', row = 'npt', margin_titles=True, despine=False)
        # fg.map_dataframe(draw_heatmap, 'met', 'gen', 'inc', cbar = False, cmap = rdgn, center = 0.0, annot = True, fmt ='.1%')
        # fg.set_axis_labels("metamodel", "generator")
        # fg.set_titles(col_template="{col_name}", row_template="{row_name}")
        # fg.tight_layout()
        # fg.savefig("results/dt_accuracy_inc_" + str(thr) + ".png")
        
        if thrtype != 'inc':
            tmp1 = a.drop(columns = ['tes'])
            tmp1['gen'] = 'No'
            tmp1 = tmp1.rename(columns={"ora": "tes"})
            tmp1 = tmp1.groupby(['alg', 'gen', 'met', 'npt']).mean()
            tmp1.to_csv(WHERE + 'tmp1.csv')
            tmp1 = pd.read_csv(WHERE + 'tmp1.csv', delimiter = ",")
            os.remove(WHERE + "tmp1.csv")
            a = pd.concat([tmp1, a.drop(columns = ['ora'])])
            
            fg = sns.FacetGrid(a, col = 'alg', row = 'npt', margin_titles=True, despine=False)
            fg.map_dataframe(draw_heatmap_c, 'met', 'gen', 'tes', cbar = False, cmap = rdgn, annot = True, fmt ='.1%')
            fg.set_axis_labels("metamodel", "generator")
            fg.set_titles(col_template="{col_name}", row_template="{row_name}")
            fg.tight_layout()
            fg.savefig("results/dt_accuracy_" + thrtype + str(thr) + ".png")
        
        else:
            a['inc'] = (a['tes'] - a['ora'])
            fg = sns.FacetGrid(a, col = 'alg', row = 'npt', margin_titles=True, despine=False)
            fg.map_dataframe(draw_heatmap, 'met', 'gen', 'inc', cbar = False, cmap = rdgn, center = 0.0, annot = True, fmt ='.1%')
            fg.set_axis_labels("metamodel", "generator")
            fg.set_titles(col_template="{col_name}", row_template="{row_name}")
            fg.tight_layout()
            fg.savefig("results/dt_accuracy_" + thrtype + str(thr) + ".png")

draw_with_thr(thrtype = 'acc', thrvals = [0])
# draw_with_thr(thrtype = 'prec', thrvals = [np.inf, 0.1, 0.05, 0.03])
# draw_with_thr(thrtype = 'inc', thrvals = [0, 0.05, 0.1, 0.15])











# #### number of rules decrease

# a = res.copy()
# a['dec'] = (-a['nle'] + a['orn'])/a['nle']

# rdgn = sns.diverging_palette(h_neg = 10, h_pos = 130, s = 99, l = 55, sep = 3, as_cmap = True)
# fg = sns.FacetGrid(a, col = 'alg', row = 'npt', margin_titles = True, despine = False)
# fg.map_dataframe(draw_heatmap, 'met', 'gen', 'dec', cbar = False, cmap = rdgn, center = 0.0, annot = True, fmt ='.0%')
# fg.set_axis_labels("metamodel", "generator")
# fg.set_titles(col_template="{col_name}", row_template="{row_name}")
# fg.tight_layout()
# fg.savefig("results/dt_complexity_heatmap.png")


# # FYI
# a = res.copy()
# a['dif'] = np.sign(a['tes'] - a['ora'])
# a = a[['alg', 'gen', 'met', 'npt', 'dif']]
# a = a.groupby(['alg', 'gen', 'met', 'npt']).dif.value_counts().unstack()
