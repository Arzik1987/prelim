import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

WHERE = 'registryrules/' 
if os.path.exists(WHERE + "res.csv"):
    os.remove(WHERE + "res.csv")

_, _, filenames = next(os.walk(WHERE))


# find errors
# k = 0
# for i in filenames:
#     k = k + 1
#     sys.stdout.write('\r' + "Loading" + "." + str(k))
#     if not "times" in i and not "zeros" in i:
#         tmp = pd.read_csv(WHERE + i, delimiter = ",", header = None)
#         if len(tmp) < 114:
#             print(i)
#             print(len(tmp))


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


def get_acc_increase(d, tmp_times, tmp_fid):
    precdefault = tmp_times[tmp_times['alg'].isin(['testprec'])]['val'].iloc[0]
    d['tes'] = d['tes'] - precdefault
    d['fid'] = pd.to_numeric(d['fid'], errors = 'coerce')
            
    algnames = d['alg'].unique()
    for i in algnames:
        orig_acc = d[(d['alg'] == i) & (d['gen'] =='na')]['tes']
        orig_nle = d[(d['alg'] == i) & (d['gen'] =='na')]['nle']
        d = d[~((d['alg'] == i) & (d['gen'] =='na'))]
        d.loc[d['alg'] == i,'ora'] = [orig_acc]*d.loc[d['alg'] == i].shape[0]
        d.loc[d['alg'] == i,'orn'] = [orig_nle]*d.loc[d['alg'] == i].shape[0]
     
    metnames = d['met'].unique()
    for i in algnames:
        for j in metnames:
            def_fid = tmp_fid[tmp_fid['alg'] == j + 'fid']['val'].iloc[0]
            orig_fid = tmp_times[tmp_times['alg'] == j + i + 'fid']['val'].iloc[0] - def_fid
            d.loc[(d['alg'] == i) & (d['met'] == j),'orf'] = [orig_fid]*d.loc[(d['alg'] == i) & (d['met'] == j)].shape[0]
            d.loc[(d['alg'] == i) & (d['met'] == j),'fid'] = d.loc[(d['alg'] == i) & (d['met'] == j),'fid'] - def_fid
           
    return d.drop(columns = 'new')


#### calculate accuracy increase with respect to naive prediction
#### for a single experiment, separately for each metamodel

def correct_ext(d, tmp_times):
    for i in ['ripper', 'irep']:
        d = d[~((d['alg'] == i) & (d['gen'] =='na'))]
        for j in ['rf', 'xgb']:
            tmp_times = tmp_times[~(tmp_times['alg'] == j + i + 'fid')]
            
    for i in ['ripperc', 'irepc']:
        d = d[~(d['alg'] == i)]
    
    for i in ['ripperext', 'irepext']:
        d.loc[d['alg'] == i,'alg'] = i[:-3]
        for j in ['rf', 'xgb']:
            tmp_times.loc[tmp_times['alg'] == j + i + 'fid', 'alg'] = j + i[:-3] + 'fid'
    
    return d, tmp_times


def get_result(fname):
    tmp_times = pd.read_csv(WHERE + fname.split(".")[0] + "_times" + '.csv',\
                            delimiter = ",", header = None)
    tmp_times.columns = ['alg', 'val']
    tmp_fid = pd.read_csv('registryfid/' + fname, delimiter = ",", header = None)
    tmp_fid.columns = ['alg', 'val']
    tmp = pd.read_csv(WHERE + fname, delimiter = ",", header = None)
    tmp.columns = ['alg', 'gen', 'met', 'tra', 'new', 'tes', 'nle', 'tme', 'fid']
    tmp, tmp_times = correct_ext(tmp, tmp_times)
    tmp = get_acc_increase(tmp, tmp_times, tmp_fid)
            
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

# res = pd.read_csv(WHERE + 'res.csv', delimiter = ",")

def draw_big_heatmap(clname, clnameo, mlt = 100, pal = 'normal', npts = 100, ylbl = True):
   
    def draw_heatmap_c(*args, **kwargs):
        data = kwargs.pop('data')
        center = data[data['gen'] == 'NO'][args[2]].iloc[0]
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        sns.heatmap(d, center = center, **kwargs)
    
    a = res.copy()   
    a['npt'] = pd.to_numeric(a['npt'])
    a = a[a['npt'].isin([npts])]
    a = a[a['alg'].isin(('ripper','irep'))]     
    a = a[['alg', 'gen', 'met', 'npt', clname, clnameo]].groupby(['alg', 'gen', 'met', 'npt']).mean()
    a.to_csv(WHERE + 'a.csv')
    a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")
    os.remove(WHERE + "a.csv")
    
    tmp1 = a.drop(columns = [clname])
    tmp1['gen'] = 'No'
    tmp1 = tmp1.rename(columns={clnameo: clname})
    tmp1 = tmp1.groupby(['alg', 'gen', 'met', 'npt']).max() # the choice of aggregate function (mean, min, median) should have no effect
    tmp1.to_csv(WHERE + 'tmp1.csv')
    tmp1 = pd.read_csv(WHERE + 'tmp1.csv', delimiter = ",")
    os.remove(WHERE + "tmp1.csv")
    a = pd.concat([tmp1, a.drop(columns = [clnameo])])
    
    a = a.replace('irep', 'IREP')
    a = a.replace('ripper', 'RIPPER')
    a = a.replace('cmmrf', 'cmm')
    a = a.replace('kdebw', 'kde')
    a = a.replace('kdebwm', 'kdem')
    a = a.replace('randn', 'norm')
    a = a.replace('randu', 'unif')
    a = a.replace('rerx', 're-rx')
    a = a.replace('No', 'NO')
    a = a.replace('rf', 'RF')
    a = a.replace('xgb', 'BT')
    
    a[clname] = np.round(a[clname]*mlt, 1)
    if clname == 'nle':
        a[clname][a['alg'] == 'dt'] = np.round(a[clname][a['alg'] == 'dt'], 0)
    
    if pal == 'inverse':
        rdgn = sns.diverging_palette(h_neg = 130, h_pos = 10, s = 99, l = 55, sep = 3, as_cmap = True)
    else:
        rdgn = sns.diverging_palette(h_neg = 10, h_pos = 130, s = 99, l = 55, sep = 3, as_cmap = True)
    
    asp = 0.4/1.15
    if ylbl == False:
        asp = 0.33/1.15
    fg = sns.FacetGrid(a, row = 'npt', col = 'alg', margin_titles=False, despine=False, height=4.0, aspect=asp)
    fg.map_dataframe(draw_heatmap_c, 'met', 'gen', clname, cbar = False, cmap = rdgn, annot = True, fmt='g')
    if ylbl == False:
        fg.set(yticklabels=[])
    
    # fg.set(yticks=[])
    fg.set_axis_labels("", "")
    fg.set_titles(col_template="{col_name}", row_template="{row_name}")
    fg.tight_layout()
    fg.savefig("results/rules_" + clname + str(npts) + ".pdf")
    
draw_big_heatmap('tes', 'ora', npts = 100)
draw_big_heatmap('tes', 'ora', npts = 400, ylbl = False)
draw_big_heatmap('fid', 'orf', npts = 100)
draw_big_heatmap('fid', 'orf', npts = 400, ylbl = False)
draw_big_heatmap('nle', 'orn', npts = 100, mlt = 1, pal = 'inverse')
draw_big_heatmap('nle', 'orn', npts = 400, mlt = 1, pal = 'inverse', ylbl = False)



# win-draw-loss
a = res.copy()
a = a[a['alg'].isin(('ripper','irep'))]   
a['dif'] = np.sign(a['tes'] - a['ora'])
a = a[['alg', 'gen', 'met', 'npt', 'dif']]
a = a.groupby(['alg', 'gen', 'met', 'npt']).dif.value_counts().unstack()
a.to_csv(WHERE + 'a.csv')
a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")
os.remove(WHERE + "a.csv")

a = a[a['gen'] == 'kdebw']
a = a[a['npt'].isin((100,400))]

a = a.fillna(0)
a['wdl'] = a['1.0'].astype(int).astype(str) + "/" + a['0.0'].astype(int).astype(str)\
    + "/" + a['-1.0'].astype(int).astype(str)
a = a[['alg', 'met', 'npt', 'wdl']]

a = a.pivot(index = ['met','npt'], columns=['alg'], values=['wdl'])
a.to_csv('results\\rules_pivot.csv')
