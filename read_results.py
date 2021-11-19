import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

WHERE = 'registry/' 
if os.path.exists(WHERE + "res.csv"):
    os.remove(WHERE + "res.csv")

_, _, filenames = next(os.walk(WHERE))

# #### delete after correction of the code for experiments.py

# k = 0
# for fname in filenames:
#     k = k + 1
#     sys.stdout.write('\r' + "Loading" + "." + str(k))
#     if not 'meta' in fname:
#         with open(WHERE + fname) as file:
#             lines = file.readlines()
        
#         lines = [line.replace('nan,','') if 'dtval,na,na' in line else line for line in lines] 
#         lines = [line.replace('rerx',',rerx') for line in lines] 
#         lines = [line.replace(',,',',') for line in lines] 
        
#         with open(WHERE + fname, 'w') as file:
#             lines = file.writelines(lines)

#### Helper functions for postprocessing

def qual_change(d, meta):
    # calculates accuracy and fidelity increase with respect to naive prediction
    # for a single experiment
    
    precdefault = meta[meta['alg'].isin(['testprec'])]['val'].iloc[0]
    for i in range(0, d.shape[0]):
        if d.iloc[i][0] not in ['primcv', 'bicv']:
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
            if i not in ['primcv', 'bicv']:
                def_fid = meta[meta['alg'] == j + 'fid']['val'].iloc[0]
                orig_fid = meta[meta['alg'] == j + i + 'fid']['val'].iloc[0] - def_fid
                d.loc[(d['alg'] == i) & (d['met'] == j),'orf'] = [orig_fid]*d.loc[(d['alg'] == i) & (d['met'] == j)].shape[0]
                d.loc[(d['alg'] == i) & (d['met'] == j),'fid'] = d.loc[(d['alg'] == i) & (d['met'] == j),'fid'] - def_fid
            else:
                d.loc[(d['alg'] == i) & (d['met'] == j),'orf'] = [np.nan]*d.loc[(d['alg'] == i) & (d['met'] == j)].shape[0]

    return d.copy()


def get_result(fname):
    # processes the results from a single experiment
    meta = pd.read_csv(WHERE + fname.split(".")[0] + "_meta" + '.csv',\
                            delimiter = ",", header = None)
    meta.columns = ['alg', 'val']
    
    d = pd.read_csv(WHERE + fname, delimiter = ",", header = None)
    d.columns = ['alg', 'gen', 'met', 'tra', 'tes', 'nle', 'tme', 'fid']
    d = qual_change(d, meta)
            
    extra = fname.split(".")[0].split("_")
    d['dat'] = [extra[0]]*d.shape[0]
    d['itr'] = [extra[1]]*d.shape[0]
    d['npt'] = [extra[2]]*d.shape[0]
    return d


#### Postprocessing: read and process data from all experiments with decision trees

res = []
k = 0
nf = str(len(filenames))
for i in filenames:
    k = k + 1
    sys.stdout.write('\r' + 'loading' + '.' + str(k) + '/' + nf)
    if not 'meta' in i:
            res.append(get_result(i))


res = pd.concat(res)
res.loc[res['gen'] == 'adasyns','gen'] = 'adasyn'
for names in ['tra', 'tes', 'nle', 'tme', 'fid', 'ora', 'orn', 'orf', 'itr', 'npt']:
    print(names)
    res[names] = pd.to_numeric(res[names])
res.to_csv(WHERE + 'res.csv')
# res = pd.read_csv(WHERE + 'res.csv', delimiter = ",")


#### Helper functions for drawing

def res_aggregate(mod, npts, clname, clnameo):
    a = res.copy()   
    a = a[a['npt'].isin([npts])]
    if mod == 'dt':
        nms = ('dt','dtc', 'dtval')
    elif mod == 'rules':
        nms = ('ripper','irep')
    elif mod == 'sd':
        nms = ('primcv','bicv')
    else:
        raise ValueError('{mod} is a wrong mod value'.format(dname = repr(dname)))
    a = a[a['alg'].isin(nms)]     
    a = a[['alg', 'gen', 'met', 'npt', clname, clnameo]].groupby(['alg', 'gen', 'met', 'npt']).mean()
    a.to_csv(WHERE + 'a.csv')
    a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")
    os.remove(WHERE + "a.csv")
    return a

def change_names(a):
    a = a.replace('dtval', 'DTcv')
    a = a.replace('dtc', 'DTcomp')
    a = a.replace('dt', 'DT')
    a = a.replace('cmmrf', 'cmm')
    a = a.replace('kdebw', 'kde')
    a = a.replace('kdebwm', 'kdem')
    a = a.replace('randn', 'norm')
    a = a.replace('randu', 'unif')
    a = a.replace('rerx', 're-rx')
    a = a.replace('rf', 'RF')
    a = a.replace('xgb', 'BT')
    a = a.replace('bicv', 'BI')
    a = a.replace('primcv', 'PRIM')
    a = a.replace('irep', 'IREP')
    a = a.replace('ripper', 'RIPPER')
    return a
    
def separate_baseline(a, clname, clnameo):
    tmp1 = a.drop(columns = [clname])
    tmp1['gen'] = 'NO'
    tmp1 = tmp1.rename(columns={clnameo: clname})
    tmp1 = tmp1.groupby(['alg', 'gen', 'met', 'npt']).max() # the choice of aggregate function (mean, min, median) should have no effect
    tmp1.to_csv(WHERE + 'tmp1.csv')
    tmp1 = pd.read_csv(WHERE + 'tmp1.csv', delimiter = ",")
    os.remove(WHERE + "tmp1.csv")
    return tmp1

#### Heatmap drawing:

def draw_heatmap(clname, clnameo, mlt = 100, pal = 'normal', npts = 100, ylbl = True, mod = 'dt'):
   
    def draw_heatmap_c(*args, **kwargs):
        data = kwargs.pop('data')
        center = data[data['gen'] == 'NO'][args[2]].iloc[0]
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        sns.heatmap(d, center = center, **kwargs)
    
    a = res_aggregate(mod, npts, clname, clnameo)
    a = pd.concat([separate_baseline(a, clname, clnameo), a.drop(columns = [clnameo])])
    a = change_names(a)    
    
    a[clname] = np.round(a[clname]*mlt, 1)
    if clname == 'nle':
        a[clname][a['alg'] == 'DT'] = np.round(a[clname][a['alg'] == 'DT'], 0)
    
    if pal == 'inverse':
        rdgn = sns.diverging_palette(h_neg = 130, h_pos = 10, s = 99, l = 55, sep = 3, as_cmap = True)
    else:
        rdgn = sns.diverging_palette(h_neg = 10, h_pos = 130, s = 99, l = 55, sep = 3, as_cmap = True)
    
    asp = 0.4/1.2
    if ylbl == False:
        asp = 0.33/1.2
    fg = sns.FacetGrid(a, row = 'npt', col = 'alg', margin_titles=False, despine=False, height=4.2, aspect=asp)
    fg.map_dataframe(draw_heatmap_c, 'met', 'gen', clname, cbar = False, cmap = rdgn, annot = True, fmt='g')
    if ylbl == False:
        fg.set(yticklabels=[])
    
    fg.set_axis_labels("", "")
    fg.set_titles(col_template="{col_name}", row_template="{row_name}")
    fg.tight_layout()
    fg.savefig("results/" + mod + clname + str(npts) + ".pdf")
  
    
draw_heatmap('tes', 'ora', npts = 100)
draw_heatmap('tes', 'ora', npts = 400)
draw_heatmap('fid', 'orf', npts = 100, ylbl = False)
draw_heatmap('fid', 'orf', npts = 400, ylbl = False)
draw_heatmap('nle', 'orn', npts = 100, mlt = 1, pal = 'inverse', ylbl = False)
draw_heatmap('nle', 'orn', npts = 400, mlt = 1, pal = 'inverse', ylbl = False)

draw_heatmap('tes', 'ora', npts = 100, mod = 'rules')
draw_heatmap('tes', 'ora', npts = 400, mod = 'rules')
draw_heatmap('fid', 'orf', npts = 100, ylbl = False, mod = 'rules')
draw_heatmap('fid', 'orf', npts = 400, ylbl = False, mod = 'rules')
draw_heatmap('nle', 'orn', npts = 100, mlt = 1, pal = 'inverse', ylbl = False, mod = 'rules')
draw_heatmap('nle', 'orn', npts = 400, mlt = 1, pal = 'inverse', ylbl = False, mod = 'rules')

draw_heatmap('tes', 'ora', npts = 100, mod = 'sd')
draw_heatmap('tes', 'ora', npts = 400, mod = 'sd')
draw_heatmap('nle', 'orn', npts = 100, mlt = 1, pal = 'inverse', ylbl = False, mod = 'sd')
draw_heatmap('nle', 'orn', npts = 400, mlt = 1, pal = 'inverse', ylbl = False, mod = 'sd')


#### BB-model accuracy increase over the naive model

def bb_qual_change(meta):
    precdefault = meta[meta['alg'].isin(['testprec'])]['val'].iloc[0]
    data = []
    for i in range(0,len(meta)):
        if 'acc' in meta['alg'].iloc[i]:
            meta['val'].iloc[i] = meta['val'].iloc[i] - precdefault
            meta['alg'].iloc[i] = meta['alg'].iloc[i][:-3]
            data.append(pd.DataFrame([meta.iloc[i]]))
            
    return pd.concat(data)

def bb_get_result(fname):
    meta = pd.read_csv(WHERE + fname.split(".")[0] + '.csv',\
                            delimiter = ",", header = None)
    meta.columns = ['alg', 'val']
    tmp = bb_qual_change(meta)
            
    extra = fname.split(".")[0].split("_")
    tmp['dat'] = [extra[0]]*tmp.shape[0]
    tmp['itr'] = [extra[1]]*tmp.shape[0]
    tmp['npt'] = [extra[2]]*tmp.shape[0]
    return tmp

res_bb = []
k = 0
for i in filenames:
    k = k + 1
    sys.stdout.write('\r' + "Loading" + "." + str(k))
    if 'meta' in i:
        res_bb.append(bb_get_result(i))

res_bb = pd.concat(res_bb) 
res_bb['npt'] = pd.to_numeric(res_bb['npt'])    
res_bb = res_bb[['alg', 'npt', 'val']].groupby(['alg', 'npt']).mean()
res_bb.to_csv('results/res_bb.csv')
res_bb = pd.read_csv('results/res_bb.csv', delimiter = ',')
os.remove('results/res_bb.csv')
res_bb = change_names(res_bb)
res_bb.columns = ['BB', 'N', 'BBacc']

#### Win-draw-loss tables

def get_table(a, mod = 'dt'):
    if mod == 'dt':
        nms = ('dt','dtc', 'dtval')
    elif mod == 'rules':
        nms = ('ripper','irep')
    elif mod == 'sd':
        nms = ('primcv','bicv')
    else:
        raise ValueError('{mod} is a wrong mod value'.format(dname = repr(dname)))
    a = a[a['alg'].isin(nms)]  
    
    a = a.fillna(0)
    a['wdl'] = a['1.0'].astype(int).astype(str) + "/" + a['0.0'].astype(int).astype(str)\
        + "/" + a['-1.0'].astype(int).astype(str)
    a = a[['alg', 'met', 'npt', 'wdl']]
    a = change_names(a)
    a = a.pivot(index = ['met','npt'], columns=['alg'], values=['wdl'])
    a.to_csv(WHERE + 'a.csv')
    a = pd.read_csv(WHERE + 'a.csv', delimiter = ',')
    os.remove(WHERE + "a.csv")
    a.iloc[0,1] = 'N'
    a.iloc[0,0] = 'BB'
    a.columns = a.iloc[0]
    a = a.drop([0,1]) 
    a['N'] = pd.to_numeric(a['N'])
    return a

a = res.copy()
a['dif'] = np.sign(a['tes'] - a['ora'])
a = a[['alg', 'gen', 'met', 'npt', 'dif']]
a = a.groupby(['alg', 'gen', 'met', 'npt']).dif.value_counts().unstack()
a.to_csv(WHERE + 'a.csv')
a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")
os.remove(WHERE + "a.csv")
a = a[a['gen'] == 'kdebw']
   
pd.merge(get_table(a, 'dt'), res_bb).to_csv('results/dt_pivot.csv', index=False)
get_table(a, 'rules').to_csv('results/rules_pivot.csv', index=False)
get_table(a, 'sd').to_csv('results/sd_pivot.csv', index=False)

