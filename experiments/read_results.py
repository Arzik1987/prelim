import os
import pandas as pd
import seaborn as sns
import numpy as np
import sys

FILEPATH = os.path.dirname(os.path.abspath(__file__))
WHERE = FILEPATH + '/registry/' 
if os.path.exists(WHERE + 'res.csv'):
    os.remove(WHERE + 'res.csv')

_, _, filenames = next(os.walk(WHERE))


#### Helper functions for postprocessing

def qual_change(d, meta):
    # calculates accuracy and fidelity increase with respect to naive prediction
    # for a single experiment
    
    # make 'baseline' rows for 'p' models
    dtmp = d[d['alg'].isin(['dt', 'dtc', 'dtval']) & (d['gen'] =='na')].copy()
    dtmp['alg'] = dtmp['alg'] + 'p'
    d = d.append(dtmp)
    
    # make sure numeric columns are numeric
    d['fid'] = pd.to_numeric(d['fid'], errors = 'coerce')
    d['bac'] = pd.to_numeric(d['bac'], errors = 'coerce')
    
    # (balanced) accuracy increase compared to the baseline  
    precdefault = meta[meta['alg'].isin(['testprec'])]['val'].iloc[0]
    for i in range(0, d.shape[0]):
        if d.iloc[i][0] not in ['primcv', 'bicv']:
            d['tes'].iloc[i] = d['tes'].iloc[i] - precdefault
            d['bac'].iloc[i] = d['bac'].iloc[i] - 0.5 # balanced accuracy of the naive model is 0.5
            
    # add baseline values as separate columns
    algnames = d['alg'].unique()
    for i in algnames:
        orig_acc = d[(d['alg'] == i) & (d['gen'] =='na')]['tes']
        orig_nle = d[(d['alg'] == i) & (d['gen'] =='na')]['nle']
        orig_bac = d[(d['alg'] == i) & (d['gen'] =='na')]['bac']
        d = d[~((d['alg'] == i) & (d['gen'] =='na'))]
        d.loc[d['alg'] == i,'ora'] = [orig_acc]*d.loc[d['alg'] == i].shape[0]
        d.loc[d['alg'] == i,'orn'] = [orig_nle]*d.loc[d['alg'] == i].shape[0]
        d.loc[d['alg'] == i,'orb'] = [orig_bac]*d.loc[d['alg'] == i].shape[0]
        
    # fidelity increase compared to the baseline
    algmet = d[['alg', 'met']].drop_duplicates()
    for i, j in zip(algmet['alg'], algmet['met']): 
        if i not in ['primcv', 'bicv']:
            def_fid = meta[meta['alg'] == j + 'fid']['val'].iloc[0]
            if i in ['dtp', 'dtcp', 'dtvalp']:
                orig_fid = meta[meta['alg'] == j + i[0:-1] + 'fid']['val'].iloc[0] - def_fid
            else:
                orig_fid = meta[meta['alg'] == j + i + 'fid']['val'].iloc[0] - def_fid
            d.loc[(d['alg'] == i) & (d['met'] == j),'orf'] = [orig_fid]*d.loc[(d['alg'] == i) & (d['met'] == j)].shape[0]
            d.loc[(d['alg'] == i) & (d['met'] == j),'fid'] = d.loc[(d['alg'] == i) & (d['met'] == j),'fid'] - def_fid
        else:
            d.loc[(d['alg'] == i) & (d['met'] == j),'orf'] = [np.nan]*d.loc[(d['alg'] == i) & (d['met'] == j)].shape[0]

    return d.copy()

def get_result(fname):
    # processes the results from a single experiment
    meta = pd.read_csv(WHERE + fname.split('.')[0] + '_meta' + '.csv',\
                            delimiter = ',', header = None)
    meta.columns = ['alg', 'val']
    
    d = pd.read_csv(WHERE + fname, delimiter = ',', header = None)
    d.columns = ['alg', 'gen', 'met', 'tra', 'tes', 'nle', 'tme', 'fid', 'bac']
    d = qual_change(d, meta)
            
    extra = fname.split('.')[0].split('_')
    d['dat'] = [extra[0]]*d.shape[0]
    d['itr'] = [extra[1]]*d.shape[0]
    d['npt'] = [extra[2]]*d.shape[0]
    return d

#### Postprocessing: read and process data from all experiments

res = []
k = 0
nf = str(len(filenames))
for i in filenames:
    k = k + 1
    sys.stdout.write('\r' + 'loading' + '.' + str(k) + '/' + nf)
    if not 'meta' in i and not 'zeros' in i:
            res.append(get_result(i))

res = pd.concat(res)
res.loc[res['gen'] == 'adasyns','gen'] = 'adasyn'
for names in ['tra','tes','ora','bac','orb','nle','orn','tme','fid','orf','itr','npt']:
    res[names] = pd.to_numeric(res[names])
res.to_csv(WHERE + 'res.csv')
# res = pd.read_csv(WHERE + 'res.csv', delimiter = ',')


#### Helper functions for drawing

def res_aggregate(mod, npts, clname, clnameo):
    a = res.copy()   
    a = a[a['npt'].isin([npts])]
    if mod == 'dt':
        nms = ('dt','dtc', 'dtval')
    elif mod == 'dtp':
        nms = ('dtp','dtcp', 'dtvalp')
    elif mod == 'dtb':
        nms = ('dtb','dtcb', 'dtvalb')
    elif mod == 'rules':
        nms = ('ripper','irep')
    elif mod == 'sd':
        nms = ('primcv','bicv')
    else:
        raise ValueError('{mod} is a wrong mod value'.format(mod = repr(mod)))
    a = a[a['alg'].isin(nms)]     
    a = a[['alg', 'gen', 'met', 'npt', clname, clnameo]].groupby(['alg', 'gen', 'met', 'npt']).mean()
    a.to_csv(WHERE + 'a.csv')
    a = pd.read_csv(WHERE + 'a.csv', delimiter = ',')
    os.remove(WHERE + 'a.csv')
    return a

def change_names(a):
    a = a.replace('dtval', 'DTcv')
    a = a.replace('dtc', 'DTcomp')
    a = a.replace('dt', 'DT')
    a = a.replace('dtvalp', 'DTcv')
    a = a.replace('dtcp', 'DTcomp')
    a = a.replace('dtp', 'DT')
    a = a.replace('dtvalb', 'DTcv')
    a = a.replace('dtcb', 'DTcomp')
    a = a.replace('dtb', 'DT')
    a = a.replace('adasyn', 'ADASYN')
    a = a.replace('cmmrf', 'CMM')
    a = a.replace('dummy', 'DUMMY')
    a = a.replace('gmm', 'GMM')
    a = a.replace('gmmal', 'GMMAL')
    a = a.replace('kdebw', 'KDE')
    a = a.replace('kdeb', 'KDEB')
    a = a.replace('kdebwm', 'KDEM')
    a = a.replace('munge', 'MUNGE')
    a = a.replace('randn', 'NORM')
    a = a.replace('randu', 'UNIF')
    a = a.replace('rerx', 'RE-RX')
    a = a.replace('smote', 'SMOTE')
    a = a.replace('ssl', 'SSL')
    a = a.replace('vva', 'VVA')
    a = a.replace('rf', 'RF')
    a = a.replace('xgb', 'BT')
    a = a.replace('rfb', 'RF')
    a = a.replace('xgbb', 'BT')
    a = a.replace('bicv', 'BI')
    a = a.replace('primcv', 'PRIM')
    a = a.replace('irep', 'IREP')
    a = a.replace('ripper', 'RIPPER')
    return a
    
def separate_baseline(a, clname, clnameo):
    tmp1 = a.drop(columns = [clname])
    tmp1['gen'] = ' NO'
    tmp1 = tmp1.rename(columns={clnameo: clname})
    tmp1 = tmp1.groupby(['alg', 'gen', 'met', 'npt']).max() # the choice of aggregate function (mean, min, median) should have no effect
    tmp1.to_csv(WHERE + 'tmp1.csv')
    tmp1 = pd.read_csv(WHERE + 'tmp1.csv', delimiter = ',')
    os.remove(WHERE + 'tmp1.csv')
    return tmp1

def my_diverging_palette(r_neg, r_pos, g_neg, g_pos, b_neg, b_pos, sep=1, n=6,  # noqa
                      center='light', as_cmap=False):

    palfunc = dict(dark=sns.dark_palette, light=sns.light_palette)[center]
    n_half = int(128 - (sep // 2))
    neg = palfunc((r_neg/255, g_neg/255, b_neg/255), n_half, reverse=True, input='rgb')
    pos = palfunc((r_pos/255, g_pos/255, b_pos/255), n_half, input='rgb')
    midpoint = dict(light=[(.95, .95, .95)], dark=[(.133, .133, .133)])[center]
    mid = midpoint * sep
    pal = sns.blend_palette(np.concatenate([neg, mid, pos]), n, as_cmap=as_cmap)
    return pal

#### Heatmap drawing:
def draw_heatmap(clname, clnameo, mlt = 100, pal = 'normal', npts = 100, ylbl = True, mod = 'dt', ytext = '', fsz = 13):
   
    def draw_heatmap_c(*args, **kwargs):
        data = kwargs.pop('data')
        center = data[data['gen'] == ' NO'][args[2]].iloc[0]
        d = data.pivot(index=args[1], columns=args[0], values=args[2])
        sns.heatmap(d, center = center, **kwargs)
    
    a = res_aggregate(mod, npts, clname, clnameo)
    a = pd.concat([separate_baseline(a, clname, clnameo), a.drop(columns = [clnameo])])
    a = change_names(a)    
    
    a[clname] = np.round(a[clname]*mlt, 1)
    if clname == 'nle':
        a[clname][a['alg'] == 'DT'] = np.round(a[clname][a['alg'] == 'DT'], 0)
    
    if pal == 'inverse':
        dvgp = my_diverging_palette(r_neg = 0, r_pos = 255, g_neg = 91, g_pos = 213, b_neg = 183, b_pos = 0, sep = 3, as_cmap = True)
    else:
        dvgp = my_diverging_palette(r_neg = 255, r_pos = 0, g_neg = 213, g_pos = 91, b_neg = 0, b_pos = 183, sep = 3, as_cmap = True)
    
    asp = 0.42/1.2
    if ylbl == False:
        asp = 0.33/1.2
        
    fg = sns.FacetGrid(a, row = 'npt', col = 'alg', margin_titles=False, despine=False, height=4.2, aspect=asp)
    fg.map_dataframe(draw_heatmap_c, 'met', 'gen', clname, cbar = False, cmap = dvgp, annot = True, fmt='g')
    if ylbl == False:
        fg.set(yticklabels=[])
    
    fg.set_axis_labels('', ytext, fontsize = fsz)
    fg.set_titles(col_template='{col_name}', row_template='{row_name}')
    fg.tight_layout()
    fg.savefig(FILEPATH + '/results/' + mod + '_' + clname + str(npts) + '.pdf')

if not os.path.exists(FILEPATH + '/results/'):
    os.mkdir(FILEPATH + '/results/')  
  
draw_heatmap('tes', 'ora', npts = 100, fsz = 13)
draw_heatmap('tes', 'ora', npts = 400, ylbl = False)
draw_heatmap('fid', 'orf', npts = 100)
draw_heatmap('fid', 'orf', npts = 400, ylbl = False)
draw_heatmap('nle', 'orn', npts = 100, mlt = 1, pal = 'inverse')
draw_heatmap('nle', 'orn', npts = 400, mlt = 1, pal = 'inverse', ylbl = False)

draw_heatmap('tes', 'ora', npts = 100, mod = 'dtp')
draw_heatmap('tes', 'ora', npts = 400, ylbl = False, mod = 'dtp')

draw_heatmap('bac', 'orb', npts = 100, mod = 'dtb')
draw_heatmap('bac', 'orb', npts = 400, ylbl = False, mod = 'dtb')

draw_heatmap('tes', 'ora', npts = 100, mod = 'rules')
draw_heatmap('tes', 'ora', npts = 400, ylbl = False, mod = 'rules')
draw_heatmap('fid', 'orf', npts = 100, mod = 'rules')
draw_heatmap('fid', 'orf', npts = 400, ylbl = False, mod = 'rules')
draw_heatmap('nle', 'orn', npts = 100, mlt = 1, pal = 'inverse', mod = 'rules')
draw_heatmap('nle', 'orn', npts = 400, mlt = 1, pal = 'inverse', ylbl = False, mod = 'rules')

draw_heatmap('tes', 'ora', npts = 100, mod = 'sd')
draw_heatmap('tes', 'ora', npts = 400, ylbl = False, mod = 'sd')
draw_heatmap('nle', 'orn', npts = 100, mlt = 1, pal = 'inverse', mod = 'sd')
draw_heatmap('nle', 'orn', npts = 400, mlt = 1, pal = 'inverse', ylbl = False, mod = 'sd')


#### PIVOT: BB-model accuracy increase over the naive model
def bb_qual_change(meta):
    precdefault = meta[meta['alg'].isin(['testprec'])]['val'].iloc[0]
    data = []
    for i in range(0,len(meta)):
        if 'acc' in meta['alg'].iloc[i] and not 'acccv' in meta['alg'].iloc[i]\
            and not 'rfb' in meta['alg'].iloc[i] and not 'xgbb' in meta['alg'].iloc[i]:
            tmp = meta.copy()
            tmp['val'].iloc[i] = tmp['val'].iloc[i] - precdefault
            tmp['alg'].iloc[i] = tmp['alg'].iloc[i][:-3]
            data.append(pd.DataFrame([tmp.iloc[i]]))
            
    return pd.concat(data)

def bb_get_result(fname):
    meta = pd.read_csv(WHERE + fname.split('.')[0] + '.csv',\
                            delimiter = ',', header = None)
    meta.columns = ['alg', 'val']
    tmp = bb_qual_change(meta)
            
    extra = fname.split('.')[0].split('_')
    tmp['dat'] = [extra[0]]*tmp.shape[0]
    tmp['itr'] = [extra[1]]*tmp.shape[0]
    tmp['npt'] = [extra[2]]*tmp.shape[0]
    return tmp

res_bb = []
k = 0
for i in filenames:
    k = k + 1
    sys.stdout.write('\r' + 'Loading' + '.' + str(k))
    if 'meta' in i:
        res_bb.append(bb_get_result(i))

res_bb = pd.concat(res_bb) 
res_bb['npt'] = pd.to_numeric(res_bb['npt'])    
res_bb = res_bb[['alg', 'npt', 'val']].groupby(['alg', 'npt']).mean()
res_bb.to_csv(FILEPATH + '/results/res_bb.csv')
res_bb = pd.read_csv(FILEPATH + '/results/res_bb.csv', delimiter = ',')
os.remove(FILEPATH + '/results/res_bb.csv')
res_bb = change_names(res_bb)
res_bb.columns = ['BB', 'N', 'BBacc']

#### PIVOT: Win-draw-loss tables
def get_table(a, mod = 'dt'):
    if mod == 'dt':
        nms = ('dt','dtc', 'dtval')
    elif mod == 'rules':
        nms = ('ripper','irep')
    elif mod == 'sd':
        nms = ('primcv','bicv')
    else:
        raise ValueError('{mod} is a wrong mod value'.format(mod = repr(mod)))
    a = a[a['alg'].isin(nms)]  
    
    a = a.fillna(0)
    a['wdl'] = a['1.0'].astype(int).astype(str) + '/' + a['0.0'].astype(int).astype(str)\
        + '/' + a['-1.0'].astype(int).astype(str)
    a = a[['alg', 'met', 'npt', 'wdl']]
    a = change_names(a)
    a = a.pivot(index = ['met','npt'], columns=['alg'], values=['wdl'])
    a.to_csv(WHERE + 'a.csv')
    a = pd.read_csv(WHERE + 'a.csv', delimiter = ',')
    os.remove(WHERE + 'a.csv')
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
a = pd.read_csv(WHERE + 'a.csv', delimiter = ',')
os.remove(WHERE + 'a.csv')
a = a[a['gen'] == 'kdebw']
   
pd.merge(pd.merge(res_bb, get_table(a, 'dt')), pd.merge(get_table(a, 'rules'),\
    get_table(a, 'sd'))).to_csv(FILEPATH + '/results/pivot.csv', index = False)

