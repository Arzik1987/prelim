import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

WHERE = 'registrydt/'
if os.path.exists(WHERE + "a.csv"):
    os.remove(WHERE + "a.csv")
else:
    print("The file does not exist")
    
_, _, filenames = next(os.walk(WHERE))


# Using differences is in line with CMM paper (they refer to differences)
def get_acc_increase(d):
    algnames = d['alg'].unique()
    for i in algnames:
        orig_acc = d[d['alg'].isin([i]) & d['met'].isin(['na'])]['tes'].iloc[0]
        new_acc = d[d['alg'].isin([i])]['tes']
        newvals = new_acc - orig_acc
        d.loc[d['alg'] == i,'tes'] = newvals
    return d

def get_result(fname):
    tmp_times = pd.read_csv(WHERE + fname.split(".")[0] + "_times" + '.csv',\
                            delimiter = ",", header = None)
    tmp_times.columns = ['alg', 'val']
    tr_acc = tmp_times['val'][tmp_times['alg'] == 'trainprec'].iloc[0]
    cv_acc = tmp_times['val'][tmp_times['alg'] == 'dtcvsc'].iloc[0]
    rf_acc = tmp_times['val'][tmp_times['alg'] == 'rfacc'].iloc[0]
    xgb_acc = tmp_times['val'][tmp_times['alg'] == 'xgbacc'].iloc[0]

    tmp = pd.read_csv(WHERE + fname, delimiter = ",", header = None)
    tmp.columns = ['alg', 'gen', 'met', 'tra', 'new', 'tes', 'tme']
    if cv_acc/rf_acc > 0.95 and cv_acc/xgb_acc > 0.95:
        tmp = tmp[tmp['alg'] != 'dtcv']
        tmp = tmp[tmp['alg'] != 'dtcomp']
    # careful with the later aggregation. Old methods should eventually become 
    # 0 or 1 score before their aggregation 
    elif cv_acc/rf_acc > 0.95:
        tmp = tmp[(tmp['alg'] != 'dtcv') | (tmp['met'] != 'rf')]
        tmp = tmp[(tmp['alg'] != 'dtcomp') | (tmp['met'] != 'rf')]
    elif cv_acc - xgb_acc > 0.95:
        tmp = tmp[(tmp['alg'] != 'dtcv') | (tmp['met'] != 'xgb')]
        tmp = tmp[(tmp['alg'] != 'dtcomp') | (tmp['met'] != 'xgb')]
        
    tmp = get_acc_increase(tmp)
    extra = fname.split(".")[0].split("_")
    tmp['dat'] = [extra[0]]*tmp.shape[0]
    tmp['itr'] = [extra[1]]*tmp.shape[0]
    tmp['npt'] = [extra[2]]*tmp.shape[0]
    return tmp

def get_result_prim(fname):
    tmp = pd.read_csv(WHERE + fname, delimiter = ",", header = None)
    # (1) model (2) gen (3) met (4) sctr (5) scnew (6) sctest (7) time (8-10) prec (11-13) rec
    tmp.columns = ['alg', 'gen', 'met', 'tra', 'new', 'tes', 'tme',\
                   'ptra', 'pnew', 'ptes', 'rtra', 'rnew', 'rtes']
    extra = fname.split(".")[0].split("_")
    tmp['dat'] = [extra[0]]*tmp.shape[0]
    tmp['itr'] = [extra[1]]*tmp.shape[0]
    tmp['npt'] = [extra[2]]*tmp.shape[0]
    return tmp


res = []
res_prim = []
for i in filenames:
    try:
        if not "times" in i and not "prim" in i:
            res.append(get_result(i))
        if 'prim' in i:
            res_prim.append(get_result_prim(i))
    except:
        print("error at " + i)

res = pd.concat(res)
# res_prim = pd.concat(res_prim)




a = res[['alg', 'gen', 'met', 'npt', 'tes']].groupby(['alg', 'gen', 'met', 'npt']).mean()
# mean is more fair than median since dataset is not in grouping 
# (if > than half datasets have zero improvement and the others have positive one)

a = res[['alg', 'gen', 'met', 'npt', 'tes']]
a.tes = np.sign(a.tes)
a = a.groupby(['alg', 'gen', 'met', 'npt']).tes.value_counts().unstack()

a.to_csv(WHERE + 'a.csv')
a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")

df = a.loc[(a['npt'] == 100) & a['alg'].isin(['dtcomp'])]
df = df.pivot('gen', 'met', 'tes')
rdgn = sns.diverging_palette(h_neg = 130, h_pos = 10, s = 99, l = 55, sep = 3, as_cmap = True)
sns.heatmap(df, cmap = rdgn, center = 0.0, annot = True, fmt ='.0%')



for i in res['dat'].unique(): 
    res1 = res.loc[res['dat'] == i]
    res1['how'] = res1['alg'] + res1['gen'] + res1['met']
    res1 = res1[res1['how'].isin(['dtcvnana', 'dtcvkdebwrf'])][['how', 'tes', 'npt']]
    sns.boxplot(y = 'tes', x = 'npt', 
                      data = res1, 
                      palette = "colorblind",
                      hue = 'how').set_title(i)
    plt.show()



# PRIM

from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt

for j in res['dat'].unique():
    a = res_prim[(res_prim['dat'] == j) & (res_prim['npt'] == '100')]
    a['how'] = a['alg'] + a['gen'] + a['met']
    pconv = []
    preds = []
    rconv = []
    rreds = []
    a['itr'] = a['itr'].astype(int)
    
    for i in a['itr'].unique():
        tmp = a[a['how'] == 'primcvnana']
        tmp1 = list(map(float, tmp[tmp['itr'] == i].iloc[0]['ptra'].split("_")))
        tmp2 = list(map(float, tmp[tmp['itr'] == i].iloc[0]['ptes'].split("_")))
        n = min(len(tmp1), len(tmp2))
        pconv.append(np.column_stack((tmp1[0:n], tmp2[0:n], np.repeat(i, n))))
        tmp = a[a['how'] == 'primcvkdebwrf']
        tmp1 = list(map(float, tmp[tmp['itr'] == i].iloc[0]['ptra'].split("_")))
        tmp2 = list(map(float, tmp[tmp['itr'] == i].iloc[0]['ptes'].split("_")))
        tmp3 = list(map(float, tmp[tmp['itr'] == i].iloc[0]['pnew'].split("_")))
        n = min(len(tmp1), len(tmp2), len(tmp3))
        preds.append(np.column_stack((tmp1[0:n], tmp2[0:n], tmp3[0:n], np.repeat(i, n))))
        
        tmp = a[a['how'] == 'primcvnana']
        tmp1 = list(map(float, tmp[tmp['itr'] == i].iloc[0]['rtra'].split("_")))
        tmp2 = list(map(float, tmp[tmp['itr'] == i].iloc[0]['rtes'].split("_")))
        n = min(len(tmp1), len(tmp2))
        rconv.append(np.column_stack((tmp1[0:n], tmp2[0:n], np.repeat(i, n))))
        tmp = a[a['how'] == 'primcvkdebwrf']
        tmp1 = list(map(float, tmp[tmp['itr'] == i].iloc[0]['rtra'].split("_")))
        tmp2 = list(map(float, tmp[tmp['itr'] == i].iloc[0]['rtes'].split("_")))
        tmp3 = list(map(float, tmp[tmp['itr'] == i].iloc[0]['rnew'].split("_")))
        n = min(len(tmp1), len(tmp2), len(tmp3))
        rreds.append(np.column_stack((tmp1[0:n], tmp2[0:n], tmp3[0:n], np.repeat(i, n))))
    
    pconv = np.concatenate(pconv, axis = 0)
    preds = np.concatenate(preds, axis = 0)
    rconv = np.concatenate(rconv, axis = 0)
    rreds = np.concatenate(rreds, axis = 0)

    x, y = lowess(rconv[:,1], rconv[:,0])[:,0], lowess(rconv[:,1], rconv[:,0])[:,1]
    plt.plot(x, y, 'green', linewidth = 2)
    x, y = lowess(rreds[:,1], rreds[:,0])[:,0], lowess(rreds[:,1], rreds[:,0])[:,1]
    plt.plot(x, y, 'red', linewidth = 2)
    x, y = lowess(rreds[:,2], rreds[:,0])[:,0], lowess(rreds[:,2], rreds[:,0])[:,1]
    plt.plot(x, y, 'orange', linewidth = 2)
    plt.plot(np.array([0.3,1]), np.array([0.3,1]), linewidth = 1)
    plt.show()





# debug
res['dat'].unique()
a = res[['alg', 'gen', 'met', 'npt', 'tes', 'dat']].groupby(['dat', 'npt', 'alg', 'gen', 'met']).count()


res[(res['tes'] < -1) & (res['met'] == 'rf') & (res['gen'] == 'kdebw')]



# a = res[['alg', 'gen', 'met', 'dat', 'npt', 'tra', 'new', 'tes']].groupby(['dat', 'alg', 'gen', 'met', 'npt']).std()
# a.to_csv(WHERE + 'a.csv')
# a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")
# b = a.loc[a['gen'].isin(['kdebw', 'na']) & a['met'].isin(['rf', 'xgb', 'na']) & a['alg'].isin(['bicv', 'bicvp']) & a['dat'].isin(['gt'])]

