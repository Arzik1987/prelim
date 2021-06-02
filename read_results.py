import os
import pandas as pd
import seaborn as sns
# import matplotlib.pyplot as plt

WHERE = 'registry/'
if os.path.exists(WHERE + "a.csv"):
    os.remove(WHERE + "a.csv")
else:
    print("The file does not exist")
    
_, _, filenames = next(os.walk(WHERE))


def get_ratios(d):
    algnames = d['alg'].unique()
    for i in algnames:
        newvals = (1 - d[d['alg'].isin([i]) & d['met'].isin(['na'])]['tes'].iloc[0])/(1 - d[d['alg'].isin([i])]['tes'])
        d.loc[d['alg'] == i,'tes'] = newvals
    return d

def get_result(fname):
    tmp_times = pd.read_csv(WHERE + fname.split(".")[0] + "_times" + '.csv', delimiter = ",", header = None)
    tmp_times.columns = ['alg', 'val']
    if (1 - tmp_times['val'][tmp_times['alg'] == 'trainprec'].iloc[0])/(1 - tmp_times['val'][tmp_times['alg'] == 'dtcvsc'].iloc[0]) > 1.1:
        tmp = pd.read_csv(WHERE + fname, delimiter = ",", header = None)
        tmp.columns = ['alg', 'gen', 'met', 'tra', 'new', 'tes', 'tme']
        tmp = get_ratios(tmp)
        extra = fname.split(".")[0].split("_")
        tmp['dat'] = [extra[0]]*tmp.shape[0]
        tmp['itr'] = [extra[1]]*tmp.shape[0]
        tmp['npt'] = [extra[2]]*tmp.shape[0]
        return tmp

def get_result_prim(fname):
    tmp = pd.read_csv(WHERE + fname, delimiter = ",", header = None)
    # (1) model (2) gen (3) met (4) sctr (5) scnew (6) sctest (7) time (8-10) prec (11-13) rec
    tmp.columns = ['alg', 'gen', 'met', 'tra', 'new', 'tes', 'tme', 'ptra', 'pnew', 'ptes', 'rtra', 'rnew', 'rtes']
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
        # if 'prim' in i:
            # res_prim.append(get_result_prim(i))
    except:
        print("error at " + i)

res = pd.concat(res)
# res_prim = pd.concat(res_prim)




# debug
res['dat'].unique()
a = res[['alg', 'gen', 'met', 'npt', 'tes', 'dat']].groupby(['dat', 'npt', 'alg', 'gen', 'met']).count()



res1 = res.loc[res['dat'] == 'ml']
res1['how'] = res1['alg'] + res1['gen'] + res1['met']
res1 = res1[res1['how'].isin(['dtnana', 'dt_compnana', 'dtkdebwrf', 'dt_compkdebwrf', 'dtkdebwxgb', 'dt_compkdebwxgb'])][['how', 'tes', 'npt']]

sns.boxplot(y = 'tes', x = 'npt', 
                  data = res1, 
                  palette = "colorblind",
                  hue = 'how')


# a = res[['alg', 'gen', 'met', 'dat', 'npt', 'tra', 'new', 'tes']].groupby(['dat', 'alg', 'gen', 'met', 'npt']).std()
# a.to_csv(WHERE + 'a.csv')
# a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")
# b = a.loc[a['gen'].isin(['kdebw', 'na']) & a['met'].isin(['rf', 'xgb', 'na']) & a['alg'].isin(['bicv', 'bicvp']) & a['dat'].isin(['gt'])]


a = res[['alg', 'gen', 'met', 'npt', 'tes']].groupby(['alg', 'gen', 'met', 'npt']).mean()
# mean is more fair than median since dataset is not in grouping 
# (if > than half datasets have zero improvement and the others have positive one)
a.to_csv(WHERE + 'a.csv')
a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")

b = a.loc[a['alg'].isin(['dtcv']) & (a['gen'] != 'na')]

df = b.loc[b['npt'] == 800]
df = df.pivot('gen', 'met', 'tes')
rdgn = sns.diverging_palette(h_neg=130, h_pos=10, s=99, l=55, sep=3, as_cmap=True)
sns.heatmap(df - 1, cmap = rdgn, center = 0.0, annot = True, fmt ='.0%')
# sns.heatmap(df)
