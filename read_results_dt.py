import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

WHERE = 'registrydt/'
if os.path.exists(WHERE + "a.csv"):
    os.remove(WHERE + "a.csv")
    
_, _, filenames = next(os.walk(WHERE))


terminated = []
for i in filenames:
    if not "times" in i:
        extra = i.split(".")[0].split("_")
        terminated.append([extra[1], extra[0], extra[2]])

terminated = pd.DataFrame(terminated, columns=['splitn','dname', 'dsize'])
stats = pd.pivot_table(terminated, values = "splitn", index = ['dname'],
                    columns=['dsize'], aggfunc = pd.Series.count)

rdgn = sns.diverging_palette(h_neg = 130, h_pos = 10, s = 99, l = 55, sep = 3, as_cmap = True)
sns.heatmap(stats, cmap = rdgn, center = 25, annot = True)



# Using differences is in line with CMM paper (they refer to differences)
def get_acc_increase(d, tmp_times):
    algnames = d['alg'].unique()
    for i in algnames:
        orig_acc = tmp_times[tmp_times['alg'].isin(['testprec'])]['val'].iloc[0]
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
    tmp.columns = ['alg', 'gen', 'met', 'tra', 'new', 'tes', 'nle','tme']
    if cv_acc/rf_acc > 0.95 and cv_acc/xgb_acc > 0.95:
        tmp = tmp[tmp['alg'] != 'dtcv']
        tmp = tmp[tmp['alg'] != 'dtcomp']
        tmp = tmp[tmp['alg'] != 'dtcomp2']
    # careful with the later aggregation. Old methods should eventually become 
    # 0 or 1 score before their aggregation 
    elif cv_acc/rf_acc > 0.95:
        tmp = tmp[(tmp['alg'] != 'dtcv') | (tmp['met'] != 'rf')]
        tmp = tmp[(tmp['alg'] != 'dtcomp') | (tmp['met'] != 'rf')]
        tmp = tmp[(tmp['alg'] != 'dtcomp2') | (tmp['met'] != 'rf')]
    elif cv_acc - xgb_acc > 0.95:
        tmp = tmp[(tmp['alg'] != 'dtcv') | (tmp['met'] != 'xgb')]
        tmp = tmp[(tmp['alg'] != 'dtcomp') | (tmp['met'] != 'xgb')]
        tmp = tmp[(tmp['alg'] != 'dtcomp2') | (tmp['met'] != 'rf')]
        
    tmp = get_acc_increase(tmp, tmp_times)
    extra = fname.split(".")[0].split("_")
    tmp['dat'] = [extra[0]]*tmp.shape[0]
    tmp['itr'] = [extra[1]]*tmp.shape[0]
    tmp['npt'] = [extra[2]]*tmp.shape[0]
    return tmp


res = []
for i in filenames:
    try:
        if not "times" in i and not "zeros" in i:
            res.append(get_result(i))
    except:
        print("error at " + i)

res = pd.concat(res)









a = res[['alg', 'gen', 'met', 'npt', 'tes']].groupby(['alg', 'gen', 'met', 'npt']).mean()
# mean is more fair than median since dataset is not in grouping 
# (if > than half datasets have zero improvement and the others have positive one)

a = res[['alg', 'gen', 'met', 'npt', 'tes']]
a.tes = np.sign(a.tes)
a = a.groupby(['alg', 'gen', 'met', 'npt']).tes.value_counts().unstack()


a.to_csv(WHERE + 'a.csv')
a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")

df = a.loc[(a['npt'] == 100) & a['alg'].isin(['dt'])]
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


