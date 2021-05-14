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

def get_result(fname):
    if not "times" in fname:
        tmp = pd.read_csv(WHERE + fname, delimiter = ",", header = None)
        tmp.columns = ['alg', 'gen', 'met', 'tra', 'new', 'tes', 'tme']
        extra = fname.replace("credit_cards", "cc").replace("eeg_eye_state", "ees").replace("gamma_telescope", "gt").split(".")[0].split("_")
        tmp['dat'] = [extra[0]]*tmp.shape[0]
        tmp['itr'] = [extra[1]]*tmp.shape[0]
        tmp['npt'] = [extra[2]]*tmp.shape[0]
        return tmp

res = []
for i in filenames:
    try:
        res.append(get_result(i))
    except:
        print("error at " + i)

res = pd.concat(res)



# res1 = res.loc[res['dat'] == 'stocks']
# res1['how'] = res1['alg'] + res1['gen'] + res1['met']
# res1 = res1[res1['how'].isin(['dtnana', 'dt_compnana', 'dtkdebwrf', 'dt_compkdebwrf', 'dtkdebwxgb', 'dt_compkdebwxgb'])][['how', 'tes', 'npt']]

# sns.boxplot(y = 'tes', x = 'npt', 
#                  data = res1, 
#                  palette = "colorblind",
#                  hue = 'how')


# a = res[['alg', 'gen', 'met', 'dat', 'npt', 'tra', 'new', 'tes']].groupby(['dat', 'alg', 'gen', 'met', 'npt']).std()
# a.to_csv(WHERE + 'a.csv')
# a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")
# b = a.loc[a['gen'].isin(['kdebw', 'na']) & a['met'].isin(['rf', 'xgb', 'na']) & a['alg'].isin(['bicv', 'bicvp']) & a['dat'].isin(['gt'])]


a = res[['alg', 'gen', 'met', 'npt', 'tes']].groupby(['alg', 'gen', 'met', 'npt']).median()
a.to_csv(WHERE + 'a.csv')
a = pd.read_csv(WHERE + 'a.csv', delimiter = ",")
b = a.loc[a['alg'].isin(['dtcv'])]

df = b.loc[b['npt'] == 400]
df = df.pivot('gen', 'met', 'tes')
sns.heatmap(df, cmap = "Blues")
