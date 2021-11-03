import numpy as np
import pandas as pd
import sys

def nunique(a, axis):
    return (np.diff(np.sort(a,axis=axis),axis=axis)!=0).sum(axis=axis) + 1

def load_data(dname):
    if dname == "occupancy": # 6 occupancy
        df = pd.concat([pd.read_csv('data/occupancy/datatest.txt', delimiter = ","), 
                pd.read_csv('data/occupancy/datatest2.txt', delimiter = ","),
                pd.read_csv('data/occupancy/datatraining.txt', delimiter = ",")])
        df['date'] = pd.to_datetime(df['date']).dt.hour
        y = (df['Occupancy'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('Occupancy')].to_numpy()
    elif dname == "higgs7": # 7 higgs
        df = pd.read_csv('data/higgs/phpZLgL9q.csv', delimiter = ",", nrows = 98049)
        y = (df['class'] == 1).astype(int).to_numpy()
        X  = df.filter(regex = 'm_', axis = 1).to_numpy()
    elif dname == "electricity": # 7 electricity
        df = pd.read_csv('data/electricity/electricity-normalized.csv', delimiter = ",")
        y = (df['class'] == 'UP').astype(int).to_numpy()
        X  = df[df.columns.drop('class')].to_numpy()
    elif dname == "htru": # 8 htru
        df = pd.read_csv('data/htru/HTRU_2.csv', delimiter = ",", header = None)
        y = (df[8] == 1).astype(int).to_numpy()
        X  = df.iloc[:, : 8].to_numpy()
    elif dname == "shuttle": # 9 shuttle
        df = pd.read_csv('data/shuttle/shuttle.tsv.gz', delimiter = "\t")
        y = (df['target'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('target')].to_numpy()
    elif dname == "avila": # 10 avila
        df = pd.concat([pd.read_csv('data/avila/avila-tr.txt', delimiter = ",", header = None), 
                        pd.read_csv('data/avila/avila-ts.txt', delimiter = ",", header = None)])
        y = (df[10] == 'A').astype(int).to_numpy()
        X  = df.iloc[:, : 10].to_numpy()
    elif dname == "gt": # 10 gamma_telescope
        df = pd.read_csv('data/gt/magic04.data', delimiter = ",", header = None)
        y = (df[10] == "g").astype(int).to_numpy()
        X  = df.iloc[:, : 10].to_numpy()
    elif dname == "cc": # 14 credit_cards
        df = pd.read_csv('data/cc/default_of_credit_card_clients.csv', delimiter = ",")
        y = (df.iloc[:, 24] == 1).astype(int).to_numpy()
        X  = df.iloc[:, 1:24].to_numpy()
    elif dname == "ees": # 14 eeg_eye_state
        df = pd.read_csv('data/ees/phplE7q6h.csv', delimiter = ",")
        y = (df['Class'] == 1).astype(int).to_numpy()
        X  = df.iloc[:, : 14].to_numpy()
    elif dname == "pendata": # 16 pendata
        df = pd.read_csv('data/pendata/dataset_32_pendigits.csv', delimiter = ",")
        y = (df['class'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('class')].to_numpy()
    elif dname == "ring": # 20 ring
        df = pd.read_csv('data/ring/ring.tsv.gz', delimiter = "\t")
        y = (df['target'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('target')].to_numpy()
    elif dname == "sylva": # 20 sylva
        df = pd.read_csv('data/sylva/sylva_prior.csv', delimiter = ",")
        y = (df['label'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('label')].to_numpy()
    elif dname == "higgs21": # 21 higgs
        df = pd.read_csv('data/higgs/phpZLgL9q.csv', delimiter = ",", nrows = 98049)
        y = (df['class'] == 1).astype(int).to_numpy()
        df = df[df.columns.drop('class')]
        X = df[df.columns.drop(list(df.filter(regex = 'm_', axis = 1)))].to_numpy()  
    elif dname == "jm1": # 21 jm1
        df = pd.read_csv('data/jm1/jm1.csv', delimiter = ",")
        y = (df['defects'] == True).astype(int).to_numpy()
        X  = df[df.columns.drop('defects')].to_numpy()
        X[X=='?'] = 'nan'
        X = X.astype(np.float64)
    elif dname == "saac2": # 21 saac2
        df = pd.read_csv('data/saac2/SAAC2.csv', delimiter = ",")
        y = (df['class'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('class')].to_numpy()
        X[X=='?'] = 'nan'
        X = X.astype(np.float64)
    elif dname == "stocks": # 21 stocks
        df = pd.read_csv('data/stocks/phpg2t68G.csv', delimiter = ",")
        y = (df['attribute_21'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('attribute_21')].to_numpy()
    elif dname == "sensorless": # 48 sensorless
        df = pd.read_csv('data/sensorless/Sensorless_drive_diagnosis.txt', delimiter = " ", header = None)
        y = (df[48] == 1).astype(int).to_numpy()
        X  = df.iloc[:, : 48].to_numpy()
    elif dname == "bankruptcy": # 64 bankruptcy
        df = pd.read_csv('data/bankruptcy/3year.csv', delimiter = ",", header = None)
        y = (df[64] == 1).astype(int).to_numpy()
        X  = df.iloc[:, : 64].to_numpy()
    elif dname == "nomao": # 69 nomao
        df = pd.read_csv('data/nomao/phpDYCOet.csv', delimiter = ",")
        y = (df['Class'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('Class')].to_numpy()
    elif dname == "gas": # 128 gas 
        df = pd.read_csv('data/gas/phpbL6t4U.csv', delimiter = ",")
        y = (df['Class'] == 1).astype(int).to_numpy()
        X  = df.iloc[:, : 128].to_numpy()
    elif dname == "clean2": # 166 clean2
        df = pd.read_csv('data/clean2/clean2.tsv.gz', delimiter = "\t")
        y = (df['target'] == 1).astype(int).to_numpy()
        X  = df.filter(regex = '^f', axis = 1).to_numpy()
    elif dname == "seizure": # 178 seizure
        df = pd.read_csv('data/seizure/data.csv', delimiter = ",")
        y = (df['y'] == 1).astype(int).to_numpy()
        X  = df.filter(regex = 'X', axis = 1).to_numpy()
    elif dname == "smartphone": # 561 smartphone 
        df = pd.read_csv('data/smartphone/php88ZB4Q.csv', delimiter = ",")
        y = (df['Class'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('Class')].to_numpy()
    # regression folder    
    elif dname == "ccpp": # ccpp 4
        df = pd.read_csv('data/ccpp/Folds5x2_pp.csv', delimiter = ",")
        y = (df['PE'] > 455).astype(int).to_numpy()
        X  = df[df.columns.drop('PE')].to_numpy()        
    elif dname == "seoul": # seoul_bike 7
        df = pd.read_csv('data/seoul/SeoulBikeData.csv', delimiter = ",", skiprows = 1, header = None)
        y = (df.iloc[:, 1] > 800).astype(int).to_numpy()
        X  = df.iloc[:, 2:9].to_numpy()
    elif dname == "turbine": # turbine 9
        df = pd.concat([pd.read_csv('data/turbine/gt_2011.csv', delimiter = ","), 
                        pd.read_csv('data/turbine/gt_2012.csv', delimiter = ","),
                        pd.read_csv('data/turbine/gt_2013.csv', delimiter = ","),
                        pd.read_csv('data/turbine/gt_2014.csv', delimiter = ","),
                        pd.read_csv('data/turbine/gt_2015.csv', delimiter = ",")])
        y = (df['NOX'] > 70).astype(int).to_numpy()
        X  = df.iloc[:, :9].to_numpy()
    elif dname == "wine": # wine 11
        df = pd.read_csv('data/wine/winequality-white.csv', delimiter = ";")
        y = (df['quality'] == 6).astype(int).to_numpy()
        X  = df[df.columns.drop('quality')].to_numpy()
    elif dname == "parkinson": # parkinson 16
        df = pd.read_csv('data/parkinson/parkinsons_updrs.data', delimiter = ",")
        y = (df['motor_UPDRS'] > 23).astype(int).to_numpy()
        X  = df.iloc[:, 6:].to_numpy()
    elif dname == "dry": # dry_bean 16
        df = pd.read_csv('data/dry/Dry_Bean_Dataset.csv', delimiter = ",")
        y = (df['Class'] == 'DERMASON').astype(int).to_numpy()
        X  = df[df.columns.drop('Class')].to_numpy()
    elif dname == "anuran": # anuran 21
        df = pd.read_csv('data/anuran/Frogs_MFCCs.csv', delimiter = ",")
        y = (df['Family'] == 'Hylidae').astype(int).to_numpy()
        X  = df.iloc[:, 1: 22].to_numpy()
    elif dname == "ml": # ml_prove 51
        df = pd.concat([pd.read_csv('data/ml/train.csv', delimiter = ",", header = None), 
                        pd.read_csv('data/ml/test.csv', delimiter = ",", header = None),
                        pd.read_csv('data/ml/validation.csv', delimiter = ",", header = None)])
        y = (df[56] == 1).astype(int).to_numpy()
        X  = df.iloc[:, : 51].to_numpy()
    else:
        sys.exit("Wrong dataset name.")

    y = y[~np.isnan(X).any(axis = 1)]
    X = X[~np.isnan(X).any(axis = 1)]
    X = X[:, nunique(X, 0) > 19]

    return X, y


res = []
for i in ['anuran', 'avila', 'bankruptcy', 'ccpp', 'cc', 'clean2', 'dry',
       'ees', 'electricity', 'gas', 'gt', 'higgs21', 'higgs7', 'htru', 'jm1',
       'ml', 'nomao', 'occupancy', 'parkinson', 'pendata', 'ring',
       'saac2', 'seizure', 'sensorless', 'seoul', 'shuttle', 'stocks',
       'sylva', 'turbine', 'wine']:
    print(i)
    X, y = load_data(i)
    res.append([i, X.shape[0], X.shape[1], np.round(sum(y)/len(y), 2)])

res = pd.DataFrame(res, columns=['name','n','m', 'pos'])
res.to_csv('datasets.csv')
    






