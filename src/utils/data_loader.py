import numpy as np
import pandas as pd
# import arff
import sys

def nunique(a, axis):
    return (np.diff(np.sort(a,axis=axis),axis=axis)!=0).sum(axis=axis) + 1

def load_data(dname):
    if dname == "occupancy": # 6 occupancy
        df = pd.concat([pd.read_csv('data/new/occupancy/datatest.txt', delimiter = ","), 
                pd.read_csv('data/new/occupancy/datatest2.txt', delimiter = ","),
                pd.read_csv('data/new/occupancy/datatraining.txt', delimiter = ",")])
        df['date'] = pd.to_datetime(df['date']).dt.hour
        y = (df['Occupancy'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('Occupancy')].to_numpy()
    elif dname == "higgs7": # 7 higgs
        df = pd.read_csv('data/new/higgs/phpZLgL9q.csv', delimiter = ",", nrows = 98049)
        y = (df['class'] == 1).astype(int).to_numpy()
        X  = df.filter(regex = 'm_', axis = 1).to_numpy()
    elif dname == "electricity": # 7 electricity
        df = pd.read_csv('data/new_additional/electricity/electricity-normalized.csv', delimiter = ",")
        y = (df['class'] == 'UP').astype(int).to_numpy()
        X  = df[df.columns.drop('class')].to_numpy()
    elif dname == "htru": # 8 htru
        df = pd.read_csv('data/new/htru/HTRU_2.csv', delimiter = ",", header = None)
        y = (df[8] == 1).astype(int).to_numpy()
        X  = df.iloc[:, : 8].to_numpy()
    elif dname == "shuttle": # 9 shuttle
        df = pd.read_csv('data/new/shuttle/shuttle.tsv', delimiter = "\t")
        y = (df['target'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('target')].to_numpy()
    elif dname == "avila": # 10 avila
        df = pd.concat([pd.read_csv('data/new/avila/avila-tr.txt', delimiter = ",", header = None), 
                        pd.read_csv('data/new/avila/avila-ts.txt', delimiter = ",", header = None)])
        y = (df[10] == 'A').astype(int).to_numpy()
        X  = df.iloc[:, : 10].to_numpy()
    elif dname == "gt": # 10 gamma_telescope
        df = pd.read_csv('data/new/gamma_telescope/magic04.data', delimiter = ",", header = None)
        y = (df[10] == "g").astype(int).to_numpy()
        X  = df.iloc[:, : 10].to_numpy()
    elif dname == "cc": # 14 credit_cards
        # df = pd.read_excel('data/new/credit_cards/default of credit card clients.xls', header = 1)
        # df.to_csv('data/new/credit_cards/default_of_credit_card_clients.csv', index = False)
        df = pd.read_csv('data/new/credit_cards/default_of_credit_card_clients.csv', delimiter = ",")
        y = (df.iloc[:, 24] == 1).astype(int).to_numpy()
        X  = df.iloc[:, 1:24].to_numpy()
    elif dname == "ees": # 14 eeg_eye_state
        df = pd.read_csv('data/new/eeg_eye_state/phplE7q6h.csv', delimiter = ",")
        y = (df['Class'] == 1).astype(int).to_numpy()
        X  = df.iloc[:, : 14].to_numpy()
    elif dname == "pendata": # 16 pendata
        df = pd.read_csv('data/new_additional/pendata/dataset_32_pendigits.csv', delimiter = ",")
        y = (df['class'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('class')].to_numpy()
    elif dname == "ring": # 20 ring
        df = pd.read_csv('data/new/ring/ring.tsv', delimiter = "\t")
        y = (df['target'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('target')].to_numpy()
    elif dname == "sylva": # 20 sylva
        df = pd.read_csv('data/new/sylva/sylva_prior.csv', delimiter = ",")
        y = (df['label'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('label')].to_numpy()
    elif dname == "higgs21": # 21 higgs
        df = pd.read_csv('data/new/higgs/phpZLgL9q.csv', delimiter = ",", nrows = 98049)
        y = (df['class'] == 1).astype(int).to_numpy()
        df = df[df.columns.drop('class')]
        X = df[df.columns.drop(list(df.filter(regex = 'm_', axis = 1)))].to_numpy()  
    elif dname == "jm1": # 21 jm1
        df = pd.read_csv('data/new/jm1/jm1.csv', delimiter = ",")
        y = (df['defects'] == True).astype(int).to_numpy()
        X  = df[df.columns.drop('defects')].to_numpy()
        X[X=='?'] = 'nan'
        X = X.astype(np.float)
    elif dname == "saac2": # 21 saac2
        df = pd.read_csv('data/new_additional/saac2/SAAC2.csv', delimiter = ",")
        y = (df['class'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('class')].to_numpy()
        X[X=='?'] = 'nan'
        X = X.astype(np.float)
    elif dname == "stocks": # 21 stocks
        df = pd.read_csv('data/new/stocks/phpg2t68G.csv', delimiter = ",")
        y = (df['attribute_21'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('attribute_21')].to_numpy()
    elif dname == "sensorless": # 48 sensorless
        df = pd.read_csv('data/new/sensorless/Sensorless_drive_diagnosis.txt', delimiter = " ", header = None)
        y = (df[48] == 1).astype(int).to_numpy()
        X  = df.iloc[:, : 48].to_numpy()
    elif dname == "bankruptcy": # 64 bankruptcy
        # df = pd.DataFrame(arff.load(open('data/new/bankruptcy/3year.arff'))['data'])
        # df.to_csv('data/new/bankruptcy/3year.csv', index = False, header = False)
        df = pd.read_csv('data/new/bankruptcy/3year.csv', delimiter = ",", header = None)
        y = (df[64] == 1).astype(int).to_numpy()
        X  = df.iloc[:, : 64].to_numpy()
    elif dname == "nomao": # 69 nomao
        df = pd.read_csv('data/new_additional/nomao/phpDYCOet.csv', delimiter = ",")
        y = (df['Class'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('Class')].to_numpy()
    elif dname == "gas": # 128 gas 
        df = pd.read_csv('data/new/gas/phpbL6t4U.csv', delimiter = ",")
        y = (df['Class'] == 1).astype(int).to_numpy()
        X  = df.iloc[:, : 128].to_numpy()
    elif dname == "clean2": # 166 clean2
        df = pd.read_csv('data/new/clean2/clean2.tsv', delimiter = "\t")
        y = (df['target'] == 1).astype(int).to_numpy()
        X  = df.filter(regex = '^f', axis = 1).to_numpy()
    elif dname == "seizure": # 178 seizure
        df = pd.read_csv('data/new/seizure/data.csv', delimiter = ",")
        y = (df['y'] == 1).astype(int).to_numpy()
        X  = df.filter(regex = 'X', axis = 1).to_numpy()
    elif dname == "smartphone": # 561 smartphone 
        df = pd.read_csv('data/new_additional/smartphone/php88ZB4Q.csv', delimiter = ",")
        y = (df['Class'] == 1).astype(int).to_numpy()
        X  = df[df.columns.drop('Class')].to_numpy()
    # regression folder    
    elif dname == "ccpp": # ccpp 4
        # df = pd.read_excel('data/regression/ccpp/Folds5x2_pp.xls', sheet_name = 'Sheet1')
        # df.to_csv('data/regression/ccpp/Folds5x2_pp.csv', index = False)
        df = pd.read_csv('data/regression/ccpp/Folds5x2_pp.csv', delimiter = ",")
        y = (df['PE'] > 455).astype(int).to_numpy()
        X  = df[df.columns.drop('PE')].to_numpy()        
    elif dname == "seoul": # seoul_bike 7
        df = pd.read_csv('data/regression/seoul_bike/SeoulBikeData.csv', delimiter = ",", skiprows = 1, header = None)
        y = (df.iloc[:, 1] > 800).astype(int).to_numpy()
        X  = df.iloc[:, 2:9].to_numpy()
    elif dname == "turbine": # turbine 9
        df = pd.concat([pd.read_csv('data/regression/turbine/gt_2011.csv', delimiter = ","), 
                        pd.read_csv('data/regression/turbine/gt_2012.csv', delimiter = ","),
                        pd.read_csv('data/regression/turbine/gt_2013.csv', delimiter = ","),
                        pd.read_csv('data/regression/turbine/gt_2014.csv', delimiter = ","),
                        pd.read_csv('data/regression/turbine/gt_2015.csv', delimiter = ",")])
        y = (df['NOX'] > 70).astype(int).to_numpy()
        X  = df.iloc[:, :9].to_numpy()
    elif dname == "wine": # wine 11
        df = pd.read_csv('data/regression/wine/winequality-white.csv', delimiter = ";")
        y = (df['quality'] == 6).astype(int).to_numpy()
        X  = df[df.columns.drop('quality')].to_numpy()
    elif dname == "parkinson": # parkinson 16
        df = pd.read_csv('data/regression/parkinson/parkinsons_updrs.data', delimiter = ",")
        y = (df['motor_UPDRS'] > 23).astype(int).to_numpy()
        X  = df.iloc[:, 6:].to_numpy()
    elif dname == "dry": # dry_bean 16
        # df = pd.read_excel('data/regression/dry_bean/Dry_Bean_Dataset.xls', sheet_name = 'Dry_Beans_Dataset')
        # df.to_csv('data/regression/dry_bean/Dry_Bean_Dataset.csv', index = False)
        df = pd.read_csv('data/regression/dry_bean/Dry_Bean_Dataset.csv', delimiter = ",")
        y = (df['Class'] == 'DERMASON').astype(int).to_numpy()
        X  = df[df.columns.drop('Class')].to_numpy()
    elif dname == "anuran": # anuran 21
        df = pd.read_csv('data/regression/anuran/Frogs_MFCCs.csv', delimiter = ",")
        y = (df['Family'] == 'Hylidae').astype(int).to_numpy()
        X  = df.iloc[:, 1: 22].to_numpy()
    elif dname == "ml": # ml_prove 51
        df = pd.concat([pd.read_csv('data/regression/ml_prove/train.csv', delimiter = ",", header = None), 
                        pd.read_csv('data/regression/ml_prove/test.csv', delimiter = ",", header = None),
                        pd.read_csv('data/regression/ml_prove/validation.csv', delimiter = ",", header = None)])
        y = (df[56] == 1).astype(int).to_numpy()
        X  = df.iloc[:, : 51].to_numpy()
    else:
        sys.exit("Wrong dataset name.")

    y = y[~np.isnan(X).any(axis = 1)]
    X = X[~np.isnan(X).any(axis = 1)]
    X = X[:, nunique(X, 0) > 19]
    # sum(y)/len(y)
    return X, y

# =============================================================================
# X, y = load_data("seoul_bike")
# =============================================================================





