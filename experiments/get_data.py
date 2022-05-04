import urllib.request
import zipfile
import tarfile
import os
import arff
import pandas as pd
from tqdm import tqdm

#### progress bar, see: https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

FILEPATH = os.path.dirname(os.path.abspath(__file__)) + '/data'
if not os.path.exists(FILEPATH):
    os.makedirs(FILEPATH)

class DownloadProgressBar(tqdm):
    def update_to(self, b = 1, bsize = 1, tsize = None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path, dname):
    with DownloadProgressBar(unit = 'B', unit_scale = True,
                             miniters = 1, desc = dname) as t:
        urllib.request.urlretrieve(url, filename = output_path, reporthook = t.update_to)


def get_single(dname):
    if dname == 'anuran':
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran%20Calls%20(MFCCs).zip',\
                                   FILEPATH + '/' + dname + '.zip', dname)
        with zipfile.ZipFile(FILEPATH + '/' + dname + '.zip', 'r') as zip_ref:
            zip_ref.extractall(FILEPATH + '/' + dname)
        os.remove(FILEPATH + '/' + dname + '.zip')
            
    elif dname == 'avila':  
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip',\
                                   FILEPATH + '/' + dname + '.zip', dname)
        with zipfile.ZipFile(FILEPATH + '/' + dname + '.zip', 'r') as zip_ref:
            zip_ref.extractall(FILEPATH)
        os.remove(FILEPATH + '/' + dname + '.zip')

    elif dname == 'bankruptcy':
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00365/data.zip',\
                                   FILEPATH + '/' + dname + '.zip', dname)
        with zipfile.ZipFile(FILEPATH + '/' + dname + '.zip', 'r') as zip_ref:
            zip_ref.extractall(FILEPATH + '/' + dname)
        os.remove(FILEPATH + '/' + dname + '.zip')
        print('converting to csv ...')
        df = pd.DataFrame(arff.load(open(FILEPATH + '/' + dname + '/3year.arff'))['data'])
        df.to_csv(FILEPATH + '/' + dname + '/3year.csv', index = False, header = False)

    elif dname == 'cc':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',\
                                       FILEPATH + '/' + dname + '/default of credit card clients.xls', dname)
        print('converting to csv ...')
        df = pd.read_excel(FILEPATH + '/' + dname + '/default of credit card clients.xls', header = 1)
        df.to_csv(FILEPATH + '/' + dname + '/default_of_credit_card_clients.csv', index = False)

    elif dname == 'ccpp':
        download_url('http://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip',\
                                   FILEPATH + '/' + dname + '.zip', dname)
        with zipfile.ZipFile(FILEPATH + '/' + dname + '.zip', 'r') as zip_ref:
            zip_ref.extractall(FILEPATH)
        os.rename(FILEPATH + '/' + 'CCPP', FILEPATH + '/' + dname)
        os.remove(FILEPATH + '/' + dname + '.zip')
        print('converting to csv ...')
        df = pd.read_excel(FILEPATH + '/' + dname + '/Folds5x2_pp.ods', engine = 'odf')
        df.to_csv(FILEPATH + '/' + dname + '/Folds5x2_pp.csv', index = False)

    elif dname == 'clean2':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://github.com/EpistasisLab/pmlb/raw/master/datasets/clean2/clean2.tsv.gz',\
                                   FILEPATH + '/' + dname + '/clean2.tsv.gz', dname)

    elif dname == 'dry':
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip',\
                                   FILEPATH + '/' + dname + '.zip', dname)
        with zipfile.ZipFile(FILEPATH + '/' + dname + '.zip', 'r') as zip_ref:
            zip_ref.extractall(FILEPATH)
        os.rename(FILEPATH + '/' + 'DryBeanDataset', FILEPATH + '/' + dname)
        os.remove(FILEPATH + '/' + dname + '.zip')
        print('converting to csv ...')
        df = pd.DataFrame(arff.load(open(FILEPATH + '/' + dname + '/Dry_Bean_Dataset.arff'))['data'])
        df.set_axis(['Area', 'Perimeter'	, 'MajorAxisLength', 'MinorAxisLength', 'AspectRation',
                     'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent'	, 'Solidity', 'roundness',	
                     'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Class'], 
                    axis = 1, inplace = True)
        df.to_csv(FILEPATH + '/' + dname + '/Dry_Bean_Dataset.csv', index = False)

    elif dname == 'ees':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://www.openml.org/data/get_csv/1587924/phplE7q6h',\
                                   FILEPATH + '/' + dname + '/phplE7q6h.csv', dname)

    elif dname == 'electricity':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://www.openml.org/data/get_csv/2419/electricity-normalized.arff',\
                                   FILEPATH + '/' + dname + '/electricity-normalized.csv', dname)

    elif dname == 'gas':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://www.openml.org/data/get_csv/1588715/phpbL6t4U',\
                                   FILEPATH + '/' + dname + '/phpbL6t4U.csv', dname)

    elif dname == 'gt':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data',\
                                   FILEPATH + '/' + dname + '/magic04.data', dname)

    elif dname == 'higgs':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://www.openml.org/data/get_csv/2063675/phpZLgL9q',\
                                   FILEPATH + '/' + dname + '/phpZLgL9q.csv', dname)

    elif dname == 'htru':
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip',\
                                   FILEPATH + '/' + dname + '.zip', dname)
        with zipfile.ZipFile(FILEPATH + '/' + dname + '.zip', 'r') as zip_ref:
            zip_ref.extractall(FILEPATH + '/' + dname)
        os.remove(FILEPATH + '/' + dname + '.zip')

    elif dname == 'jm1':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://www.openml.org/data/get_csv/53936/jm1.arff',\
                                   FILEPATH + '/' + dname + '/jm1.csv', dname)

    elif dname == 'ml':
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00249/ml-prove.tar.gz',\
                                   FILEPATH + '/' + dname + '.tar.gz', dname)
        tar = tarfile.open(FILEPATH + '/' + dname + '.tar.gz', 'r:gz')
        tar.extractall(FILEPATH)
        tar.close()
        os.rename(FILEPATH + '/' + 'ml-prove', FILEPATH + '/' + dname)
        os.remove(FILEPATH + '/' + dname + '.tar.gz')

    elif dname == 'nomao':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://www.openml.org/data/get_csv/1592278/phpDYCOet',\
                                   FILEPATH + '/' + dname + '/phpDYCOet.csv', dname)

    elif dname == 'occupancy':
        download_url('http://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip',\
                                   FILEPATH + '/' + dname + '.zip', dname)
        with zipfile.ZipFile(FILEPATH + '/' + dname + '.zip', 'r') as zip_ref:
            zip_ref.extractall(FILEPATH + '/' + dname)
        os.remove(FILEPATH + '/' + dname + '.zip')

    elif dname == 'parkinson':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data',\
                                   FILEPATH + '/' + dname + '/parkinsons_updrs.data', dname)

    elif dname == 'pendata':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://www.openml.org/data/get_csv/32/dataset_32_pendigits.arff',\
                                   FILEPATH + '/' + dname + '/dataset_32_pendigits.csv', dname)

    elif dname == 'ring':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://github.com/EpistasisLab/pmlb/raw/master/datasets/ring/ring.tsv.gz',\
                                   FILEPATH + '/' + dname + '/ring.tsv.gz', dname)

    elif dname == 'saac2':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://www.openml.org/data/get_csv/21230748/SAAC2.arff',\
                                   FILEPATH + '/' + dname + '/SAAC2.csv', dname)

    elif dname == 'seizure':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://github.com/akshayg056/Epileptic-seizure-detection-/raw/master/data.csv',\
                                   FILEPATH + '/' + dname + '/data.csv', dname)

    elif dname == 'sensorless':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('http://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt',\
                                   FILEPATH + '/' + dname + '/Sensorless_drive_diagnosis.txt', dname)
    
    elif dname == 'seoul':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv',\
                                   FILEPATH + '/' + dname + '/SeoulBikeData.csv', dname)
    
    elif dname == 'shuttle':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://github.com/EpistasisLab/pmlb/raw/master/datasets/shuttle/shuttle.tsv.gz',\
                                   FILEPATH + '/' + dname + '/shuttle.tsv.gz', dname)    

    elif dname == 'stocks':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://www.openml.org/data/get_csv/2160285/phpg2t68G',\
                                   FILEPATH + '/' + dname + '/phpg2t68G.csv', dname)

    elif dname == 'sylva':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://www.openml.org/data/get_csv/53923/sylva_prior.arff',\
                                   FILEPATH + '/' + dname + '/sylva_prior.csv', dname)

    elif dname == 'turbine':
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00551/pp_gas_emission.zip',\
                                   FILEPATH + '/' + dname + '.zip', dname)
        with zipfile.ZipFile(FILEPATH + '/' + dname + '.zip', 'r') as zip_ref:
            zip_ref.extractall(FILEPATH + '/' + dname)
        os.remove(FILEPATH + '/' + dname + '.zip')       

    elif dname == 'wine':
        if not os.path.exists(FILEPATH + '/' + dname):
            os.makedirs(FILEPATH + '/' + dname)
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',\
                                   FILEPATH + '/' + dname + '/winequality-red.csv', dname)
        download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',\
                                   FILEPATH + '/' + dname + '/winequality-white.csv', dname)
    
    else: 
        raise ValueError('{dname} is a wrong dataset name'.format(dname = repr(dname)))
  

# get_single('anuran')

def get_multiple(dnames):
    for dname in dnames:
        get_single(dname)

dnames = ['anuran', 'avila', 'bankruptcy', 'ccpp', 'cc', 'clean2', 'dry',
        'ees', 'electricity', 'gas', 'gt', 'higgs', 'htru', 'jm1',
        'ml', 'nomao', 'occupancy', 'parkinson', 'pendata', 'ring',
        'saac2', 'seizure', 'sensorless', 'seoul', 'shuttle', 'stocks',
        'sylva', 'turbine', 'wine']
get_multiple(dnames)
        
        
