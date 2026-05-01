import numpy as np
from sklearn.preprocessing import StandardScaler

from data.loader import load_data
from data.splitter import DataSplitter
from results.artifacts import result_paths, write_meta


def load_experiment_split(config, split_index, dataset_name, dataset_size, data_loader=load_data):
    paths = result_paths(config, dataset_name, split_index, dataset_size)
    X, y = data_loader(dataset_name)
    splitter = DataSplitter(seed=config.split_seed)
    splitter.fit(X, y)
    splitter.configure(config.nsets, dataset_size)
    X, y = splitter.get_train(split_index)
    if y.sum() == 0:
        open(paths["zeros"], "a", encoding="utf-8").close()
        return None

    Xtest, ytest = splitter.get_test(split_index)
    variable_mask = X.max(axis=0) != X.min(axis=0)
    Xtest = Xtest[:, variable_mask]
    X = X[:, variable_mask]

    scaler = StandardScaler()
    scaler.fit(X)
    return {
        "X": scaler.transform(X),
        "y": y,
        "Xtest": scaler.transform(Xtest),
        "ytest": ytest,
    }


def write_default_classifier_metadata(filetme, y, ytest):
    default_prediction = 1 if y.mean() >= 0.5 else 0
    write_meta(filetme, "testprec", ytest.mean() if default_prediction == 1 else 1 - ytest.mean())
    write_meta(filetme, "trainprec", y.mean() if default_prediction == 1 else 1 - y.mean())
    return np.ones(len(ytest)) * default_prediction
