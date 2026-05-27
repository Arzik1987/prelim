import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from .helpers import n_leaves


RULE_MODEL_NAMES = {"ripper", "irep", "grl"}
SD_MODEL_NAMES = {"primcv", "bicv"}
GENERATED_TREE_ALIASES = {
    "dt": "dtp",
    "dtc": "dtcp",
    "dtval": "dtvalp",
}


def model_size(name, model):
    if name in {"ripper", "irep"}:
        return len(model.ruleset_)
    if name == "grl":
        return len(model.rules_)
    if name in SD_MODEL_NAMES:
        return model.get_nrestr()
    return n_leaves(model)


def fidelity_score(predicted, reference):
    return np.count_nonzero(predicted == reference) / len(reference)


def fit_score_classifier(model, Xfit, yfit, Xtrain, ytrain, Xtest, ytest):
    start = time.time()
    model.fit(Xfit, yfit)
    end = time.time()
    return {
        "elapsed": end - start,
        "train": model.score(Xtrain, ytrain),
        "test": model.score(Xtest, ytest),
        "bactest": balanced_accuracy_score(ytest, model.predict(Xtest)),
    }


def fit_score_sd_model(model, Xfit, yfit, Xtrain, ytrain, Xtest, ytest):
    start = time.time()
    model.fit(Xfit, yfit)
    end = time.time()
    return {
        "elapsed": end - start,
        "train": model.score(Xtrain, ytrain),
        "test": model.score(Xtest, ytest),
    }


def get_supervised_models(meta_model, tree_models, balanced_tree_models, rule_models, dtval, dtvalb, is_balanced_metamodel, include_rules=True):
    if is_balanced_metamodel(meta_model):
        models = list(balanced_tree_models.items())
        if dtvalb is not None:
            models.append(("dtvalb", dtvalb))
        return models

    models = list(tree_models.items())
    if dtval is not None:
        models.append(("dtval", dtval))
    if include_rules:
        models.extend(rule_models.items())
    return models


def get_vva_models(meta_model, tree_models, balanced_tree_models, rule_models, dtval, dtvalb, primcv, bicv, is_balanced_metamodel):
    models = get_supervised_models(
        meta_model,
        tree_models,
        balanced_tree_models,
        rule_models,
        dtval,
        dtvalb,
        is_balanced_metamodel,
    )
    if is_balanced_metamodel(meta_model):
        return models
    return models + get_standard_sd_models(primcv, bicv)


def get_standard_sd_models(primcv, bicv):
    models = []
    if primcv is not None:
        models.append(("primcv", primcv))
    if bicv is not None:
        models.append(("bicv", bicv))
    return models
