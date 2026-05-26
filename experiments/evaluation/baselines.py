import copy
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from .helpers import get_bi_param, n_leaves, opt_param
from .scoring import fit_score_classifier, model_size
from prelim.sd.bi import BI
from prelim.sd.prim import PRIM
from ..results.artifacts import write_result


def fit_reference_models(config, X, y, Xtest, ytest, tree_models, balanced_tree_models, rule_models, dtval, dtvalb, fileres):
    tiled_repeats = int(np.ceil(config.rules_sample_size / X.shape[0]))
    Xr = np.tile(X, [tiled_repeats, 1])
    yr = np.tile(y, tiled_repeats)

    for name, model in list(tree_models.items()) + list(balanced_tree_models.items()):
        score = fit_score_classifier(model, X, y, X, y, Xtest, ytest)
        write_result(
            fileres,
            name,
            "na",
            "na",
            score["train"],
            score["test"],
            model_size(name, model),
            score["elapsed"],
            "na",
            score["bactest"],
        )

    for name, model in rule_models.items():
        if name == "grl":
            Xfit, yfit = X, y
        else:
            Xfit, yfit = Xr, yr
        score = fit_score_classifier(model, Xfit, yfit, Xfit, yfit, Xtest, ytest)
        write_result(
            fileres,
            name,
            "na",
            "na",
            score["train"],
            score["test"],
            model_size(name, model),
            score["elapsed"],
            "na",
            score["bactest"],
        )

    par_vals = [2**number for number in [1, 2, 3, 4, 5, 6, 7]]
    parameters = {"max_leaf_nodes": par_vals}

    start = time.time()
    tmp = GridSearchCV(dtval, parameters, refit=False).fit(X, y).cv_results_
    tmp = opt_param(tmp, len(par_vals))
    dtval = DecisionTreeClassifier(max_leaf_nodes=par_vals[np.argmax(tmp)])
    dtval.fit(X, y)
    end = time.time()
    write_result(
        fileres,
        "dtval",
        "na",
        "na",
        dtval.score(X, y),
        dtval.score(Xtest, ytest),
        model_size("dtval", dtval),
        end - start,
        "na",
        balanced_accuracy_score(ytest, dtval.predict(Xtest)),
    )

    start = time.time()
    tmp = GridSearchCV(dtvalb, parameters, refit=False, scoring="balanced_accuracy").fit(X, y).cv_results_
    tmp = opt_param(tmp, len(par_vals))
    dtvalb = DecisionTreeClassifier(max_leaf_nodes=par_vals[np.argmax(tmp)], class_weight="balanced")
    dtvalb.fit(X, y)
    end = time.time()
    write_result(
        fileres,
        "dtvalb",
        "na",
        "na",
        dtvalb.score(X, y),
        dtvalb.score(Xtest, ytest),
        model_size("dtvalb", dtvalb),
        end - start,
        "na",
        balanced_accuracy_score(ytest, dtvalb.predict(Xtest)),
    )

    dtvalold = copy.deepcopy(dtval)
    dtvalbold = copy.deepcopy(dtvalb)
    dtval = DecisionTreeClassifier(max_leaf_nodes=max(n_leaves(dtval), 2))
    dtvalb = DecisionTreeClassifier(max_leaf_nodes=max(n_leaves(dtvalb), 2), class_weight="balanced")

    parsbi = get_bi_param(5, X.shape[1])
    start = time.time()
    tmp = GridSearchCV(BI(), {"depth": parsbi}, refit=False).fit(X, y).cv_results_
    tmp = opt_param(tmp, len(parsbi))
    bicv = BI(depth=parsbi[np.argmax(tmp)])
    bicv.fit(X, y)
    end = time.time()
    write_result(
        fileres,
        "bicv",
        "na",
        "na",
        bicv.score(X, y),
        bicv.score(Xtest, ytest),
        model_size("bicv", bicv),
        end - start,
        "na",
        "na",
    )
    bicv = BI(depth=bicv.get_nrestr())

    par_vals = [0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2]
    start = time.time()
    tmp = GridSearchCV(PRIM(), {"alpha": par_vals}, refit=False).fit(X, y).cv_results_
    tmp = opt_param(tmp, len(par_vals))
    primcv = PRIM(alpha=par_vals[np.argmax(tmp)])
    primcv.fit(X, y)
    end = time.time()
    write_result(
        fileres,
        "primcv",
        "na",
        "na",
        primcv.score(X, y),
        primcv.score(Xtest, ytest),
        model_size("primcv", primcv),
        end - start,
        "na",
        "na",
    )

    return {
        "dtval": dtval,
        "dtvalb": dtvalb,
        "dtvalold": dtvalold,
        "dtvalbold": dtvalbold,
        "bicv": bicv,
        "primcv": primcv,
    }
