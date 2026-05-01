import copy
import time

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from evaluation.helpers import get_bi_param, n_leaves, opt_param
from prelim.generators.vva import Gen_vva
from prelim.sd.bi import BI
from prelim.sd.prim import PRIM
from results.artifacts import write_meta, write_result


RULE_MODEL_NAMES = {"ripper", "irep"}
SD_MODEL_NAMES = {"primcv", "bicv"}
GENERATED_TREE_ALIASES = {
    "dt": "dtp",
    "dtc": "dtcp",
    "dtval": "dtvalp",
}


def model_size(name, model):
    if name in RULE_MODEL_NAMES:
        return len(model.ruleset_)
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
        return list(balanced_tree_models.items()) + [("dtvalb", dtvalb)]

    models = list(tree_models.items()) + [("dtval", dtval)]
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
    return models + [("primcv", primcv), ("bicv", bicv)]


def get_standard_sd_models(primcv, bicv):
    return [("primcv", primcv), ("bicv", bicv)]


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
        score = fit_score_classifier(model, Xr, yr, Xr, yr, Xtest, ytest)
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


def fit_generators_and_metamodels(state, filetme, is_balanced_metamodel):
    for generator in state.generators:
        start = time.time()
        generator.fit(state.X, state.y)
        end = time.time()
        write_meta(filetme, generator.my_name() + "time", end - start)

    for meta_model in state.all_metamodels:
        start = time.time()
        meta_model.fit(state.X, state.y)
        end = time.time()
        write_meta(filetme, meta_model.my_name() + "time", end - start)
        write_meta(filetme, meta_model.my_name() + "acccv", meta_model.fit_score())

        ypredtest = meta_model.predict(state.Xtest)
        write_meta(filetme, meta_model.my_name() + "fid", fidelity_score(ypredtest, state.ydeftest))
        write_meta(filetme, meta_model.my_name() + "acc", accuracy_score(state.ytest, ypredtest))
        write_meta(filetme, meta_model.my_name() + "bac", balanced_accuracy_score(state.ytest, ypredtest))

        for name, model in get_supervised_models(
            meta_model,
            state.tree_models,
            state.balanced_tree_models,
            state.rule_models,
            state.dtvalold,
            state.dtvalbold,
            is_balanced_metamodel,
        ):
            write_meta(filetme, meta_model.my_name() + name + "fid", fidelity_score(model.predict(state.Xtest), ypredtest))


def evaluate_rerx(state, fileres, is_balanced_metamodel):
    for meta_model in state.all_metamodels:
        state.genrerx.fit(state.X, state.y, meta_model)
        ypredtest = meta_model.predict(state.Xtest)
        Xnew = state.genrerx.sample()
        ynew = meta_model.predict(Xnew)

        for name, model in get_supervised_models(
            meta_model,
            state.tree_models,
            state.balanced_tree_models,
            state.rule_models,
            state.dtval,
            state.dtvalb,
            is_balanced_metamodel,
        ):
            score = fit_score_classifier(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
            write_result(
                fileres,
                name,
                "rerx",
                meta_model.my_name(),
                score["train"],
                score["test"],
                model_size(name, model),
                score["elapsed"],
                fidelity_score(model.predict(state.Xtest), ypredtest),
                score["bactest"],
            )

        if not is_balanced_metamodel(meta_model):
            ynew = meta_model.predict_proba(Xnew)
            for name, model in get_standard_sd_models(state.primcv, state.bicv):
                score = fit_score_sd_model(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
                write_result(
                    fileres,
                    name,
                    "rerx",
                    meta_model.my_name(),
                    score["train"],
                    score["test"],
                    model_size(name, model),
                    score["elapsed"],
                    "na",
                    "na",
                )


def evaluate_vva(config, state, fileres, filetme, is_balanced_metamodel):
    for meta_model in state.all_metamodels:
        ntrain = int(np.ceil(state.X.shape[0] * 2 / 3))
        Xtrain = state.X[:ntrain, :].copy()
        Xval = state.X[ntrain:, :].copy()
        ytrain = state.y[:ntrain].copy()
        yval = state.y[ntrain:].copy()
        start = time.time()
        state.genvva.fit(Xtrain, meta_model)
        end = time.time()
        write_meta(filetme, meta_model.my_name() + "vva", end - start)
        ypredtest = meta_model.predict(state.Xtest)

        for name, model in get_vva_models(
            meta_model,
            state.tree_models,
            state.balanced_tree_models,
            state.rule_models,
            state.dtval,
            state.dtvalb,
            state.primcv,
            state.bicv,
            is_balanced_metamodel,
        ):
            start = time.time()
            if name in SD_MODEL_NAMES:
                model.fit(Xtrain, meta_model.predict_proba(Xtrain))
            else:
                model.fit(Xtrain, ytrain)
            sctest0 = model.score(Xval, yval)
            ropt = 0

            if state.genvva.will_generate():
                for r in config.vva_grid:
                    Xnew = state.genvva.sample(r)
                    ynew = meta_model.predict(Xnew)
                    Xnew = np.concatenate([Xnew, Xtrain])
                    ynew = np.concatenate([ynew, ytrain])
                    if name in SD_MODEL_NAMES:
                        model.fit(Xnew, meta_model.predict_proba(Xnew))
                    else:
                        model.fit(Xnew, ynew)
                    sctest = model.score(Xval, yval)
                    if sctest > sctest0:
                        sctest0 = sctest
                        ropt = r

            end = time.time()
            write_meta(filetme, name + meta_model.my_name() + "vvaopt", end - start)
            write_meta(filetme, name + meta_model.my_name() + "ropt", ropt)

            start = time.time()
            if ropt > 0:
                Xnew = Gen_vva().fit(state.X, meta_model).sample(ropt)
                ynew = meta_model.predict(Xnew)
                Xnew = np.concatenate([Xnew, state.X])
                ynew = np.concatenate([ynew, state.y])
            else:
                Xnew = state.X.copy()
                ynew = state.y.copy()
            end = time.time()
            write_meta(filetme, name + meta_model.my_name() + "vvagen", end - start)

            if name in SD_MODEL_NAMES:
                score = fit_score_sd_model(model, Xnew, meta_model.predict_proba(Xnew), state.X, state.y, state.Xtest, state.ytest)
                fidelity = "na"
                bactest = "na"
            else:
                score = fit_score_classifier(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
                fidelity = fidelity_score(model.predict(state.Xtest), ypredtest)
                bactest = score["bactest"]

            write_result(
                fileres,
                name,
                "vva",
                meta_model.my_name(),
                score["train"],
                score["test"],
                model_size(name, model),
                score["elapsed"],
                fidelity,
                bactest,
            )


def evaluate_standard_generator(generator, Xgen, meta_model, state, config, fileres, filetme):
    ypredtest = meta_model.predict(state.Xtest)

    start = time.time()
    Xnew = Xgen.copy()
    predicted_labels = meta_model.predict(Xnew)
    end = time.time()
    write_meta(filetme, generator.my_name() + meta_model.my_name(), end - start)

    for name, model in [("dt", state.tree_models["dt"]), ("dtc", state.tree_models["dtc"]), ("dtval", state.dtval)]:
        score = fit_score_classifier(model, Xnew, predicted_labels, state.X, state.y, state.Xtest, state.ytest)
        write_result(
            fileres,
            GENERATED_TREE_ALIASES[name],
            generator.my_name(),
            meta_model.my_name(),
            score["train"],
            score["test"],
            model_size(name, model),
            score["elapsed"],
            fidelity_score(model.predict(state.Xtest), ypredtest),
            score["bactest"],
        )

    Xnew = Xnew[: config.generated_sample_size - len(state.y), :]
    predicted_labels = predicted_labels[: config.generated_sample_size - len(state.y)]
    Xnew = np.concatenate([state.X, Xnew])
    predicted_labels = np.concatenate([state.y, predicted_labels])

    for name, model in [("dt", state.tree_models["dt"]), ("dtc", state.tree_models["dtc"]), ("dtval", state.dtval)]:
        score = fit_score_classifier(model, Xnew, predicted_labels, state.X, state.y, state.Xtest, state.ytest)
        write_result(
            fileres,
            name,
            generator.my_name(),
            meta_model.my_name(),
            score["train"],
            score["test"],
            model_size(name, model),
            score["elapsed"],
            fidelity_score(model.predict(state.Xtest), ypredtest),
            score["bactest"],
        )

    Xnew = Xnew[: config.rules_sample_size, :]
    predicted_labels = predicted_labels[: config.rules_sample_size]

    for name, model in state.rule_models.items():
        score = fit_score_classifier(model, Xnew, predicted_labels, state.X, state.y, state.Xtest, state.ytest)
        write_result(
            fileres,
            name,
            generator.my_name(),
            meta_model.my_name(),
            score["train"],
            score["test"],
            model_size(name, model),
            score["elapsed"],
            fidelity_score(model.predict(state.Xtest), ypredtest),
            score["bactest"],
        )

    ynew = meta_model.predict_proba(Xnew)
    for name, model in get_standard_sd_models(state.primcv, state.bicv):
        score = fit_score_sd_model(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
        write_result(
            fileres,
            name,
            generator.my_name(),
            meta_model.my_name(),
            score["train"],
            score["test"],
            model_size(name, model),
            score["elapsed"],
            "na",
            "na",
        )


def evaluate_balanced_generator(generator, Xgen, meta_model, state, config, fileres, filetme):
    ypredtest = meta_model.predict(state.Xtest)

    start = time.time()
    Xnew = Xgen[: config.generated_sample_size - len(state.y), :].copy()
    ynew = meta_model.predict(Xnew)
    Xnew = np.concatenate([state.X, Xnew])
    ynew = np.concatenate([state.y, ynew])
    end = time.time()
    write_meta(filetme, generator.my_name() + meta_model.my_name(), end - start)

    for name, model in list(state.balanced_tree_models.items()) + [("dtvalb", state.dtvalb)]:
        score = fit_score_classifier(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
        write_result(
            fileres,
            name,
            generator.my_name(),
            meta_model.my_name(),
            score["train"],
            score["test"],
            model_size(name, model),
            score["elapsed"],
            fidelity_score(model.predict(state.Xtest), ypredtest),
            score["bactest"],
        )


def evaluate_sampled_generators(config, state, fileres, filetme):
    for generator in state.generators:
        start = time.time()
        Xgen = generator.sample(config.generated_sample_size)
        end = time.time()
        write_meta(filetme, generator.my_name() + "gen", end - start)

        for meta_model in state.standard_metamodels:
            evaluate_standard_generator(generator, Xgen, meta_model, state, config, fileres, filetme)
        for meta_model in state.balanced_metamodels:
            evaluate_balanced_generator(generator, Xgen, meta_model, state, config, fileres, filetme)


def evaluate_ssl(config, state, fileres, is_balanced_metamodel, get_new_test):
    Xtest, ytest, Xgen = get_new_test(state.Xtest, state.ytest, len(state.y), new_size=config.ssl_pool_size)
    for meta_model in state.all_metamodels:
        ypredtest = meta_model.predict(Xtest)
        ynew = np.concatenate([meta_model.predict(Xgen), state.y])
        Xnew = np.concatenate([Xgen, state.X])

        for name, model in get_supervised_models(
            meta_model,
            state.tree_models,
            state.balanced_tree_models,
            state.rule_models,
            state.dtval,
            state.dtvalb,
            is_balanced_metamodel,
        ):
            score = fit_score_classifier(model, Xnew, ynew, state.X, state.y, Xtest, ytest)
            write_result(
                fileres,
                name,
                "ssl",
                meta_model.my_name(),
                score["train"],
                score["test"],
                model_size(name, model),
                score["elapsed"],
                fidelity_score(model.predict(Xtest), ypredtest),
                score["bactest"],
            )

        if not is_balanced_metamodel(meta_model):
            ynew = meta_model.predict_proba(Xnew)
            for name, model in get_standard_sd_models(state.primcv, state.bicv):
                score = fit_score_sd_model(model, Xnew, ynew, state.X, state.y, Xtest, ytest)
                write_result(
                    fileres,
                    name,
                    "ssl",
                    meta_model.my_name(),
                    score["train"],
                    score["test"],
                    model_size(name, model),
                    score["elapsed"],
                    "na",
                    "na",
                )
