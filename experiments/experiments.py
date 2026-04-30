# Prevent numpy multithreading: https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import argparse
import copy
import json
import logging
import time
import traceback
from dataclasses import dataclass
from itertools import product

import numpy as np
import wittgenstein as lw
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import (
    DEFAULT_DATASET_NAMES,
    DEFAULT_DATASET_SIZES,
    DEFAULT_VVA_GRID,
    ExperimentConfig,
    default_run_id,
    ensure_run_layout,
    parse_csv_list,
)
from metamodels.rf import Meta_rf
from metamodels.rfb import Meta_rf_bal
from metamodels.xgb import Meta_xgb
from metamodels.xgbb import Meta_xgb_bal
from prelim.generators.adasyn import Gen_adasyn
from prelim.generators.dummy import Gen_dummy
from prelim.generators.gmm import Gen_gmmbic, Gen_gmmbical
from prelim.generators.kde import Gen_kdebw
from prelim.generators.kdeb import Gen_kdeb
from prelim.generators.kdem import Gen_kdebwm
from prelim.generators.munge import Gen_munge
from prelim.generators.rand import Gen_randn, Gen_randu
from prelim.generators.rerx import Gen_rerx
from prelim.generators.rfdens import Gen_rfdens
from prelim.generators.smote import Gen_smote
from prelim.generators.vva import Gen_vva
from prelim.sd.bi import BI
from prelim.sd.prim import PRIM
from utils.data_loader import load_data
from utils.data_splitter import DataSplitter
from utils.helpers import get_bi_param, get_new_test, n_leaves, opt_param


GENERATOR_FACTORIES = (
    Gen_gmmbic,
    Gen_kdebw,
    Gen_munge,
    Gen_randu,
    Gen_randn,
    Gen_dummy,
    Gen_gmmbical,
    Gen_smote,
    Gen_adasyn,
    Gen_rfdens,
    Gen_kdebwm,
    Gen_kdeb,
)

STANDARD_METAMODEL_FACTORIES = (
    Meta_rf,
    Meta_xgb,
)

BALANCED_METAMODEL_FACTORIES = (
    Meta_rf_bal,
    Meta_xgb_bal,
)

TREE_MODEL_FACTORIES = (
    ('dt', lambda: DecisionTreeClassifier(min_samples_split = 10)),
    # one could restrict depth instead. Results will be worse, but
    # ranking of generator's will not generally change (still kde is the best)
    ('dtc', lambda: DecisionTreeClassifier(max_leaf_nodes = 8)),
)

BALANCED_TREE_MODEL_FACTORIES = (
    ('dtb', lambda: DecisionTreeClassifier(min_samples_split = 10, class_weight = 'balanced')),
    ('dtcb', lambda: DecisionTreeClassifier(max_leaf_nodes = 8, class_weight = 'balanced')),
)

RULE_MODEL_FACTORIES = (
    ('ripper', lambda: lw.RIPPER(max_rules = 8)),
    ('irep', lambda: lw.IREP(max_rules = 8)),
)

RULE_MODEL_NAMES = {'ripper', 'irep'}
SD_MODEL_NAMES = {'primcv', 'bicv'}
GENERATED_TREE_ALIASES = {
    'dt': 'dtp',
    'dtc': 'dtcp',
    'dtval': 'dtvalp',
}


def build_generators():
    return [factory() for factory in GENERATOR_FACTORIES], Gen_rerx(), Gen_vva()


def build_metamodel_groups():
    standard = [factory() for factory in STANDARD_METAMODEL_FACTORIES]
    balanced = [factory() for factory in BALANCED_METAMODEL_FACTORIES]
    return standard, balanced


def build_tree_models():
    return {name: factory() for name, factory in TREE_MODEL_FACTORIES}


def build_balanced_tree_models():
    return {name: factory() for name, factory in BALANCED_TREE_MODEL_FACTORIES}


def build_rule_models():
    return {name: factory() for name, factory in RULE_MODEL_FACTORIES}


def is_balanced_metamodel(model):
    return isinstance(model, BALANCED_METAMODEL_FACTORIES)


def result_prefix(config, dataset_name, split_index, dataset_size):
    return os.path.join(config.raw_dir, '%s_%s_%s' % (dataset_name, split_index, dataset_size))


def result_paths(config, dataset_name, split_index, dataset_size):
    prefix = result_prefix(config, dataset_name, split_index, dataset_size)
    return {
        'raw': prefix + '.csv',
        'meta': prefix + '_meta.csv',
        'zeros': prefix + '_zeros.csv',
    }


def shard_is_complete(config, dataset_name, split_index, dataset_size):
    paths = result_paths(config, dataset_name, split_index, dataset_size)
    return (
        os.path.exists(paths['zeros']) or
        (os.path.exists(paths['raw']) and os.path.exists(paths['meta']))
    )


def model_size(name, model):
    if name in RULE_MODEL_NAMES:
        return len(model.ruleset_)
    if name in SD_MODEL_NAMES:
        return model.get_nrestr()
    return n_leaves(model)


def fidelity_score(predicted, reference):
    return np.count_nonzero(predicted == reference) / len(reference)


def write_result(handle, model_name, gen_name, meta_name, sctrain, sctest, complexity, elapsed, fidelity, bactest):
    handle.write(
        model_name + ',%s,%s,%s,%s,%s,%s,%s,%s\n'
        % (gen_name, meta_name, sctrain, sctest, complexity, elapsed, fidelity, bactest)
    )


def write_meta(handle, key, value):
    handle.write('%s,%s\n' % (key, value))


def fit_score_classifier(model, Xfit, yfit, Xtrain, ytrain, Xtest, ytest):
    start = time.time()
    model.fit(Xfit, yfit)
    end = time.time()
    return {
        'elapsed': end - start,
        'train': model.score(Xtrain, ytrain),
        'test': model.score(Xtest, ytest),
        'bactest': balanced_accuracy_score(ytest, model.predict(Xtest)),
    }


def fit_score_sd_model(model, Xfit, yfit, Xtrain, ytrain, Xtest, ytest):
    start = time.time()
    model.fit(Xfit, yfit)
    end = time.time()
    return {
        'elapsed': end - start,
        'train': model.score(Xtrain, ytrain),
        'test': model.score(Xtest, ytest),
    }


def get_supervised_models(meta_model, tree_models, balanced_tree_models, rule_models, dtval, dtvalb, include_rules = True):
    if is_balanced_metamodel(meta_model):
        return list(balanced_tree_models.items()) + [('dtvalb', dtvalb)]

    models = list(tree_models.items()) + [('dtval', dtval)]
    if include_rules:
        models.extend(rule_models.items())
    return models


def get_vva_models(meta_model, tree_models, balanced_tree_models, rule_models, dtval, dtvalb, primcv, bicv):
    models = get_supervised_models(
        meta_model,
        tree_models,
        balanced_tree_models,
        rule_models,
        dtval,
        dtvalb,
    )
    if is_balanced_metamodel(meta_model):
        return models
    return models + [('primcv', primcv), ('bicv', bicv)]


def get_standard_sd_models(primcv, bicv):
    return [('primcv', primcv), ('bicv', bicv)]


@dataclass
class ExperimentState:
    X: np.ndarray
    y: np.ndarray
    Xtest: np.ndarray
    ytest: np.ndarray
    ydeftest: np.ndarray
    generators: list
    genrerx: object
    genvva: object
    standard_metamodels: list
    balanced_metamodels: list
    all_metamodels: list
    tree_models: dict
    balanced_tree_models: dict
    rule_models: dict
    dtval: object
    dtvalb: object
    dtvalold: object
    dtvalbold: object
    bicv: object
    primcv: object


def build_model_state():
    generators, genrerx, genvva = build_generators()
    standard_metamodels, balanced_metamodels = build_metamodel_groups()
    return {
        'generators': generators,
        'genrerx': genrerx,
        'genvva': genvva,
        'standard_metamodels': standard_metamodels,
        'balanced_metamodels': balanced_metamodels,
        'all_metamodels': standard_metamodels + balanced_metamodels,
        'tree_models': build_tree_models(),
        'balanced_tree_models': build_balanced_tree_models(),
        'rule_models': build_rule_models(),
        'dtval': DecisionTreeClassifier(),
        'dtvalb': DecisionTreeClassifier(class_weight = 'balanced'),
    }


def load_experiment_split(config, split_index, dataset_name, dataset_size):
    paths = result_paths(config, dataset_name, split_index, dataset_size)
    X, y = load_data(dataset_name)
    splitter = DataSplitter(seed = config.split_seed)
    splitter.fit(X, y)
    splitter.configure(config.nsets, dataset_size)
    X, y = splitter.get_train(split_index)
    if y.sum() == 0:
        open(paths['zeros'], 'a', encoding = 'utf-8').close()
        return None

    Xtest, ytest = splitter.get_test(split_index)
    variable_mask = X.max(axis = 0) != X.min(axis = 0)
    Xtest = Xtest[:, variable_mask]
    X = X[:, variable_mask]

    scaler = StandardScaler()
    scaler.fit(X)
    return {
        'X': scaler.transform(X),
        'y': y,
        'Xtest': scaler.transform(Xtest),
        'ytest': ytest,
    }


def write_default_classifier_metadata(filetme, y, ytest):
    default_prediction = 1 if y.mean() >= 0.5 else 0
    write_meta(filetme, 'testprec', ytest.mean() if default_prediction == 1 else 1 - ytest.mean())
    write_meta(filetme, 'trainprec', y.mean() if default_prediction == 1 else 1 - y.mean())
    return np.ones(len(ytest)) * default_prediction


def fit_reference_models(config, X, y, Xtest, ytest, tree_models, balanced_tree_models, rule_models, dtval, dtvalb, fileres):
    tiled_repeats = int(np.ceil(config.rules_sample_size / X.shape[0]))
    Xr = np.tile(X, [tiled_repeats, 1])
    yr = np.tile(y, tiled_repeats)

    for name, model in list(tree_models.items()) + list(balanced_tree_models.items()):
        score = fit_score_classifier(model, X, y, X, y, Xtest, ytest)
        write_result(
            fileres,
            name,
            'na',
            'na',
            score['train'],
            score['test'],
            model_size(name, model),
            score['elapsed'],
            'na',
            score['bactest'],
        )

    for name, model in rule_models.items():
        score = fit_score_classifier(model, Xr, yr, Xr, yr, Xtest, ytest)
        write_result(
            fileres,
            name,
            'na',
            'na',
            score['train'],
            score['test'],
            model_size(name, model),
            score['elapsed'],
            'na',
            score['bactest'],
        )

    par_vals = [2 ** number for number in [1, 2, 3, 4, 5, 6, 7]]
    parameters = {'max_leaf_nodes': par_vals}

    start = time.time()
    tmp = GridSearchCV(dtval, parameters, refit = False).fit(X, y).cv_results_
    tmp = opt_param(tmp, len(par_vals))
    dtval = DecisionTreeClassifier(max_leaf_nodes = par_vals[np.argmax(tmp)])
    dtval.fit(X, y)
    end = time.time()
    write_result(
        fileres,
        'dtval',
        'na',
        'na',
        dtval.score(X, y),
        dtval.score(Xtest, ytest),
        model_size('dtval', dtval),
        end - start,
        'na',
        balanced_accuracy_score(ytest, dtval.predict(Xtest)),
    )

    start = time.time()
    tmp = GridSearchCV(dtvalb, parameters, refit = False, scoring = 'balanced_accuracy').fit(X, y).cv_results_
    tmp = opt_param(tmp, len(par_vals))
    dtvalb = DecisionTreeClassifier(max_leaf_nodes = par_vals[np.argmax(tmp)], class_weight = 'balanced')
    dtvalb.fit(X, y)
    end = time.time()
    write_result(
        fileres,
        'dtvalb',
        'na',
        'na',
        dtvalb.score(X, y),
        dtvalb.score(Xtest, ytest),
        model_size('dtvalb', dtvalb),
        end - start,
        'na',
        balanced_accuracy_score(ytest, dtvalb.predict(Xtest)),
    )

    dtvalold = copy.deepcopy(dtval)
    dtvalbold = copy.deepcopy(dtvalb)
    dtval = DecisionTreeClassifier(max_leaf_nodes = max(n_leaves(dtval), 2))
    dtvalb = DecisionTreeClassifier(max_leaf_nodes = max(n_leaves(dtvalb), 2), class_weight = 'balanced')

    parsbi = get_bi_param(5, X.shape[1])
    start = time.time()
    tmp = GridSearchCV(BI(), {'depth': parsbi}, refit = False).fit(X, y).cv_results_
    tmp = opt_param(tmp, len(parsbi))
    bicv = BI(depth = parsbi[np.argmax(tmp)])
    bicv.fit(X, y)
    end = time.time()
    write_result(
        fileres,
        'bicv',
        'na',
        'na',
        bicv.score(X, y),
        bicv.score(Xtest, ytest),
        model_size('bicv', bicv),
        end - start,
        'na',
        'na',
    )
    bicv = BI(depth = bicv.get_nrestr())

    par_vals = [0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2]
    start = time.time()
    tmp = GridSearchCV(PRIM(), {'alpha': par_vals}, refit = False).fit(X, y).cv_results_
    tmp = opt_param(tmp, len(par_vals))
    primcv = PRIM(alpha = par_vals[np.argmax(tmp)])
    primcv.fit(X, y)
    end = time.time()
    write_result(
        fileres,
        'primcv',
        'na',
        'na',
        primcv.score(X, y),
        primcv.score(Xtest, ytest),
        model_size('primcv', primcv),
        end - start,
        'na',
        'na',
    )

    return {
        'dtval': dtval,
        'dtvalb': dtvalb,
        'dtvalold': dtvalold,
        'dtvalbold': dtvalbold,
        'bicv': bicv,
        'primcv': primcv,
    }


def fit_generators_and_metamodels(state, filetme):
    for generator in state.generators:
        start = time.time()
        generator.fit(state.X, state.y)
        end = time.time()
        write_meta(filetme, generator.my_name() + 'time', end - start)

    for meta_model in state.all_metamodels:
        start = time.time()
        meta_model.fit(state.X, state.y)
        end = time.time()
        write_meta(filetme, meta_model.my_name() + 'time', end - start)
        write_meta(filetme, meta_model.my_name() + 'acccv', meta_model.fit_score())

        ypredtest = meta_model.predict(state.Xtest)
        write_meta(filetme, meta_model.my_name() + 'fid', fidelity_score(ypredtest, state.ydeftest))
        write_meta(filetme, meta_model.my_name() + 'acc', accuracy_score(state.ytest, ypredtest))
        write_meta(filetme, meta_model.my_name() + 'bac', balanced_accuracy_score(state.ytest, ypredtest))

        for name, model in get_supervised_models(
            meta_model,
            state.tree_models,
            state.balanced_tree_models,
            state.rule_models,
            state.dtvalold,
            state.dtvalbold,
        ):
            write_meta(filetme, meta_model.my_name() + name + 'fid', fidelity_score(model.predict(state.Xtest), ypredtest))


def evaluate_rerx(state, fileres):
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
        ):
            score = fit_score_classifier(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
            write_result(
                fileres,
                name,
                'rerx',
                meta_model.my_name(),
                score['train'],
                score['test'],
                model_size(name, model),
                score['elapsed'],
                fidelity_score(model.predict(state.Xtest), ypredtest),
                score['bactest'],
            )

        if not is_balanced_metamodel(meta_model):
            ynew = meta_model.predict_proba(Xnew)
            for name, model in get_standard_sd_models(state.primcv, state.bicv):
                score = fit_score_sd_model(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
                write_result(
                    fileres,
                    name,
                    'rerx',
                    meta_model.my_name(),
                    score['train'],
                    score['test'],
                    model_size(name, model),
                    score['elapsed'],
                    'na',
                    'na',
                )


def evaluate_vva(config, state, fileres, filetme):
    for meta_model in state.all_metamodels:
        ntrain = int(np.ceil(state.X.shape[0] * 2 / 3))
        Xtrain = state.X[:ntrain, :].copy()
        Xval = state.X[ntrain:, :].copy()
        ytrain = state.y[:ntrain].copy()
        yval = state.y[ntrain:].copy()
        start = time.time()
        state.genvva.fit(Xtrain, meta_model)
        end = time.time()
        write_meta(filetme, meta_model.my_name() + 'vva', end - start)
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
            write_meta(filetme, name + meta_model.my_name() + 'vvaopt', end - start)
            write_meta(filetme, name + meta_model.my_name() + 'ropt', ropt)

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
            write_meta(filetme, name + meta_model.my_name() + 'vvagen', end - start)

            if name in SD_MODEL_NAMES:
                score = fit_score_sd_model(model, Xnew, meta_model.predict_proba(Xnew), state.X, state.y, state.Xtest, state.ytest)
                fidelity = 'na'
                bactest = 'na'
            else:
                score = fit_score_classifier(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
                fidelity = fidelity_score(model.predict(state.Xtest), ypredtest)
                bactest = score['bactest']

            write_result(
                fileres,
                name,
                'vva',
                meta_model.my_name(),
                score['train'],
                score['test'],
                model_size(name, model),
                score['elapsed'],
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

    for name, model in [('dt', state.tree_models['dt']), ('dtc', state.tree_models['dtc']), ('dtval', state.dtval)]:
        score = fit_score_classifier(model, Xnew, predicted_labels, state.X, state.y, state.Xtest, state.ytest)
        write_result(
            fileres,
            GENERATED_TREE_ALIASES[name],
            generator.my_name(),
            meta_model.my_name(),
            score['train'],
            score['test'],
            model_size(name, model),
            score['elapsed'],
            fidelity_score(model.predict(state.Xtest), ypredtest),
            score['bactest'],
        )

    Xnew = Xnew[:config.generated_sample_size - len(state.y), :]
    predicted_labels = predicted_labels[:config.generated_sample_size - len(state.y)]
    Xnew = np.concatenate([state.X, Xnew])
    predicted_labels = np.concatenate([state.y, predicted_labels])

    for name, model in [('dt', state.tree_models['dt']), ('dtc', state.tree_models['dtc']), ('dtval', state.dtval)]:
        score = fit_score_classifier(model, Xnew, predicted_labels, state.X, state.y, state.Xtest, state.ytest)
        write_result(
            fileres,
            name,
            generator.my_name(),
            meta_model.my_name(),
            score['train'],
            score['test'],
            model_size(name, model),
            score['elapsed'],
            fidelity_score(model.predict(state.Xtest), ypredtest),
            score['bactest'],
        )

    Xnew = Xnew[:config.rules_sample_size, :]
    predicted_labels = predicted_labels[:config.rules_sample_size]

    for name, model in state.rule_models.items():
        score = fit_score_classifier(model, Xnew, predicted_labels, state.X, state.y, state.Xtest, state.ytest)
        write_result(
            fileres,
            name,
            generator.my_name(),
            meta_model.my_name(),
            score['train'],
            score['test'],
            model_size(name, model),
            score['elapsed'],
            fidelity_score(model.predict(state.Xtest), ypredtest),
            score['bactest'],
        )

    ynew = meta_model.predict_proba(Xnew)
    for name, model in get_standard_sd_models(state.primcv, state.bicv):
        score = fit_score_sd_model(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
        write_result(
            fileres,
            name,
            generator.my_name(),
            meta_model.my_name(),
            score['train'],
            score['test'],
            model_size(name, model),
            score['elapsed'],
            'na',
            'na',
        )


def evaluate_balanced_generator(generator, Xgen, meta_model, state, config, fileres, filetme):
    ypredtest = meta_model.predict(state.Xtest)

    start = time.time()
    Xnew = Xgen[:config.generated_sample_size - len(state.y), :].copy()
    ynew = meta_model.predict(Xnew)
    Xnew = np.concatenate([state.X, Xnew])
    ynew = np.concatenate([state.y, ynew])
    end = time.time()
    write_meta(filetme, generator.my_name() + meta_model.my_name(), end - start)

    for name, model in list(state.balanced_tree_models.items()) + [('dtvalb', state.dtvalb)]:
        score = fit_score_classifier(model, Xnew, ynew, state.X, state.y, state.Xtest, state.ytest)
        write_result(
            fileres,
            name,
            generator.my_name(),
            meta_model.my_name(),
            score['train'],
            score['test'],
            model_size(name, model),
            score['elapsed'],
            fidelity_score(model.predict(state.Xtest), ypredtest),
            score['bactest'],
        )


def evaluate_sampled_generators(config, state, fileres, filetme):
    for generator in state.generators:
        start = time.time()
        Xgen = generator.sample(config.generated_sample_size)
        end = time.time()
        write_meta(filetme, generator.my_name() + 'gen', end - start)

        for meta_model in state.standard_metamodels:
            evaluate_standard_generator(generator, Xgen, meta_model, state, config, fileres, filetme)
        for meta_model in state.balanced_metamodels:
            evaluate_balanced_generator(generator, Xgen, meta_model, state, config, fileres, filetme)


def evaluate_ssl(config, state, fileres):
    Xtest, ytest, Xgen = get_new_test(state.Xtest, state.ytest, len(state.y), new_size = config.ssl_pool_size)
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
        ):
            score = fit_score_classifier(model, Xnew, ynew, state.X, state.y, Xtest, ytest)
            write_result(
                fileres,
                name,
                'ssl',
                meta_model.my_name(),
                score['train'],
                score['test'],
                model_size(name, model),
                score['elapsed'],
                fidelity_score(model.predict(Xtest), ypredtest),
                score['bactest'],
            )

        if not is_balanced_metamodel(meta_model):
            ynew = meta_model.predict_proba(Xnew)
            for name, model in get_standard_sd_models(state.primcv, state.bicv):
                score = fit_score_sd_model(model, Xnew, ynew, state.X, state.y, Xtest, ytest)
                write_result(
                    fileres,
                    name,
                    'ssl',
                    meta_model.my_name(),
                    score['train'],
                    score['test'],
                    model_size(name, model),
                    score['elapsed'],
                    'na',
                    'na',
                )


def experiment(config, split_index, dataset_name, dataset_size):
    if shard_is_complete(config, dataset_name, split_index, dataset_size):
        return 'skipped'

    paths = result_paths(config, dataset_name, split_index, dataset_size)
    dataset = load_experiment_split(config, split_index, dataset_name, dataset_size)
    if dataset is None:
        return 'zero-class'

    started_at = time.time()
    state_dict = build_model_state()
    fileres = open(paths['raw'], 'a', encoding = 'utf-8')
    filetme = open(paths['meta'], 'a', encoding = 'utf-8')

    # The run is organized in phases:
    # 1) prepare split data, 2) fit reference baselines, 3) fit generators/metamodels,
    # 4) evaluate each PRELIM transfer strategy, 5) record aggregate timing.
    ydeftest = write_default_classifier_metadata(filetme, dataset['y'], dataset['ytest'])
    references = fit_reference_models(
        config,
        dataset['X'],
        dataset['y'],
        dataset['Xtest'],
        dataset['ytest'],
        state_dict['tree_models'],
        state_dict['balanced_tree_models'],
        state_dict['rule_models'],
        state_dict['dtval'],
        state_dict['dtvalb'],
        fileres,
    )
    merged_state = dict(state_dict)
    merged_state.update(references)
    state = ExperimentState(
        X = dataset['X'],
        y = dataset['y'],
        Xtest = dataset['Xtest'],
        ytest = dataset['ytest'],
        ydeftest = ydeftest,
        **merged_state,
    )

    fit_generators_and_metamodels(state, filetme)
    evaluate_rerx(state, fileres)
    evaluate_vva(config, state, fileres, filetme)
    evaluate_sampled_generators(config, state, fileres, filetme)
    evaluate_ssl(config, state, fileres)

    # Only after all phases finish do we mark the shard as completed.
    fileres.close()
    write_meta(filetme, 'overall', time.time() - started_at)
    filetme.close()
    return 'completed'


def configure_logging(config):
    logger = logging.getLogger('error')
    logger.handlers.clear()
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler(config.log_path, encoding = 'utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def non_interrupting_experiment(config, dataset_name, dataset_size, split_index):
    logger = logging.getLogger('error')
    status = 'failed'
    stacktrace = None
    try:
        status = experiment(config, split_index, dataset_name, dataset_size)
    except Exception:
        logger.error(
            'Error occured in experiment with split=%s dataset=%s size=%s',
            split_index,
            dataset_name,
            dataset_size,
        )
        logger.error(traceback.format_exc())
        stacktrace = traceback.format_exc()

    return status, split_index, dataset_name, dataset_size, stacktrace


def iter_experiment_args(config):
    return product(config.datasets, config.dataset_sizes, config.split_indices)


def write_manifest(config, status, summary = None):
    manifest = config.to_manifest()
    manifest['status'] = status
    if summary is not None:
        manifest['summary'] = summary
    with open(config.manifest_path, 'w', encoding = 'utf-8') as handle:
        json.dump(manifest, handle, indent = 2, sort_keys = True)


def summarize_results(result_list):
    summary = {
        'completed': 0,
        'skipped': 0,
        'zero_class': 0,
        'failed': 0,
    }
    for status, _, _, _, _ in result_list:
        if status == 'completed':
            summary['completed'] += 1
        elif status == 'skipped':
            summary['skipped'] += 1
        elif status == 'zero-class':
            summary['zero_class'] += 1
        else:
            summary['failed'] += 1
    summary['total'] = len(result_list)
    return summary


def exp_parallel(config):
    result_list = Parallel(n_jobs = config.jobs, verbose = 100)(
        delayed(non_interrupting_experiment)(config, *args) for args in iter_experiment_args(config)
    )
    summary = summarize_results(result_list)
    print(json.dumps(summary, indent = 2, sort_keys = True))
    return result_list, summary


def parse_args():
    parser = argparse.ArgumentParser(description = 'Run PRELIM experiments with versioned outputs.')
    parser.add_argument('--run-id', default = None, help = 'Unique run identifier. Defaults to a UTC timestamp-based id.')
    parser.add_argument('--datasets', default = ','.join(DEFAULT_DATASET_NAMES), help = 'Comma-separated dataset names.')
    parser.add_argument('--sizes', default = ','.join(str(size) for size in DEFAULT_DATASET_SIZES), help = 'Comma-separated dataset sizes.')
    parser.add_argument('--nsets', type = int, default = 25, help = 'Number of train/test splits per dataset size.')
    parser.add_argument('--split-seed', type = int, default = 2020, help = 'Seed used by the data splitter.')
    parser.add_argument('--jobs', type = int, default = os.cpu_count() or 1, help = 'Parallel worker count.')
    parser.add_argument('--generated-sample-size', type = int, default = 100000, help = 'Synthetic sample size used for generator evaluation.')
    parser.add_argument('--rules-sample-size', type = int, default = 10000, help = 'Maximum sample size used for rule learners.')
    parser.add_argument('--ssl-pool-size', type = int, default = 10000, help = 'Maximum unlabeled pool size used in SSL evaluation.')
    parser.add_argument('--vva-grid', default = ','.join(str(value) for value in DEFAULT_VVA_GRID), help = 'Comma-separated VVA ratio grid.')
    parser.add_argument('--resume', action = 'store_true', help = 'Reuse an existing run directory and skip completed shards.')
    return parser.parse_args()


def build_config(args):
    run_id = args.run_id or default_run_id()
    return ExperimentConfig(
        run_id = run_id,
        datasets = parse_csv_list(args.datasets, str),
        dataset_sizes = parse_csv_list(args.sizes, int),
        nsets = args.nsets,
        split_seed = args.split_seed,
        generated_sample_size = args.generated_sample_size,
        rules_sample_size = args.rules_sample_size,
        ssl_pool_size = args.ssl_pool_size,
        vva_grid = parse_csv_list(args.vva_grid, float),
        jobs = args.jobs,
        resume = args.resume,
    )


def main():
    args = parse_args()
    config = build_config(args)
    ensure_run_layout(config)
    configure_logging(config)
    write_manifest(config, status = 'running')
    result_list, summary = exp_parallel(config)
    final_status = 'failed' if summary['failed'] else 'completed'
    write_manifest(config, status = final_status, summary = summary)
    return result_list


if __name__ == '__main__':
    main()
