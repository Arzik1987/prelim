# Prevent numpy multithreading: https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import os
os.environ.update(
    OMP_NUM_THREADS = '1',
    OPENBLAS_NUM_THREADS = '1',
    NUMEXPR_NUM_THREADS = '1',
    MKL_NUM_THREADS = '1',
)

import argparse
import json
import logging
import time
import traceback
from dataclasses import dataclass

import numpy as np
import wittgenstein as lw
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

from data.split import load_experiment_split as prepare_experiment_split
from data.split import write_default_classifier_metadata
from data.loader import load_data
from evaluation.helpers import get_new_test
from evaluation.phases import (
    evaluate_rerx,
    evaluate_sampled_generators,
    evaluate_ssl,
    evaluate_vva,
    fit_generators_and_metamodels,
    fit_reference_models,
)
from results.artifacts import (
    iter_experiment_args,
    result_paths,
    shard_is_complete,
    summarize_results,
    write_manifest,
    write_meta,
)
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
from metamodels.lgbm import Meta_lgbm
from metamodels.lgbmb import Meta_lgbm_bal
from metamodels.xgb import Meta_xgb
from metamodels.xgbb import Meta_xgb_bal
from prelim.generators.adasyn import Gen_adasyn
from prelim.generators.ctgan import Gen_ctgan
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
from prelim.generators.tabgan import Gen_tabgan
from prelim.generators.vva import Gen_vva


GENERATOR_FACTORIES = (
    Gen_gmmbic,
    Gen_kdebw,
    Gen_munge,
    Gen_ctgan,
    Gen_randu,
    Gen_randn,
    Gen_dummy,
    Gen_gmmbical,
    Gen_smote,
    Gen_adasyn,
    Gen_tabgan,
    Gen_rfdens,
    Gen_kdebwm,
    Gen_kdeb,
)

STANDARD_METAMODEL_FACTORIES = (
    Meta_rf,
    Meta_lgbm,
    Meta_xgb,
)

BALANCED_METAMODEL_FACTORIES = (
    Meta_rf_bal,
    Meta_lgbm_bal,
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
    return prepare_experiment_split(
        config,
        split_index,
        dataset_name,
        dataset_size,
        data_loader=load_data,
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

    fit_generators_and_metamodels(state, filetme, is_balanced_metamodel)
    evaluate_rerx(state, fileres, is_balanced_metamodel)
    evaluate_vva(config, state, fileres, filetme, is_balanced_metamodel)
    evaluate_sampled_generators(config, state, fileres, filetme)
    evaluate_ssl(config, state, fileres, is_balanced_metamodel, get_new_test)

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
