# Prevent numpy multithreading: https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import os
import sys
from pathlib import Path
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

from joblib import Parallel, delayed

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "experiments"

from .data.preparation import load_experiment_split as prepare_experiment_split
from .data.preparation import write_default_classifier_metadata
from .data.loader import load_data
from .evaluation.helpers import get_new_test
from .evaluation.baselines import fit_reference_models
from .evaluation.strategies import (
    evaluate_rerx,
    evaluate_sampled_generators,
    evaluate_ssl,
    evaluate_vva,
    fit_generators_and_metamodels,
)
from .results.artifacts import (
    iter_experiment_args,
    result_paths,
    shard_is_complete,
    summarize_results,
    write_manifest,
    write_meta,
)
from .config import (
    DEFAULT_BALANCED_METAMODELS,
    DEFAULT_DATASET_NAMES,
    DEFAULT_DATASET_SIZES,
    DEFAULT_STANDARD_METAMODELS,
    DEFAULT_VVA_GRID,
    ExperimentConfig,
    default_run_id,
    ensure_run_layout,
    parse_csv_list,
)
from .registries import (
    BALANCED_METAMODEL_FACTORIES,
    BALANCED_METAMODEL_FACTORIES_BY_NAME,
    BALANCED_TREE_MODEL_FACTORIES,
    build_generators as build_registry_generators,
    GENERATOR_FACTORIES,
    RULE_MODEL_FACTORIES,
    STANDARD_METAMODEL_FACTORIES,
    STANDARD_METAMODEL_FACTORIES_BY_NAME,
    TREE_MODEL_FACTORIES,
    Gen_rerx,
    Gen_tabddpm,
    Gen_vva,
)
from .state import ExperimentState
from .state import build_model_state as _build_model_state
from prelim.generators.registry import EXPERIMENT_GENERATOR_NAMES


def build_generators(generator_names=None):
    return build_registry_generators(generator_names)


def _resolve_named_factories(selected_names, factories_by_name, kind):
    factories = []
    for name in selected_names:
        try:
            factories.append(factories_by_name[name])
        except KeyError as exc:
            valid_names = ", ".join(sorted(factories_by_name))
            raise ValueError(f"Unknown {kind} metamodel '{name}'. Expected one of: {valid_names}") from exc
    return tuple(factories)


def build_metamodel_groups(config=None):
    if config is None or config.standard_metamodels == DEFAULT_STANDARD_METAMODELS:
        standard_factories = STANDARD_METAMODEL_FACTORIES
    else:
        standard_factories = _resolve_named_factories(
            config.standard_metamodels,
            STANDARD_METAMODEL_FACTORIES_BY_NAME,
            "standard",
        )
    if config is None or config.balanced_metamodels == DEFAULT_BALANCED_METAMODELS:
        balanced_factories = BALANCED_METAMODEL_FACTORIES
    else:
        balanced_factories = _resolve_named_factories(
            config.balanced_metamodels,
            BALANCED_METAMODEL_FACTORIES_BY_NAME,
            "balanced",
        )
    standard = [factory() for factory in standard_factories]
    balanced = [factory() for factory in balanced_factories]
    return standard, balanced


def build_tree_models():
    return {name: factory() for name, factory in TREE_MODEL_FACTORIES}


def build_balanced_tree_models():
    return {name: factory() for name, factory in BALANCED_TREE_MODEL_FACTORIES}


def build_rule_models():
    return {name: factory() for name, factory in RULE_MODEL_FACTORIES}


def is_balanced_metamodel(model):
    return isinstance(model, BALANCED_METAMODEL_FACTORIES)


def build_model_state(config=None):
    return _build_model_state(
        build_generators_fn=lambda: build_generators(config.generator_names if config is not None else None),
        build_metamodel_groups_fn=lambda: build_metamodel_groups(config),
        build_tree_models_fn=build_tree_models,
        build_balanced_tree_models_fn=build_balanced_tree_models,
        build_rule_models_fn=build_rule_models,
    )


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
    state_dict = build_model_state(config)
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
    parser.add_argument('--nsets', type = int, default = 10, help = 'Number of train/test splits per dataset size.')
    parser.add_argument('--split-seed', type = int, default = 2020, help = 'Seed used by the data partitioner.')
    parser.add_argument('--jobs', type = int, default = os.cpu_count() or 1, help = 'Parallel worker count.')
    parser.add_argument('--generated-sample-size', type = int, default = 100000, help = 'Synthetic sample size used for generator evaluation.')
    parser.add_argument('--rules-sample-size', type = int, default = 10000, help = 'Maximum sample size used for rule learners.')
    parser.add_argument('--ssl-pool-size', type = int, default = 10000, help = 'Maximum unlabeled pool size used in SSL evaluation.')
    parser.add_argument('--vva-grid', default = ','.join(str(value) for value in DEFAULT_VVA_GRID), help = 'Comma-separated VVA ratio grid.')
    parser.add_argument('--generators', default = ','.join(EXPERIMENT_GENERATOR_NAMES), help = 'Comma-separated generator names to fit and evaluate.')
    parser.add_argument('--standard-metamodels', default = ','.join(DEFAULT_STANDARD_METAMODELS), help = 'Comma-separated standard metamodel names: rf,lgbm,xgb.')
    parser.add_argument('--balanced-metamodels', default = ','.join(DEFAULT_BALANCED_METAMODELS), help = 'Comma-separated balanced metamodel names: rf,lgbm,xgb.')
    parser.add_argument(
        '--include-generated-only-tree-models',
        action = 'store_true',
        help = 'Also evaluate dtp/dtcp/dtvalp models trained only on generated pseudo-labeled data.',
    )
    parser.add_argument('--resume', action = 'store_true', help = 'Reuse an existing run directory and skip completed shards.')
    return parser.parse_args()


def build_config(args):
    run_id = args.run_id or default_run_id()
    standard_metamodels = parse_csv_list(args.standard_metamodels, str)
    balanced_metamodels = parse_csv_list(args.balanced_metamodels, str)
    _resolve_named_factories(standard_metamodels, STANDARD_METAMODEL_FACTORIES_BY_NAME, "standard")
    _resolve_named_factories(balanced_metamodels, BALANCED_METAMODEL_FACTORIES_BY_NAME, "balanced")
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
        generator_names = parse_csv_list(args.generators, str),
        standard_metamodels = standard_metamodels,
        balanced_metamodels = balanced_metamodels,
        include_generated_only_tree_models = getattr(args, "include_generated_only_tree_models", False),
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
