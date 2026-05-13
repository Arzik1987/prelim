from pathlib import Path
import sys


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "experiments"
    if __spec__ is not None and __spec__.parent != __package__:
        __spec__ = None

from . import run as _run
from .config import (
    DEFAULT_DATASET_NAMES,
    DEFAULT_DATASET_SIZES,
    DEFAULT_VVA_GRID,
    ExperimentConfig,
    default_run_id,
    ensure_run_layout,
    parse_csv_list,
)
from .results.artifacts import (
    iter_experiment_args,
    result_paths,
    shard_is_complete,
    summarize_results,
    write_manifest,
    write_meta,
)
from .run import (
    build_config,
    build_model_state,
    configure_logging,
    load_experiment_split,
    main,
    non_interrupting_experiment,
    parse_args,
)


GENERATOR_FACTORIES = _run.GENERATOR_FACTORIES
STANDARD_METAMODEL_FACTORIES = _run.STANDARD_METAMODEL_FACTORIES
BALANCED_METAMODEL_FACTORIES = _run.BALANCED_METAMODEL_FACTORIES
TREE_MODEL_FACTORIES = _run.TREE_MODEL_FACTORIES
BALANCED_TREE_MODEL_FACTORIES = _run.BALANCED_TREE_MODEL_FACTORIES
RULE_MODEL_FACTORIES = _run.RULE_MODEL_FACTORIES
Gen_binarydiffusion = _run.Gen_binarydiffusion
Gen_rerx = _run.Gen_rerx
Gen_tabddpm = _run.Gen_tabddpm
Gen_vva = _run.Gen_vva
load_data = _run.load_data
is_balanced_metamodel = _run.is_balanced_metamodel


def _sync_run_overrides():
    for name in (
        "GENERATOR_FACTORIES",
        "STANDARD_METAMODEL_FACTORIES",
        "BALANCED_METAMODEL_FACTORIES",
        "TREE_MODEL_FACTORIES",
        "BALANCED_TREE_MODEL_FACTORIES",
        "RULE_MODEL_FACTORIES",
        "Gen_binarydiffusion",
        "Gen_rerx",
        "Gen_tabddpm",
        "Gen_vva",
        "load_data",
        "is_balanced_metamodel",
    ):
        setattr(_run, name, globals()[name])


def build_generators():
    _sync_run_overrides()
    return _run.build_generators()


def build_metamodel_groups():
    _sync_run_overrides()
    return _run.build_metamodel_groups()


def build_tree_models():
    _sync_run_overrides()
    return _run.build_tree_models()


def build_balanced_tree_models():
    _sync_run_overrides()
    return _run.build_balanced_tree_models()


def build_rule_models():
    _sync_run_overrides()
    return _run.build_rule_models()


def experiment(config, split_index, dataset_name, dataset_size):
    _sync_run_overrides()
    return _run.experiment(config, split_index, dataset_name, dataset_size)


def exp_parallel(config):
    _sync_run_overrides()
    return _run.exp_parallel(config)


if __name__ == "__main__":
    main()
