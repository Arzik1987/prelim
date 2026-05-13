from dataclasses import dataclass

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .registries import (
    build_balanced_tree_models,
    build_generators,
    build_metamodel_groups,
    build_rule_models,
    build_tree_models,
)


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


def build_model_state(
    build_generators_fn=build_generators,
    build_metamodel_groups_fn=build_metamodel_groups,
    build_tree_models_fn=build_tree_models,
    build_balanced_tree_models_fn=build_balanced_tree_models,
    build_rule_models_fn=build_rule_models,
):
    generators, genrerx, genvva = build_generators_fn()
    standard_metamodels, balanced_metamodels = build_metamodel_groups_fn()
    return {
        "generators": generators,
        "genrerx": genrerx,
        "genvva": genvva,
        "standard_metamodels": standard_metamodels,
        "balanced_metamodels": balanced_metamodels,
        "all_metamodels": standard_metamodels + balanced_metamodels,
        "tree_models": build_tree_models_fn(),
        "balanced_tree_models": build_balanced_tree_models_fn(),
        "rule_models": build_rule_models_fn(),
        "dtval": DecisionTreeClassifier(),
        "dtvalb": DecisionTreeClassifier(class_weight="balanced"),
    }
