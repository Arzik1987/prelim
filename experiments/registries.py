from importlib.util import find_spec
import os

import wittgenstein as lw
from sklearn.tree import DecisionTreeClassifier

from .metamodels.rf import Meta_rf
from .metamodels.rfb import Meta_rf_bal
from .metamodels.lgbm import Meta_lgbm
from .metamodels.lgbmb import Meta_lgbm_bal
from .metamodels.xgb import Meta_xgb
from .metamodels.xgbb import Meta_xgb_bal
from prelim.generators import EXPERIMENT_GENERATOR_NAMES, get_generator_class, make_generator_factory


GENERATOR_FACTORIES = tuple(make_generator_factory(name) for name in EXPERIMENT_GENERATOR_NAMES)
Gen_binarydiffusion = get_generator_class("binarydiffusion")
Gen_rerx = get_generator_class("rerx")
Gen_tabddpm = get_generator_class("tabddpm")
Gen_vva = get_generator_class("vva_legacy")

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
    ("dt", lambda: DecisionTreeClassifier(min_samples_split=10)),
    # One could restrict depth instead. Results will be worse, but
    # ranking of generators will not generally change (still kde is the best).
    ("dtc", lambda: DecisionTreeClassifier(max_leaf_nodes=8)),
)

BALANCED_TREE_MODEL_FACTORIES = (
    ("dtb", lambda: DecisionTreeClassifier(min_samples_split=10, class_weight="balanced")),
    ("dtcb", lambda: DecisionTreeClassifier(max_leaf_nodes=8, class_weight="balanced")),
)

RULE_MODEL_FACTORIES = (
    ("ripper", lambda: lw.RIPPER(max_rules=8)),
    ("irep", lambda: lw.IREP(max_rules=8)),
)


def build_generators():
    factories = list(GENERATOR_FACTORIES)
    if find_spec("binary_diffusion_tabular") is not None:
        factories.append(Gen_binarydiffusion)
    if os.environ.get("TABDDPM_REPO_PATH"):
        factories.append(Gen_tabddpm)
    return [factory() for factory in factories], Gen_rerx(), Gen_vva()


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
