from importlib.util import find_spec
import os

from imodels import GreedyRuleListClassifier
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
GENERATOR_FACTORIES_BY_NAME = {name: make_generator_factory(name) for name in EXPERIMENT_GENERATOR_NAMES}
Gen_rerx = get_generator_class("rerx")
Gen_tabddpm = get_generator_class("tabddpm")
Gen_vva = get_generator_class("vva_legacy")

STANDARD_METAMODEL_FACTORIES = (
    Meta_rf,
    Meta_lgbm,
    Meta_xgb,
)
STANDARD_METAMODEL_FACTORIES_BY_NAME = {
    "rf": Meta_rf,
    "lgbm": Meta_lgbm,
    "xgb": Meta_xgb,
}

BALANCED_METAMODEL_FACTORIES = (
    Meta_rf_bal,
    Meta_lgbm_bal,
    Meta_xgb_bal,
)
BALANCED_METAMODEL_FACTORIES_BY_NAME = {
    "rf": Meta_rf_bal,
    "lgbm": Meta_lgbm_bal,
    "xgb": Meta_xgb_bal,
}

TREE_MODEL_FACTORIES = (
    ("dt", lambda: DecisionTreeClassifier(min_samples_split=10)),
    # One could restrict depth instead. Results will be worse, but
    # ranking of generators will not generally change (still kde is the best).
    ("dtc", lambda: DecisionTreeClassifier(max_leaf_nodes=8)),
)
TREE_MODEL_FACTORIES_BY_NAME = dict(TREE_MODEL_FACTORIES)

BALANCED_TREE_MODEL_FACTORIES = (
    ("dtb", lambda: DecisionTreeClassifier(min_samples_split=10, class_weight="balanced")),
    ("dtcb", lambda: DecisionTreeClassifier(max_leaf_nodes=8, class_weight="balanced")),
)
BALANCED_TREE_MODEL_FACTORIES_BY_NAME = dict(BALANCED_TREE_MODEL_FACTORIES)

RULE_MODEL_FACTORIES = (
    ("ripper", lambda: lw.RIPPER(max_rules=8, random_state=2020)),
    ("irep", lambda: lw.IREP(max_rules=8, random_state=2020)),
    ("grl", lambda: GreedyRuleListClassifier()),
)
RULE_MODEL_FACTORIES_BY_NAME = dict(RULE_MODEL_FACTORIES)


def build_generators(generator_names=None):
    if generator_names is None:
        factories = list(GENERATOR_FACTORIES)
    else:
        factories = [GENERATOR_FACTORIES_BY_NAME[name] for name in generator_names]
    if os.environ.get("TABDDPM_REPO_PATH"):
        factories.append(Gen_tabddpm)
    return [factory() for factory in factories], Gen_rerx(), Gen_vva()


def build_metamodel_groups():
    standard = [factory() for factory in STANDARD_METAMODEL_FACTORIES]
    balanced = [factory() for factory in BALANCED_METAMODEL_FACTORIES]
    return standard, balanced


def build_tree_models(model_names=None):
    factories = TREE_MODEL_FACTORIES if model_names is None else tuple(
        (name, TREE_MODEL_FACTORIES_BY_NAME[name]) for name in model_names
    )
    return {name: factory() for name, factory in factories}


def build_balanced_tree_models(model_names=None):
    factories = BALANCED_TREE_MODEL_FACTORIES if model_names is None else tuple(
        (name, BALANCED_TREE_MODEL_FACTORIES_BY_NAME[name]) for name in model_names
    )
    return {name: factory() for name, factory in factories}


def build_rule_models(model_names=None):
    factories = RULE_MODEL_FACTORIES if model_names is None else tuple(
        (name, RULE_MODEL_FACTORIES_BY_NAME[name]) for name in model_names
    )
    return {name: factory() for name, factory in factories}


def is_balanced_metamodel(model):
    return isinstance(model, BALANCED_METAMODEL_FACTORIES)
