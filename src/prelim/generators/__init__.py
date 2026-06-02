from importlib import import_module

from .base import BaseGenerator
from .registry import (
    EXPERIMENT_GENERATOR_NAMES,
    GENERATOR_SPECS,
    PUBLIC_GENERATOR_NAMES,
    build_generator,
    get_generator_class,
    make_generator_factory,
)


_LAZY_EXPORTS = {
    "Gen_adasyn": (".adasyn", "Gen_adasyn"),
    "Gen_bayesnet": (".bayesnet", "Gen_bayesnet"),
    "Gen_binarydiffusion": (".binarydiffusion", "Gen_binarydiffusion"),
    "Gen_classgmm": (".gmm", "Gen_classgmm"),
    "Gen_classkde": (".kde", "Gen_classkde"),
    "Gen_copulagan": (".copulagan", "Gen_copulagan"),
    "Gen_ctgan": (".ctgan", "Gen_ctgan"),
    "Gen_dummy": (".dummy", "Gen_dummy"),
    "Gen_forestdiffusion": (".forestdiffusion", "Gen_forestdiffusion"),
    "Gen_gaussiancopula": (".gaussiancopula", "Gen_gaussiancopula"),
    "Gen_gibbs": (".gibbs", "Gen_gibbs"),
    "Gen_gmm": (".gmm", "Gen_gmm"),
    "Gen_gmmbic": (".gmm", "Gen_gmmbic"),
    "Gen_gmmbical": (".gmm", "Gen_gmmbical"),
    "Gen_great": (".great", "Gen_great"),
    "Gen_kdeb": (".kde", "Gen_kdeb"),
    "Gen_kdebw": (".kde", "Gen_kdebw"),
    "Gen_kdebwhl": (".kde", "Gen_kdebwhl"),
    "Gen_kdebwm": (".kde", "Gen_kdebwm"),
    "Gen_lhs": (".rand", "Gen_lhs"),
    "Gen_munge": (".munge", "Gen_munge"),
    "Gen_noise": (".noise", "Gen_noise"),
    "Gen_part": (".part", "Gen_part"),
    "Gen_perfect": (".perfect", "Gen_perfect"),
    "Gen_randn": (".rand", "Gen_randn"),
    "Gen_randu": (".rand", "Gen_randu"),
    "Gen_rerx": (".rerx", "Gen_rerx"),
    "Gen_rfdens": (".rfdens", "Gen_rfdens"),
    "Gen_rose": (".rose", "Gen_rose"),
    "Gen_smote": (".smote", "Gen_smote"),
    "Gen_tabddpm": (".tabddpm", "Gen_tabddpm"),
    "Gen_tabgan": (".tabgan", "Gen_tabgan"),
    "Gen_tabsyn": (".tabsyn", "Gen_tabsyn"),
    "Gen_treedens": (".treedens", "Gen_treedens"),
    "Gen_tvae": (".tvae", "Gen_tvae"),
    "Gen_vinecopula": (".vinecopula", "Gen_vinecopula"),
    "Gen_vva": (".vva", "Gen_vva"),
    "Gen_vva_proba": (".vva_p", "Gen_vva"),
}


def __getattr__(name):
    try:
        module_name, class_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    value = getattr(module, class_name)
    globals()[name] = value
    return value


__all__ = [
    "BaseGenerator",
    "Gen_adasyn",
    "Gen_bayesnet",
    "Gen_binarydiffusion",
    "Gen_classgmm",
    "Gen_classkde",
    "Gen_copulagan",
    "Gen_ctgan",
    "Gen_dummy",
    "Gen_forestdiffusion",
    "Gen_gaussiancopula",
    "Gen_gibbs",
    "Gen_gmm",
    "Gen_gmmbic",
    "Gen_gmmbical",
    "Gen_great",
    "Gen_kdeb",
    "Gen_kdebw",
    "Gen_kdebwhl",
    "Gen_kdebwm",
    "Gen_lhs",
    "Gen_munge",
    "Gen_noise",
    "Gen_part",
    "Gen_perfect",
    "Gen_randn",
    "Gen_randu",
    "Gen_rerx",
    "Gen_rfdens",
    "Gen_rose",
    "Gen_smote",
    "Gen_tabddpm",
    "Gen_tabgan",
    "Gen_tabsyn",
    "Gen_treedens",
    "Gen_tvae",
    "Gen_vinecopula",
    "Gen_vva",
    "Gen_vva_proba",
    "EXPERIMENT_GENERATOR_NAMES",
    "GENERATOR_SPECS",
    "PUBLIC_GENERATOR_NAMES",
    "build_generator",
    "get_generator_class",
    "make_generator_factory",
]
