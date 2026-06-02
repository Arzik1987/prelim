from importlib import import_module

from .adasyn import Gen_adasyn
from .base import BaseGenerator
from .bayesnet import Gen_bayesnet
from .copulagan import Gen_copulagan
from .ctgan import Gen_ctgan
from .dummy import Gen_dummy
from .forestdiffusion import Gen_forestdiffusion
from .gaussiancopula import Gen_gaussiancopula
from .gibbs import Gen_gibbs
from .gmm import Gen_classgmm, Gen_gmm, Gen_gmmbic, Gen_gmmbical
from .kde import Gen_classkde, Gen_kdeb, Gen_kdebw, Gen_kdebwhl, Gen_kdebwm
from .munge import Gen_munge
from .noise import Gen_noise
from .part import Gen_part
from .perfect import Gen_perfect
from .rand import Gen_lhs, Gen_randn, Gen_randu
from .rerx import Gen_rerx
from .rfdens import Gen_rfdens
from .rose import Gen_rose
from .smote import Gen_smote
from .tabgan import Gen_tabgan
from .treedens import Gen_treedens
from .tvae import Gen_tvae
from .vinecopula import Gen_vinecopula
from .vva import Gen_vva as Gen_vva_legacy
from .vva_p import Gen_vva as Gen_vva_proba
from .registry import (
    EXPERIMENT_GENERATOR_NAMES,
    GENERATOR_SPECS,
    PUBLIC_GENERATOR_NAMES,
    build_generator,
    get_generator_class,
    make_generator_factory,
)


_LAZY_EXPORTS = {
    "Gen_binarydiffusion": (".binarydiffusion", "Gen_binarydiffusion"),
    "Gen_great": (".great", "Gen_great"),
    "Gen_tabddpm": (".tabddpm", "Gen_tabddpm"),
    "Gen_tabsyn": (".tabsyn", "Gen_tabsyn"),
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
    "Gen_copulagan",
    "Gen_ctgan",
    "Gen_dummy",
    "Gen_forestdiffusion",
    "Gen_gaussiancopula",
    "Gen_gibbs",
    "Gen_classgmm",
    "Gen_classkde",
    "Gen_great",
    "Gen_gmm",
    "Gen_gmmbic",
    "Gen_gmmbical",
    "Gen_kdeb",
    "Gen_kdebw",
    "Gen_kdebwhl",
    "Gen_kdebwm",
    "Gen_munge",
    "Gen_noise",
    "Gen_part",
    "Gen_perfect",
    "Gen_lhs",
    "Gen_randn",
    "Gen_randu",
    "Gen_rerx",
    "Gen_rfdens",
    "Gen_rose",
    "Gen_smote",
    "Gen_tabgan",
    "Gen_tabddpm",
    "Gen_tabsyn",
    "Gen_treedens",
    "Gen_tvae",
    "Gen_vinecopula",
    "Gen_vva_legacy",
    "Gen_vva_proba",
    "EXPERIMENT_GENERATOR_NAMES",
    "GENERATOR_SPECS",
    "PUBLIC_GENERATOR_NAMES",
    "build_generator",
    "get_generator_class",
    "make_generator_factory",
]
