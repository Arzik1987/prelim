from importlib import import_module

from .adasyn import Gen_adasyn
from .base import BaseGenerator
from .bayesnet import Gen_bayesnet
from .copulagan import Gen_copulagan
from .ctgan import Gen_ctgan
from .dummy import Gen_dummy
from .forestdiffusion import Gen_forestdiffusion
from .gaussiancopula import Gen_gaussiancopula
from .gmm import Gen_classgmm, Gen_gmm, Gen_gmmbic, Gen_gmmbical
from .kde import Gen_kdebw, Gen_kdebwhl
from .kdeb import Gen_kdeb
from .kdem import Gen_kdebwm
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


_LAZY_EXPORTS = {
    "Gen_binarydiffusion": (".binarydiffusion", "Gen_binarydiffusion"),
    "Gen_great": (".great", "Gen_great"),
    "Gen_tabddpm": (".tabddpm", "Gen_tabddpm"),
    "Gen_tabsyn": (".tabsyn", "Gen_tabsyn"),
}


def build_generator(gen_name, seed=2020):
    registry = {
        "adasyn": (".adasyn", "Gen_adasyn"),
        "bayesnet": (".bayesnet", "Gen_bayesnet"),
        "binarydiffusion": (".binarydiffusion", "Gen_binarydiffusion"),
        "class_gmm": (".gmm", "Gen_classgmm"),
        "cmm": (".rfdens", "Gen_rfdens"),
        "copulagan": (".copulagan", "Gen_copulagan"),
        "ctgan": (".ctgan", "Gen_ctgan"),
        "dummy": (".dummy", "Gen_dummy"),
        "forestdiffusion": (".forestdiffusion", "Gen_forestdiffusion"),
        "gaussiancopula": (".gaussiancopula", "Gen_gaussiancopula"),
        "great": (".great", "Gen_great"),
        "gmm": (".gmm", "Gen_gmmbic"),
        "gmmal": (".gmm", "Gen_gmmbical"),
        "kde": (".kde", "Gen_kdebw"),
        "kdeb": (".kdeb", "Gen_kdeb"),
        "kdem": (".kdem", "Gen_kdebwm"),
        "lhs": (".rand", "Gen_lhs"),
        "munge": (".munge", "Gen_munge"),
        "norm": (".rand", "Gen_randn"),
        "cmmpart": (".part", "Gen_part"),
        "rerx": (".rerx", "Gen_rerx"),
        "rose": (".rose", "Gen_rose"),
        "smote": (".smote", "Gen_smote"),
        "tabgan": (".tabgan", "Gen_tabgan"),
        "tabddpm": (".tabddpm", "Gen_tabddpm"),
        "tabsyn": (".tabsyn", "Gen_tabsyn"),
        "treedens": (".treedens", "Gen_treedens"),
        "tvae": (".tvae", "Gen_tvae"),
        "unif": (".rand", "Gen_randu"),
        "vinecopula": (".vinecopula", "Gen_vinecopula"),
        "vva": (".vva_p", "Gen_vva"),
    }

    try:
        module_name, class_name = registry[gen_name]
    except KeyError as exc:
        valid_names = ", ".join(sorted(registry))
        raise ValueError(f"Unknown gen_name '{gen_name}'. Expected one of: {valid_names}") from exc

    module = import_module(module_name, __name__)
    return getattr(module, class_name)(seed=seed)


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
    "Gen_classgmm",
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
    "build_generator",
]
