from importlib import import_module

from .adasyn import Gen_adasyn
from .base import BaseGenerator
from .dummy import Gen_dummy
from .gmm import Gen_gmm, Gen_gmmbic, Gen_gmmbical
from .kde import Gen_kdebw, Gen_kdebwhl
from .kdeb import Gen_kdeb
from .kdem import Gen_kdebwm
from .munge import Gen_munge
from .noise import Gen_noise
from .perfect import Gen_perfect
from .rand import Gen_randn, Gen_randu
from .rerx import Gen_rerx
from .rfdens import Gen_rfdens
from .smote import Gen_smote
from .tabgan import Gen_tabgan
from .vva import Gen_vva as Gen_vva_legacy
from .vva_p import Gen_vva as Gen_vva_proba


def build_generator(gen_name, seed=2020):
    registry = {
        "adasyn": (".adasyn", "Gen_adasyn"),
        "cmm": (".rfdens", "Gen_rfdens"),
        "dummy": (".dummy", "Gen_dummy"),
        "gmm": (".gmm", "Gen_gmmbic"),
        "gmmal": (".gmm", "Gen_gmmbical"),
        "kde": (".kde", "Gen_kdebw"),
        "kdeb": (".kdeb", "Gen_kdeb"),
        "kdem": (".kdem", "Gen_kdebwm"),
        "munge": (".munge", "Gen_munge"),
        "norm": (".rand", "Gen_randn"),
        "rerx": (".rerx", "Gen_rerx"),
        "smote": (".smote", "Gen_smote"),
        "tabgan": (".tabgan", "Gen_tabgan"),
        "unif": (".rand", "Gen_randu"),
        "vva": (".vva_p", "Gen_vva"),
    }

    try:
        module_name, class_name = registry[gen_name]
    except KeyError as exc:
        valid_names = ", ".join(sorted(registry))
        raise ValueError(f"Unknown gen_name '{gen_name}'. Expected one of: {valid_names}") from exc

    module = import_module(module_name, __name__)
    return getattr(module, class_name)(seed=seed)


__all__ = [
    "BaseGenerator",
    "Gen_adasyn",
    "Gen_dummy",
    "Gen_gmm",
    "Gen_gmmbic",
    "Gen_gmmbical",
    "Gen_kdeb",
    "Gen_kdebw",
    "Gen_kdebwhl",
    "Gen_kdebwm",
    "Gen_munge",
    "Gen_noise",
    "Gen_perfect",
    "Gen_randn",
    "Gen_randu",
    "Gen_rerx",
    "Gen_rfdens",
    "Gen_smote",
    "Gen_tabgan",
    "Gen_vva_legacy",
    "Gen_vva_proba",
    "build_generator",
]
