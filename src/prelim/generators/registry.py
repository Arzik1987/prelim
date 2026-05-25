from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class GeneratorSpec:
    module_name: str
    class_name: str


GENERATOR_SPECS = {
    "adasyn": GeneratorSpec(".adasyn", "Gen_adasyn"),
    "bayesnet": GeneratorSpec(".bayesnet", "Gen_bayesnet"),
    "binarydiffusion": GeneratorSpec(".binarydiffusion", "Gen_binarydiffusion"),
    "class_gmm": GeneratorSpec(".gmm", "Gen_classgmm"),
    "class_kde": GeneratorSpec(".kde", "Gen_classkde"),
    "cmm": GeneratorSpec(".rfdens", "Gen_rfdens"),
    "cmmpart": GeneratorSpec(".part", "Gen_part"),
    "copulagan": GeneratorSpec(".copulagan", "Gen_copulagan"),
    "ctgan": GeneratorSpec(".ctgan", "Gen_ctgan"),
    "dummy": GeneratorSpec(".dummy", "Gen_dummy"),
    "forestdiffusion": GeneratorSpec(".forestdiffusion", "Gen_forestdiffusion"),
    "gaussiancopula": GeneratorSpec(".gaussiancopula", "Gen_gaussiancopula"),
    "gmm": GeneratorSpec(".gmm", "Gen_gmmbic"),
    "gmmal": GeneratorSpec(".gmm", "Gen_gmmbical"),
    "great": GeneratorSpec(".great", "Gen_great"),
    "kde": GeneratorSpec(".kde", "Gen_kdebw"),
    "kdeb": GeneratorSpec(".kde", "Gen_kdeb"),
    "kdem": GeneratorSpec(".kde", "Gen_kdebwm"),
    "lhs": GeneratorSpec(".rand", "Gen_lhs"),
    "munge": GeneratorSpec(".munge", "Gen_munge"),
    "noise": GeneratorSpec(".noise", "Gen_noise"),
    "norm": GeneratorSpec(".rand", "Gen_randn"),
    "perfect": GeneratorSpec(".perfect", "Gen_perfect"),
    "rerx": GeneratorSpec(".rerx", "Gen_rerx"),
    "rose": GeneratorSpec(".rose", "Gen_rose"),
    "smote": GeneratorSpec(".smote", "Gen_smote"),
    "tabddpm": GeneratorSpec(".tabddpm", "Gen_tabddpm"),
    "tabgan": GeneratorSpec(".tabgan", "Gen_tabgan"),
    "tabsyn": GeneratorSpec(".tabsyn", "Gen_tabsyn"),
    "treedens": GeneratorSpec(".treedens", "Gen_treedens"),
    "tvae": GeneratorSpec(".tvae", "Gen_tvae"),
    "unif": GeneratorSpec(".rand", "Gen_randu"),
    "vinecopula": GeneratorSpec(".vinecopula", "Gen_vinecopula"),
    "vva": GeneratorSpec(".vva_p", "Gen_vva"),
    "vva_legacy": GeneratorSpec(".vva", "Gen_vva"),
}

EXPERIMENT_GENERATOR_NAMES = (
    "gmm",
    "class_gmm",
    "class_kde",
    "kde",
    "munge",
    "gaussiancopula",
    "copulagan",
    "ctgan",
    "lhs",
    "unif",
    "norm",
    "noise",
    "treedens",
    "dummy",
    "gmmal",
    "perfect",
    "rose",
    "smote",
    "adasyn",
    "tabgan",
    "tvae",
    "cmm",
    "cmmpart",
    "kdem",
    "kdeb",
)

PUBLIC_GENERATOR_NAMES = tuple(
    name for name in GENERATOR_SPECS if name not in {"noise", "perfect", "vva_legacy"}
)


def get_generator_class(gen_name):
    try:
        spec = GENERATOR_SPECS[gen_name]
    except KeyError as exc:
        valid_names = ", ".join(sorted(GENERATOR_SPECS))
        raise ValueError(f"Unknown generator key '{gen_name}'. Expected one of: {valid_names}") from exc

    module = import_module(spec.module_name, "prelim.generators")
    return getattr(module, spec.class_name)


def make_generator_factory(gen_name, **kwargs):
    def factory():
        return get_generator_class(gen_name)(**kwargs)

    factory.__name__ = f"build_{gen_name}"
    return factory


def build_generator(gen_name, seed=2020):
    if gen_name not in PUBLIC_GENERATOR_NAMES:
        valid_names = ", ".join(sorted(PUBLIC_GENERATOR_NAMES))
        raise ValueError(f"Unknown gen_name '{gen_name}'. Expected one of: {valid_names}")
    return get_generator_class(gen_name)(seed=seed)
