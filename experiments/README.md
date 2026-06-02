## Experiments Supporting PRELIM

This folder contains the code to reproduce the experiments from our manuscript.

Internal layout:
- `run.py`: CLI configuration, logging, shard execution, and parallel dispatch
- `registries.py`: experiment generator, metamodel, tree, and rule-model factory registries
- `state.py`: experiment state assembly
- `data/`: dataset loading and split preparation
- `evaluation/scoring.py`: shared scoring, fidelity, and model-size helpers
- `evaluation/baselines.py`: reference baseline fitting
- `evaluation/strategies.py`: RERX, VVA, sampled-generator, and SSL evaluation phases
- `evaluation/phases.py`: compatibility re-export layer for older imports
- `results/`: artifact paths and result post-processing
- `metamodels/`: experiment-local metamodel wrappers

### Generator Sets
Generator usage in this folder is defined by code, not by the public API surface:
- the default sampled-generator sweep uses `EXPERIMENT_GENERATOR_NAMES` from `src/prelim/generators/registry.py`
- `rerx` and original `vva` are instantiated separately in `registries.py` and evaluated through dedicated phases in `evaluation/strategies.py`
- `tabddpm` is only added to the experiment generator list when `TABDDPM_REPO_PATH` is set in the environment

Default sampled-generator sweep:
```text
gmm
class_gmm
class_kde
kde
munge
gaussiancopula
copulagan
ctgan
lhs
unif
norm
noise
treedens
dummy
gmmal
perfect
rose
smote
adasyn
tabgan
tvae
cmm
cmmpart
kdem
kdeb
```

Separate experiment paths that are used, but not part of the default sampled-generator sweep:
- `rerx`
- original `vva` (`vva` registry key)

Conditionally used in experiments:
- `tabddpm` via `TABDDPM_REPO_PATH`

Implemented generators that are not used by the experiment runner:
- `bayesnet`
- `binarydiffusion`
- `forestdiffusion`
- `gibbs`
- `great`
- `tabsyn`
- `vinecopula`
- `vva_proba`

Notes:
- the experiment runner wires `vva` to `src/prelim/generators/vva.py`, the original VVA implementation used by the experiments; `vva_proba` points to `src/prelim/generators/vva_p.py` and is not used here
- `cmmpart` is part of the default experiment sweep and requires `python-weka-wrapper3` plus a working Java installation available through `java` or `JAVA_HOME`
- `tabddpm` is implemented in the experiment registry, but it is only instantiated when `TABDDPM_REPO_PATH` is present

The CLI entry points remain:
- `experiments.py`
- `read_results.py`

### Usage
- install the `prelim` module following the instructions in the main directory
- clone this folder (`experiments`) to the desired place where you intend to run the experiments and navigate to it
- install additional requirements necessary to run the experiments by executing
```
pip install -r requirements.txt
```
- get datasets for the experiments by executing
```
python3 get_data.py
```
- run the experiments with
```
[nohup] python3 experiments.py
```
- each execution now creates a versioned run directory under `experiments/registry/runs/<run-id>/`
  containing:
  - `raw/`: per-shard CSV outputs
  - `derived/`: post-processed tables
  - `figures/`: generated figures
  - `manifest.json`: run configuration, git revision, and run status
- common rerun controls:
```
# `paper-main` below is only an example run id chosen by the user.
# It becomes the directory name under `experiments/registry/runs/`.
python3 experiments.py --run-id paper-main
python3 experiments.py --run-id paper-main --resume
python3 experiments.py --datasets clean2,gas --sizes 100 --nsets 5 --jobs 4
```
- after the end of the experiments, post-process the raw results to obtain figures from the paper and numbers from the tables in the paper:
```
python3 read_results.py
python3 read_results.py --run-id paper-main
```

Due to certain randomness, the resulting numbers might deviate slightly from those reported in the article.
