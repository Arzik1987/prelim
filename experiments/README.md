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
The default experiment sweep uses `EXPERIMENT_GENERATOR_NAMES` from `src/prelim/generators/registry.py`:
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

Implemented generators that are not part of `EXPERIMENT_GENERATOR_NAMES`:
- `bayesnet`
- `binarydiffusion`
- `forestdiffusion`
- `great`
- `rerx`
- `tabddpm`
- `tabsyn`
- `vinecopula`
- `vva`

Notes:
- `rerx` and `vva` are implemented as separate evaluation paths, not as standard sweep generators.
- `bayesnet`, `binarydiffusion`, `forestdiffusion`, `great`, `tabddpm`, `tabsyn`, and `vinecopula` are implemented backends that are available through the generator API, but they are excluded from the default experiment sweep because fitting them is prohibitively slow in the current experiment setup.
- `cmmpart` is part of the default experiment sweep and requires `python-weka-wrapper3` plus a working Java installation available through `java` or `JAVA_HOME`.

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
