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
adasyn
dummy
gmm
gmmal
class_gmm
class_kde
kde
munge
norm
smote
unif
gaussiancopula
copulagan
ctgan
treedens
tvae
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
- `cmm`
- `forestdiffusion`
- `gibbs`
- `great`
- `lhs`
- `noise`
- `perfect`
- `rose`
- `tabgan`
- `tabsyn`
- `vinecopula`
- `vva_proba`

The omitted generators are not all excluded for the same reason:
- some are not used by default because they are very similar to generators already included in the paper sweep
- others are excluded because they are too computationally expensive for the paper-scale runs or scale poorly across the full dataset and split grid

Notes:
- the experiment runner wires `vva` to `src/prelim/generators/vva.py`, the original VVA implementation used by the experiments; `vva_proba` points to `src/prelim/generators/vva_p.py` and is not used here
- `cmmpart` is part of the default experiment sweep and requires `python-weka-wrapper3` plus a working Java installation available through `java` or `JAVA_HOME`
- `cmmpart` starts a Weka JVM through `python-weka-wrapper3`; in mixed parallel sweeps this can be less robust than the pure-Python generators, so if Java or Weka-related worker errors appear, run `cmmpart` separately
- `tabddpm` is implemented in the experiment registry, but it is only instantiated when `TABDDPM_REPO_PATH` is present

The CLI entry points remain:
- `experiments.py`
- `read_results.py`

### Environment
The paper reproduction commands below were inspected against the project virtual environment.

Core runtime:
- `python==3.12.7`
- `prelim @ 10c69fcae7cbf9074c8e59099e38a15c3e8e87e2`

Experiment-relevant package versions in that environment:
- `joblib==1.5.3`
- `numpy==2.4.6`
- `pandas==2.3.3`
- `scikit-learn==1.8.0`
- `scipy==1.17.1`
- `seaborn==0.13.2`
- `matplotlib==3.10.9`
- `statsmodels==0.14.6`
- `lightgbm==4.6.0`
- `xgboost==3.2.0`
- `imbalanced-learn==0.14.1`
- `wittgenstein==0.3.5`
- `imodels==2.0.4`
- `python_weka_wrapper3==0.3.3`
- `liac-arff==2.5.0`
- `odfpy==1.4.1`
- `xlrd==2.0.2`
- `pgmpy==1.1.2`
- `sdv==1.36.2`
- `ctgan==0.12.1`
- `tabgan==3.2.0`
- `be_great==0.0.14`
- `ForestDiffusion==1.0.6`

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
- to reproduce the results from the paper, run the full experiment sweep with
```powershell
python experiments.py `
  --run-id paper-repro-all-in-one `
  --sizes 100,400 `
  --generators adasyn,dummy,gmm,gmmal,kde,kdem,munge,norm,smote,unif,class_gmm,gaussiancopula,copulagan,ctgan,treedens,tvae,class_kde,cmmpart,kdeb `
  --standard-metamodels rf,xgb `
  --balanced-metamodels rf,xgb `
  --tree-models dt,dtc,dtval `
  --balanced-tree-models dtb,dtcb,dtvalb `
  --rule-models irep,grl `
  --sd-models primcv,bicv `
  --jobs 8
```
- if `cmmpart` causes Java or Weka-related failures in that mixed run, use this alternative split execution:
```powershell
python experiments.py `
  --run-id paper-repro-all-in-one-main `
  --sizes 100,400 `
  --generators adasyn,dummy,gmm,gmmal,kde,kdem,munge,norm,smote,unif,class_gmm,gaussiancopula,copulagan,ctgan,treedens,tvae,class_kde,kdeb `
  --standard-metamodels rf,xgb `
  --balanced-metamodels rf,xgb `
  --tree-models dt,dtc,dtval `
  --balanced-tree-models dtb,dtcb,dtvalb `
  --rule-models irep,grl `
  --sd-models primcv,bicv `
  --jobs 8

python experiments.py `
  --run-id paper-repro-all-in-one-cmmpart `
  --sizes 100,400 `
  --generators cmmpart `
  --standard-metamodels rf,xgb `
  --balanced-metamodels rf,xgb `
  --tree-models dt,dtc,dtval `
  --balanced-tree-models dtb,dtcb,dtvalb `
  --rule-models irep,grl `
  --sd-models primcv,bicv `
  --jobs 1
```
- if you use the split execution above, merge the `cmmpart` raw results into the main run before calling `read_results.py`; post-processing expects one combined run directory
- each execution now creates a versioned run directory under `experiments/registry/runs/<run-id>/`
  containing:
  - `raw/`: per-shard CSV outputs
  - `derived/`: post-processed tables
  - `figures/`: generated figures
  - `manifest.json`: run configuration, git revision, and run status
- to reproduce the paper figures and derived tables from that one-shot run, post-process with
```powershell
python read_results.py `
  --run-id paper-repro-all-in-one `
  --figure-height 5.1
```
- optional rerun controls:
```powershell
python experiments.py `
  --run-id paper-repro-all-in-one `
  --resume

python read_results.py `
  --run-id paper-repro-all-in-one `
  --figure-height 5.1 `
  --skip-figures

python read_results.py `
  --run-id paper-repro-all-in-one `
  --figure-height 5.1 `
  --reuse-res
```

Due to certain randomness, the resulting numbers might deviate slightly from those reported in the article.
