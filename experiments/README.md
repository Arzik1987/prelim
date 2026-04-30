## Experiments Supporting PRELIM

This folder contains the code to reproduce the experiments from our manuscript.

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
