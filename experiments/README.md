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
- after the end of the experiments, post-process the raw results to obtain figures from the paper and numbers from the tables in the paper:
```
python3 read_results.py
```

Due to certain randomness, the resulting numbers might deviate slightly from those reported in the article.

