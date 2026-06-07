## PRELIM

`prelim` is a Python package for improving decision-tree-like interpretable models trained on small datasets. It uses a stronger mediator model and a generated transfer set to give tree-based models, rules, and related simple classifiers better training data than the raw small sample alone. The example below shows this for a scikit-learn decision tree, and this repository also contains `experiments/`, the code used to reproduce the manuscript results.

### Installation

This repository contains two pieces:

- `prelim`: the installable Python package under `src/prelim`.
- `experiments/`: repo-local manuscript reproduction code with additional dependencies.

For local development with the `prelim` package, create a virtual environment and install `prelim` in editable mode.

With [uv](https://docs.astral.sh/uv/):
```
uv venv
source .venv/bin/activate
uv pip install -e .
```

With the standard library [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/):
```
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

To install directly from GitHub instead of a local checkout:
```
python -m pip install git+https://github.com/Arzik1987/prelim
```

The base package install keeps only the core dependencies needed by `prelim` and the lightweight built-in generators. Some generator backends are intentionally optional because they add heavy dependencies. To install the package together with those optional generator backends, use:
```
python -m pip install -e .[optional-generators]
```

To run the manuscript reproduction code, install the package first and then install the experiment-only requirements:
```
uv pip install -e .
uv pip install -r experiments/requirements.txt
```

Then run experiment entry points from the repository root:
```
PYTHONPATH=src python experiments/experiments.py
PYTHONPATH=src python experiments/read_results.py
```

### Testing the package contents
Call `pytest` from the project root after installing the package locally:
```
pytest
```

For source-level checks without installing, many tests can also be run with:
```
PYTHONPATH=src pytest
```

To remove generated local artifacts such as `__pycache__`, `.pytest_cache`, build directories, and egg-info metadata, run:
```
python scripts/clean_artifacts.py
```

### Basic Usage

`prelim` trains an interpretable model through a stronger mediator model:

1. Fit or reuse a mediator model on the small training set.
2. Generate extra feature rows with a transfer-set generator such as `kde`.
3. Label those generated rows with the mediator.
4. Fit the target interpretable model on the generated data plus the original data.

In the example below, the mediator is a random forest and the target is a small scikit-learn decision tree.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from prelim.prelim import prelim

# Generate synthetic dataset
n_samples = 50
covariance_matrix = [[1, 0], [0, 1]]

X_class0 = np.random.multivariate_normal([0, 0], covariance_matrix, n_samples)
X_class1 = np.random.multivariate_normal([1, 1], covariance_matrix, n_samples)
X = np.vstack((X_class0, X_class1))
y = np.hstack((np.zeros(n_samples), np.ones(n_samples))).astype(int)

mediator = RandomForestClassifier()
student_tree = DecisionTreeClassifier(max_leaf_nodes=8)

# Train using Prelim
wb_model = prelim(
    X,
    y,
    mediator,
    student_tree,
    gen_name='kde',
    new_size=2000,
    proba=False,
    verbose=True
) 
```

### Small Reproducible Demonstration

The script below compares PRELIM against the same scikit-learn decision tree fitted directly on the small training data. It repeats a synthetic two-class problem across several sample sizes, saves the plot, and writes the summary numbers used here.

```bash
PYTHONPATH=src python examples/readme_small_experiment.py
```

![PRELIM vs. direct decision-tree fitting](docs/assets/readme-small-experiment.png)

The plot shows mean test accuracy over 20 seeded repetitions. Error bars are standard errors. The PRELIM curve stays above the direct decision-tree baseline across all sample sizes shown, with the biggest gain at the smallest training set. That happens because PRELIM augments the limited training data with mediator-labeled synthetic samples before fitting the final tree.

Summary from the generated run:

| Training examples per class | PRELIM mean accuracy | Baseline mean accuracy | Mean improvement |
| ---: | ---: | ---: | ---: |
| 25 | 0.717 | 0.691 | +0.026 |
| 50 | 0.730 | 0.715 | +0.015 |
| 100 | 0.735 | 0.720 | +0.015 |
| 200 | 0.744 | 0.728 | +0.016 |
| 400 | 0.744 | 0.737 | +0.007 |

Across all runs, PRELIM averaged `0.734` accuracy versus `0.718` for the direct decision-tree baseline, an average improvement of `+0.016`.

### Real-Dataset Demonstration

The repository also includes a second README-scale demonstration on the real `gt` dataset. Unlike the synthetic toy example, this script follows the experiment protocol much more closely:

- tuned `rf` mediator from the experiment code
- `DecisionTreeClassifier(max_leaf_nodes=8)` student tree
- `kde` generator
- training sizes of `25, 50, 100, 200, 400` total rows
- `20` deterministic split positions from the experiment-style splitter
- z-score scaling fitted on each small training split
- large synthetic transfer set, matching the experiment setup

For each split and size, the script takes a contiguous small training window from the experiment splitter, evaluates on the complementary held-out data, saves a plot, and writes a JSON summary.

```bash
PYTHONPATH=src python examples/readme_gt_experiment.py
```

![PRELIM vs. direct decision-tree fitting on gt](docs/assets/readme-gt-experiment.png)

Summary from the generated run:

| Training set size | PRELIM mean accuracy | Baseline mean accuracy | Mean improvement |
| ---: | ---: | ---: | ---: |
| 25 | 0.696 | 0.678 | +0.018 |
| 50 | 0.726 | 0.718 | +0.008 |
| 100 | 0.775 | 0.755 | +0.020 |
| 200 | 0.790 | 0.775 | +0.015 |
| 400 | 0.798 | 0.793 | +0.005 |

Across all runs, PRELIM averaged `0.757` accuracy versus `0.744` for the direct decision-tree baseline, an average improvement of `+0.013`.

### Reproducing the Experiments
See respective description in the subdirectory `experiments`.
