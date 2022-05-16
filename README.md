## PRELIM &mdash; **P**edagogical **R**ule **E**xtraction to **L**earn **I**nterpretable **M**odels

`prelim` is the python module that allows one to learn better comprehensible rule-based models (e.g., decision trees, classification rules, subgroups) from small datasets. Besides the distribution of `prelim`, this folder also contains the subdirectory `experiments` with the code to reproduce the experiments from the manuscript.

The project has emerged as a continuation of another [project](https://github.com/bobboman1000/gr_prim) that considers fewer generators.

### Installation

Use the following commands to set up the environment and install the module `prelim`.

With [Anaconda](https://www.anaconda.com/products/distribution):
```
conda create -n yourenv pip
conda activate yourenv
pip install git+https://github.com/Arzik1987/prelim
```

With [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
```
python3 -m venv yourenv
source yourenv/bin/activate
pip install git+https://github.com/Arzik1987/prelim
```

### Testing the package contents
Call <code>pytest</code> in the command line from the project root directory to run the generator tests.

### Exemplary Usage

```
import numpy as np

# generating a small synthetic dataset (training)
npt = 50
cov = [[1, 0], [0, 1]]
X = np.vstack((np.random.multivariate_normal([0, 0], cov, npt), np.random.multivariate_normal([1, 1], cov, npt)))
y = np.hstack((np.zeros(npt), np.ones(npt))).astype(int)

# generating a large synthetic dataset (testing)
Xtest = np.vstack((np.random.multivariate_normal([0, 0], cov, 100*npt), np.random.multivariate_normal([1, 1], cov, 100*npt)))
ytest = np.hstack((np.zeros(100*npt), np.ones(100*npt))).astype(int)

# import a comprehensible rule-based model learner (a decision tree)
# and an auxiliary model (random forest)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# learn a rule-based model with prelim method
from prelim.prelim import prelim
wb_model = prelim(X, y, RandomForestClassifier(),\
DecisionTreeClassifier(max_leaf_nodes = 8),\
gen_name = 'kde', new_size = 2000, proba = False, verbose = True)

# compare the learned model's accuracy to the baseline
wb_model_baseline = DecisionTreeClassifier(max_leaf_nodes = 8).fit(X, y)
print('prelim_score = %s' % wb_model.score(Xtest, ytest))
print('baseline_score = %s' % wb_model_baseline.score(Xtest, ytest))
```


### Reproducing the Experiments
See respective description in the subdirectory `experiments`.
