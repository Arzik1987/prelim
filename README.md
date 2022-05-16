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
X = np.vstack((np.random.multivariate_normal([0, 0], cov, npt),\
               np.random.multivariate_normal([1, 1], cov, npt)))
y = np.hstack((np.zeros(npt), np.ones(npt))).astype(int)

# import a comprehensible rule-based model learner (a decision tree)
# and an auxiliary model (random forest)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# learn a rule-based model with prelim method
from prelim.prelim import prelim
wb_model = prelim(X, y, RandomForestClassifier(),\
DecisionTreeClassifier(max_leaf_nodes = 8),\
gen_name = 'kde', new_size = 2000, proba = False, verbose = True)
```

The following code extends this example and compares the quality of a comprehensible model learned with `prelim` with a 'baseline' comprehensible model learned directly from the train data.

```
import numpy as np
from sklearn.tree import DecisionTreeClassifier # a comprehensible rule-based model learner
from sklearn.ensemble import RandomForestClassifier # an auxiliary model
from prelim.prelim import prelim

# define a function to compare prelim to the baseline
def small_exp(npt, cov = [[1, 0], [0, 1]], m1 = [0, 0], m2 = [1, 1]):
    # generating synthetic data for training of the size npt
    X = np.vstack((np.random.multivariate_normal(m1, cov, npt),\
                   np.random.multivariate_normal(m2, cov, npt)))
    y = np.hstack((np.zeros(npt), np.ones(npt))).astype(int)
    
    # generating a large synthetic dataset (testing)
    Xtest = np.vstack((np.random.multivariate_normal(m1, cov, 100*npt),\
                       np.random.multivariate_normal(m2, cov, 100*npt)))
    ytest = np.hstack((np.zeros(100*npt), np.ones(100*npt))).astype(int)
    
    # learn a rule-based model with prelim method
    wb_model = prelim(X, y, RandomForestClassifier(),\
    DecisionTreeClassifier(max_leaf_nodes = 8),\
    gen_name = 'kde', new_size = 100*npt, proba = False, verbose = False)
    
    # learn a baseline model from small synthetic data
    wb_model_baseline = DecisionTreeClassifier(max_leaf_nodes = 8).fit(X, y)
    
    return wb_model.score(Xtest, ytest), wb_model_baseline.score(Xtest, ytest)

# conduct the experiments
reps = 30
res = []
k = 0
import sys
import pandas as pd

for i in range(0,reps):
    for j in [25,50,100,200,400]:
        k = k + 1
        sys.stdout.write('\r' + 'experiment' + ' ' + str(k) + '/' + str(reps*5))
        accp, accb = small_exp(j)
        res.append(pd.DataFrame([[j, accp, accb]]))

# postprocess the results
res = pd.concat(res)
res.columns = ('size', 'PRELIM', 'Baseline')
tmpres1 = res[['size', 'PRELIM']]
tmpres1['method'] = pd.Series(['PRELIM']).repeat(tmpres1.shape[0])
tmpres1.columns = ('Small data size', 'Accuracy', 'method')
tmpres2 = res[['size', 'Baseline']]
tmpres2['method'] = pd.Series(['Baseline']).repeat(tmpres2.shape[0])
tmpres2.columns = ('Small data size', 'Accuracy', 'method')
res = tmpres1.append(tmpres2)

# draw the plot
import seaborn as sns
sns.pointplot(x = 'Small data size', y = 'Accuracy', hue = 'method', data = res)
```


### Reproducing the Experiments
See respective description in the subdirectory `experiments`.
