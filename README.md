## PRELIM &mdash; **P**edagogical **R**ule **E**xtraction to **L**earn **I**nterpretable **M**odels

`prelim` is the python module that allows one to learn better comprehensible rule-based models (e.g., decision trees, classification rules, subgroups) from small datasets. Besides the distribution of `prelim`, this folder also contains the subdirectory `experiments` with the code to reproduce the experiments from the manuscript.

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
To run tests, it is required to install the package locally.

### Exemplary Usage

`prelim` takes the target `rule_based_model` algorithm and uses a powerful `mediator` model coupled with a transfer set generator (here - `kde`) to fit a `wb_model` of the target model class.
The resulting `wb_model` is often more accurate than the `rule_based_model` would have been if fitted directly to the train data.

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

# Define models
mediator = RandomForestClassifier()
rule_based_model = DecisionTreeClassifier(max_leaf_nodes=8)

# Train using Prelim
wb_model = prelim(
    X,
    y,
    mediator,
    rule_based_model,
    gen_name='kde',
    new_size=2000,
    proba=False,
    verbose=True
) 
```

The following code shows how the quality of the `wb_model' learned with `prelim' exceeds the quality of the `baseline_model', 
which is the model of the same class but estimated directly from the train data without a mediator model and a transfer set.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from prelim.prelim import prelim

# Function to compare PRELIM to a baseline model
def small_exp(n_samples, cov=[[1, 0], [0, 1]], mean1=[0, 0], mean2=[1, 1]):
    # Generate training data
    X_train = np.vstack((
        np.random.multivariate_normal(mean1, cov, n_samples),
        np.random.multivariate_normal(mean2, cov, n_samples)
    ))
    y_train = np.hstack((np.zeros(n_samples), np.ones(n_samples))).astype(int)

    # Generate testing data
    X_test = np.vstack((
        np.random.multivariate_normal(mean1, cov, 100 * n_samples),
        np.random.multivariate_normal(mean2, cov, 100 * n_samples)
    ))
    y_test = np.hstack((np.zeros(100 * n_samples), np.ones(100 * n_samples))).astype(int)

    # Train using PRELIM
    wb_model = prelim(
        X_train,
        y_train,
        RandomForestClassifier(),
        DecisionTreeClassifier(max_leaf_nodes=8),
        gen_name='kde',
        new_size=100 * n_samples,
        proba=False,
        verbose=False
    )

    # Train baseline decision tree
    baseline_model = DecisionTreeClassifier(max_leaf_nodes=8).fit(X_train, y_train)

    return wb_model.score(X_test, y_test), baseline_model.score(X_test, y_test)

# Run experiments
repetitions = 30
sample_sizes = [25, 50, 100, 200, 400]
results = []
experiment_counter = 0

for _ in range(repetitions):
    for size in sample_sizes:
        experiment_counter += 1
        sys.stdout.write(f'\rExperiment {experiment_counter}/{repetitions * len(sample_sizes)}')
        acc_prelim, acc_baseline = small_exp(size)
        results.append(pd.DataFrame([[size, acc_prelim, acc_baseline]]))

# Aggregate results
results_df = pd.concat(results)
results_df.columns = ['Small data size', 'PRELIM', 'Baseline']

# Reshape for plotting
prelim_df = results_df[['Small data size', 'PRELIM']].copy()
prelim_df['method'] = 'PRELIM'
prelim_df.columns = ['Small data size', 'Accuracy', 'method']

baseline_df = results_df[['Small data size', 'Baseline']].copy()
baseline_df['method'] = 'Baseline'
baseline_df.columns = ['Small data size', 'Accuracy', 'method']

plot_df = pd.concat([prelim_df, baseline_df])

# Plot the results
sns.pointplot(x='Small data size', y='Accuracy', hue='method', data=plot_df)
```


### Reproducing the Experiments
See respective description in the subdirectory `experiments`.
