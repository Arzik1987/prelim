# Turbine learning-curve demo

Run from the repository root:

```powershell
python experiments/demo/run_demo.py --step 50 --max-train-size 1000
```

The script evaluates the direct models used by the main experiments, plus a demo-specific post-pruned tree, on the
`turbine` dataset for training sizes `50, 100, ..., 1000`, with ten
repetitions per size.  At a given size, the ten training sets are disjoint when possible;
the test set for each repetition is the complement of its training set.

Outputs are written to `experiments/demo/output/`:

- `results.csv`: one row per model, size, and repetition, written after every
  completed repetition;
- `manifest.json`: configuration and preprocessing details;
- `mean_results.csv`: means and standard deviations grouped by model and size.

The models match the experiment code:

- `rf`: `RandomForestClassifier` with a 5-fold search over
  `max_features=[2, "sqrt", None]` and `random_state=2020`;
- `dt`: `DecisionTreeClassifier(min_samples_split=10)`;
- `dtc`: `DecisionTreeClassifier(max_leaf_nodes=8)`;
- `dtval`: a decision tree with 5-fold selection from
  `max_leaf_nodes=[2, 4, 8, 16, 32, 64, 128]`.

The `naive` point at size zero is the largest class proportion in the full
loaded dataset.

Use --step and --max-train-size to change the learning-curve grid. The partitions are generated in memory and are not written to disk.

For a nonuniform grid, provide an explicit comma-separated list instead:

```powershell
python experiments/demo/run_demo.py --train-sizes 50,75,100,150,225,350,500,750,1000,1500,2250,3000
```

When `--train-sizes` is supplied, it overrides `--step` and `--max-train-size`.
With `--resume`, already completed sizes are skipped, so the list can include
both existing and new sizes.


- `dt_pruned`: a Gini CART tree with 5-fold selection of the cost-complexity
  pruning parameter (`ccp_alpha`). Candidate alphas are drawn from the
  training split's cost-complexity pruning paths.
