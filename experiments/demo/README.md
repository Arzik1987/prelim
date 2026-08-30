# Turbine learning-curve demo

Run from the repository root:

```powershell
python experiments/demo/run_demo.py --train-sizes "50,75,100,6000,7000,8000,9000,10000" --threads 8
```

The runner evaluates five models on the requested number of splits for each train size (controlled by `--repetitions`, default 10). Each `(train_size, repetition)` experiment is an independent task and writes one atomic five-row CSV to `output/experiments/`. The worker queue is shared by all threads, and `--threads` limits the total number of concurrent worker threads (default: 8). The scikit-learn grid searches use `n_jobs=1` so they do not create another layer of demo workers.

Use `--resume` to skip valid task files already present. Use `--overwrite` to rerun all splits for requested train sizes; old excess splits for those sizes are removed. Unrequested train sizes are left untouched. Do not run two runner processes against the same output directory at once.

Aggregate the task files separately:

```powershell
python experiments/demo/aggregate_results.py
```

Aggregation writes `results.csv` and `mean_results.csv` to `output/`; these are the inputs used by `plot_learning_curves.py`. The run manifest is written to `output/manifest.json`.

For a regular grid, use `--step` and `--max-train-size`:

```powershell
python experiments/demo/run_demo.py --step 50 --max-train-size 1000 --threads 8
```

The partitions are generated deterministically in memory and are not written to disk.
