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
## Generator-assisted learning experiment

Run the generator experiment with a 600-point training set by default:

```powershell
python experiments/demo/generator_learning.py `
  --output-dir experiments/demo/generator_output `
  --train-size 600 `
  --gen-sizes "100,500,1000,2000,5000,10000" `
  --repetitions 10 `
  --threads 8
```

Each outer repetition creates its CSV immediately and appends rows as fits finish, so partial progress is visible. A repetition is complete only when all expected configuration rows are present. With `--resume`, existing rows are retained and only missing configurations are appended, so additional generation sizes can be added to an existing CSV without duplicating results. The experiment records baseline RF, pruned-tree, and shallow-tree accuracy; generated-point augmentation using `uniform` and Silverman-bandwidth `kde`; and the optional RF-labelled test-cut augmentation when more than 5,000 test points remain. Use `--resume` to skip completed repetitions or add missing configurations; use `--overwrite` to reset the requested repetition files.
For the generator-learning experiment, aggregate first and plot second:

```powershell
python experiments/demo/aggregate_generator_learning.py `
  --input-dir experiments/demo/generator_output

python experiments/demo/plot_generator_learning.py
python experiments/demo/plot_generator_learning.py --log-x `
  --output experiments/demo/generator_output/generator_learning_dt_pruned_logx.png
```
