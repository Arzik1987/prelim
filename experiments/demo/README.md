# Turbine learning-curve demo

Run commands from the repository root.

## Learning curve

Run independent learning-curve tasks:

```powershell
python experiments/demo/run_learning_curve.py --train-sizes "50,75,100,6000,7000,8000,9000,10000" --threads 8
```

Each `(train_size, repetition)` task writes one five-row CSV to `learning_curve_tasks/`. Use `--resume` to reuse complete task files, or `--overwrite` to rerun the requested sizes.

Aggregate and plot:

```powershell
python experiments/demo/aggregate_learning_curve.py
python experiments/demo/plot_learning_curve.py
```

The learning-curve plot always uses a logarithmic x-axis. Its outputs are kept directly in `experiments/demo/`:

- `learning_curve_manifest.json`
- `learning_curve_results.csv`
- `learning_curve_summary.csv`
- `learning_curve.png`

For a regular grid, use `--step` and `--max-train-size`:

```powershell
python experiments/demo/run_learning_curve.py --step 50 --max-train-size 1000 --threads 8
```

## Generator-assisted learning

Run the generator experiment:

```powershell
python experiments/demo/run_generator_learning.py `
  --train-size 600 `
  --gen-sizes "100,500,1000,2000,5000,10000" `
  --repetitions 10 `
  --threads 8
```

Task CSVs are stored in `generator_learning_tasks/`. `--resume` retains complete configurations and fills only missing rows; `--overwrite` resets the requested repetitions.

Aggregate and plot:

```powershell
python experiments/demo/aggregate_generator_learning.py
python experiments/demo/plot_generator_learning.py
python experiments/demo/plot_generator_learning.py --model dtc
```

Generator artifacts are also stored directly in `experiments/demo/`:

- `generator_learning_manifest.json`
- `generator_learning_results.csv`
- `generator_learning_dt_pruned.png`
- `generator_learning_dtc.png`

Generator plots always use a logarithmic x-axis.

## Icons

Create the presentation icons and schematic figures with:

```powershell
python experiments/demo/draw_demo_icons.py
```

They are written to `icons/`.