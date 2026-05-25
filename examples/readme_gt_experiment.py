import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.data.loader import load_data
from experiments.data.partitioner import DataSplitter
from experiments.metamodels.rf import Meta_rf
from prelim.generators import build_generator


ASSET_DIR = ROOT / "docs" / "assets"
FIGURE_PATH = ASSET_DIR / "readme-gt-experiment.png"
SUMMARY_PATH = ASSET_DIR / "readme-gt-experiment-summary.json"

REPETITIONS = 20
SAMPLE_SIZES = [25, 50, 100, 200, 400]
SPLIT_SEED = 2020
GENERATED_SAMPLE_SIZE = 100000


def load_scaled_split(split_index, n_samples):
    X, y = load_data("gt")
    partitioner = DataSplitter(seed=SPLIT_SEED).fit(X, y).configure(REPETITIONS, n_samples)
    X_train, y_train = partitioner.get_train(split_index)
    X_test, y_test = partitioner.get_test(split_index)

    variable_mask = X_train.max(axis=0) != X_train.min(axis=0)
    X_train = X_train[:, variable_mask]
    X_test = X_test[:, variable_mask]

    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), y_train, scaler.transform(X_test), y_test


def gt_exp(split_index, n_samples):
    seed = SPLIT_SEED + split_index
    X_train, y_train, X_test, y_test = load_scaled_split(split_index, n_samples)

    mediator = Meta_rf(seed=seed).fit(X_train, y_train)
    generator = build_generator("kde", seed=seed).fit(X_train, y_train)
    X_gen = generator.sample(GENERATED_SAMPLE_SIZE - len(y_train))
    y_gen = mediator.predict(X_gen)

    prelim_tree = DecisionTreeClassifier(max_leaf_nodes=8, random_state=seed)
    prelim_tree.fit(
        np.concatenate([X_train, X_gen]),
        np.concatenate([y_train, y_gen]),
    )

    baseline_tree = DecisionTreeClassifier(max_leaf_nodes=8, random_state=seed).fit(X_train, y_train)
    return prelim_tree.score(X_test, y_test), baseline_tree.score(X_test, y_test)


def run_experiment():
    rows = []
    for split_index in range(REPETITIONS):
        for size in SAMPLE_SIZES:
            acc_prelim, acc_baseline = gt_exp(split_index, size)
            rows.append(
                {
                    "Training set size": size,
                    "PRELIM": acc_prelim,
                    "Baseline": acc_baseline,
                }
            )
    return pd.DataFrame(rows)


def summarize(results):
    grouped = results.groupby("Training set size")[["PRELIM", "Baseline"]]
    means = grouped.mean()
    stds = grouped.std()
    summary = {
        "overall_mean": {
            "PRELIM": float(results["PRELIM"].mean()),
            "Baseline": float(results["Baseline"].mean()),
            "Difference": float((results["PRELIM"] - results["Baseline"]).mean()),
        },
        "by_size": {},
    }
    for size in means.index:
        summary["by_size"][str(size)] = {
            "PRELIM_mean": float(means.loc[size, "PRELIM"]),
            "PRELIM_std": float(stds.loc[size, "PRELIM"]),
            "Baseline_mean": float(means.loc[size, "Baseline"]),
            "Baseline_std": float(stds.loc[size, "Baseline"]),
            "Difference_mean": float(means.loc[size, "PRELIM"] - means.loc[size, "Baseline"]),
        }
    return summary


def plot_results(results):
    grouped = results.groupby("Training set size")[["PRELIM", "Baseline"]]
    means = grouped.mean()
    sems = grouped.sem()

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = {"PRELIM": "#2563eb", "Baseline": "#6b7280"}
    for method in ["PRELIM", "Baseline"]:
        ax.errorbar(
            means.index.astype(str),
            means[method],
            yerr=sems[method],
            marker="o",
            linewidth=2,
            capsize=4,
            label=method,
            color=colors[method],
        )

    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy on held-out data")
    ax.set_ylim(0.65, 0.9)
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(frameon=False)
    ax.set_title("PRELIM vs. direct decision-tree fitting on gt")
    fig.tight_layout()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=160)
    plt.close(fig)


def main():
    results = run_experiment()
    summary = summarize(results)
    plot_results(results)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
