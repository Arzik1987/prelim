import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from prelim.prelim import prelim


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets"
FIGURE_PATH = ASSET_DIR / "readme-small-experiment.png"
SUMMARY_PATH = ASSET_DIR / "readme-small-experiment-summary.json"


def make_dataset(n_samples, rng, cov=((1, 0), (0, 1)), mean0=(0, 0), mean1=(1, 1)):
    X0 = rng.multivariate_normal(mean0, cov, n_samples)
    X1 = rng.multivariate_normal(mean1, cov, n_samples)
    X = np.vstack((X0, X1))
    y = np.hstack((np.zeros(n_samples), np.ones(n_samples))).astype(int)
    return X, y


def small_exp(n_samples, seed):
    rng = np.random.default_rng(seed)
    X_train, y_train = make_dataset(n_samples, rng)
    X_test, y_test = make_dataset(100 * n_samples, rng)

    mediator = RandomForestClassifier(random_state=seed)
    tree = DecisionTreeClassifier(max_leaf_nodes=8, random_state=seed)
    wb_model = prelim(
        X_train,
        y_train,
        mediator,
        tree,
        gen_name="kde",
        new_size=100 * n_samples,
        proba=False,
        verbose=False,
        seed=seed,
    )

    baseline_model = DecisionTreeClassifier(max_leaf_nodes=8, random_state=seed).fit(X_train, y_train)
    return wb_model.score(X_test, y_test), baseline_model.score(X_test, y_test)


def run_experiment():
    repetitions = 20
    sample_sizes = [25, 50, 100, 200, 400]
    rows = []

    for rep in range(repetitions):
        for size in sample_sizes:
            seed = 2020 + rep * 1000 + size
            acc_prelim, acc_baseline = small_exp(size, seed)
            rows.append(
                {
                    "Small data size": size,
                    "PRELIM": acc_prelim,
                    "Baseline": acc_baseline,
                }
            )

    return pd.DataFrame(rows)


def summarize(results):
    grouped = results.groupby("Small data size")[["PRELIM", "Baseline"]]
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
    grouped = results.groupby("Small data size")[["PRELIM", "Baseline"]]
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

    ax.set_xlabel("Training examples per class")
    ax.set_ylabel("Accuracy on held-out data")
    ax.set_ylim(0.55, 0.85)
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(frameon=False)
    ax.set_title("PRELIM vs. direct decision-tree fitting")
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
