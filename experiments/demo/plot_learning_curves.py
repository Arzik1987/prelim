"""Plot median turbine learning curves with standard-deviation bands."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator, MaxNLocator


MODELS = ("rf", "dt", "dtc", "dtval", "dt_pruned")
LABELS = {"rf": "RF", "dt": "DT", "dtc": "DTc", "dtval": "DT-CV", "dt_pruned": "DT-pruned"}
COLORS = {"rf": "#1f77b4", "dt": "#d62728", "dtc": "#2ca02c", "dtval": "#9467bd", "dt_pruned": "#ff7f0e"}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "output" / "results.csv",
        help="Per-iteration results CSV produced by run_demo.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "output" / "learning_curves.png",
        help="Output image path.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Use a logarithmic x-axis and place the naive point at x=1.",
    )
    return parser.parse_args()


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    results = results[results["model"].isin(MODELS)].copy()
    results["train_size"] = pd.to_numeric(results["train_size"])
    results["accuracy"] = pd.to_numeric(results["accuracy"])
    return (
        results.groupby(["model", "train_size"], as_index=False)
        .agg(median_accuracy=("accuracy", "median"), std_accuracy=("accuracy", "std"))
        .sort_values(["model", "train_size"])
    )


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.input)
    summary = summarize(results)
    if summary.empty:
        raise ValueError(f"No learning-curve model rows found in {args.input}")

    max_train_size = int(summary["train_size"].max())
    naive = results[results["model"] == "naive"]
    naive_accuracy = None
    if not naive.empty:
        naive_accuracy = pd.to_numeric(naive["accuracy"]).median()

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    for model in MODELS:
        curve = summary[summary["model"] == model]
        if curve.empty:
            continue
        x = curve["train_size"].to_numpy()
        median = curve["median_accuracy"].to_numpy()
        std = curve["std_accuracy"].fillna(0).to_numpy()
        if naive_accuracy is not None:
            naive_x = 1 if args.log_x else 0
            x = pd.concat([pd.Series([naive_x]), curve["train_size"]]).to_numpy()
            median = pd.concat([pd.Series([naive_accuracy]), curve["median_accuracy"]]).to_numpy()
            std = pd.concat([pd.Series([0.0]), curve["std_accuracy"].fillna(0)]).to_numpy()
        color = COLORS[model]
        axis.plot(x, median, label=LABELS[model], color=color, linewidth=2)
        axis.fill_between(
            x,
            median - std,
            median + std,
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    naive = results[results["model"] == "naive"]
    if not naive.empty:
        naive_accuracy = pd.to_numeric(naive["accuracy"]).median()
        axis.axhline(
            naive_accuracy,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=f"Naive ({naive_accuracy:.3f})",
        )

    axis.set_xlabel("Number of labeled training points")
    axis.set_ylabel("Held-out accuracy")
    axis.set_title("Turbine learning curves")
    if args.log_x:
        axis.set_xscale("log")
        axis.set_xlim(1, max_train_size)
        axis.xaxis.set_major_locator(LogLocator(base=10))
    else:
        axis.set_xlim(0, max_train_size)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    axis.set_ylim(0.7, 0.925)
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi)
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
