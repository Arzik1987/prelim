"""Plot minimal turbine learning curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator


MODELS = ("rf", "dtc", "dt_pruned")
COLORS = {"rf": "#0072B2", "dtc": "#009E73", "dt_pruned": "#D55E00"}
MARKERS = {"rf": "o", "dtc": "s", "dt_pruned": "^"}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=here / "learning_curve_results.csv")
    parser.add_argument("--output", type=Path, default=here / "learning_curve.png")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    results = results[results["model"].isin(MODELS)].copy()
    results["train_size"] = pd.to_numeric(results["train_size"])
    results["accuracy"] = pd.to_numeric(results["accuracy"])
    return (
        results.groupby(["model", "train_size"], as_index=False)
        .agg(median_accuracy=("accuracy", "median"))
        .sort_values(["model", "train_size"])
    )


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.input)
    summary = summarize(results)
    if summary.empty:
        raise ValueError(f"No learning-curve model rows found in {args.input}")

    max_train_size = int(summary["train_size"].max())
    figure, axis = plt.subplots(figsize=(5.5, 5.5))
    for model in MODELS:
        curve = summary[summary["model"] == model]
        if curve.empty:
            continue
        axis.plot(
            curve["train_size"].to_numpy(),
            curve["median_accuracy"].to_numpy(),
            color=COLORS[model],
            linewidth=2.8,
            marker=MARKERS[model],
            markersize=5,
            markeredgewidth=0.7,
        )

    axis.set_xlabel("train data size", fontsize=18)
    axis.set_ylabel("quality", fontsize=18)
    axis.set_xscale("log")
    axis.set_xlim(40, max_train_size * 1.45)
    axis.set_ylim(0.7, 0.95)
    axis.xaxis.set_major_locator(LogLocator(base=10))

    # Keep only the two spines that form the L-shaped frame.
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(True)
    axis.spines["bottom"].set_visible(True)
    axis.tick_params(axis="both", which="both", bottom=False, left=False,
                     labelbottom=False, labelleft=False)
    axis.grid(False)

    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
