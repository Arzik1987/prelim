"""Plot annotated turbine learning curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator, MaxNLocator


MODELS = ("rf", "dtc", "dt_pruned")
LABELS = {"rf": "random forest", "dtc": "shallow decision tree", "dt_pruned": "decision tree"}
COLORS = {"rf": "#0072B2", "dtc": "#009E73", "dt_pruned": "#D55E00"}
MARKERS = {"rf": "o", "dtc": "s", "dt_pruned": "^"}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=here / "output" / "results.csv")
    parser.add_argument("--output", type=Path, default=here / "output" / "learning_curves.png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--log-x", action="store_true", help="Use a logarithmic x-axis.")
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
    naive = results[results["model"] == "naive"]
    naive_accuracy = pd.to_numeric(naive["accuracy"]).median() if not naive.empty else None
    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    plotted = []
    for model in MODELS:
        curve = summary[summary["model"] == model]
        if curve.empty:
            continue
        train_x = curve["train_size"].to_numpy()
        train_y = curve["median_accuracy"].to_numpy()
        x = train_x
        y = train_y
        if naive_accuracy is not None:
            naive_x = 1 if args.log_x else 0
            x = pd.concat([pd.Series([naive_x]), curve["train_size"]]).to_numpy()
            y = pd.concat([pd.Series([naive_accuracy]), curve["median_accuracy"]]).to_numpy()
        color = COLORS[model]
        axis.plot(x, y, color=color, linewidth=2.8, marker=MARKERS[model], markersize=5, markeredgewidth=0.7)
        last_x, last_y = train_x[-1], train_y[-1]
        axis.annotate(
            f"{last_y:.3f}", (last_x, last_y), xytext=(12, {"rf": 8, "dtc": 18, "dt_pruned": 8}[model]),
            textcoords="offset points", ha="center", color=color, fontsize=13,
        )
        plotted.append((model, last_y))

    if naive_accuracy is not None:
        naive_start = 1 if args.log_x else 0
        naive_end = max_train_size * (1.04 if args.log_x else 1.02)
        axis.plot([naive_start, naive_end], [naive_accuracy, naive_accuracy], color="black", linestyle="--", linewidth=1.1)
        axis.annotate(f"{naive_accuracy:.3f}", (naive_end, naive_accuracy), xytext=(12, 7),
                      textcoords="offset points", ha="center", color="black", fontsize=13)

    axis.set_xlabel("training data size", fontsize=13)
    axis.set_title(
        "At small training-data sizes, random forest accuracy\nrises faster than decision-tree accuracy",
        fontsize=18, pad=32, loc="left",
    )
    axis.tick_params(axis="both", labelsize=11)
    axis.set_ylim(0.7, 0.95)
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(True)
    axis.spines["bottom"].set_visible(True)

    label_x = max_train_size * (1.25 if args.log_x else 1.12)
    for model, last_y in plotted:
        axis.annotate(
            LABELS[model], xy=(max_train_size, last_y), xytext=(label_x, last_y - 0.004),
            color=COLORS[model], va="center", ha="left", fontsize=13,
            annotation_clip=False,
        )
    if naive_accuracy is not None:
        axis.annotate("naive", xy=(max_train_size, naive_accuracy), xytext=(label_x, naive_accuracy - 0.004),
                      color="black", va="center", ha="left", fontsize=13, annotation_clip=False)

    if args.log_x:
        axis.set_xscale("log")
        axis.set_xlim(1, max_train_size * 1.45)
        axis.xaxis.set_major_locator(LogLocator(base=10))
    else:
        axis.set_xlim(0, max_train_size * 1.3)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
