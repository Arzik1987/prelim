"""Plot annotated generator-learning results."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator, MaxNLocator


GENERATORS = {
    "uniform": ("uniform", "#E69F00", "o"),
    "kde": ("kde", "#56B4E9", "s"),
    "rf_labelled_test": ("true distribution", "#CC79A7", "^"),
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "generator_output" / "aggregated_results.csv",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model",
        choices=["dt_pruned", "dtc"],
        default="dt_pruned",
        help="Tree model to plot (default: dt_pruned).",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--log-x", action="store_true", help="Use a logarithmic x-axis.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    suffix = "_logx" if args.log_x else ""
    output = args.output or here / "generator_output" / f"generator_learning_{args.model}{suffix}.png"

    data = pd.read_csv(args.input)
    train_size = int(data["train_size"].iloc[0])
    model_label = "decision tree" if args.model == "dt_pruned" else "shallow decision tree"
    model_color = "#D55E00" if args.model == "dt_pruned" else "#009E73"
    curves = data[
        (data["model"] == args.model)
        & data["stage"].isin(["generated", "test_cut"])
    ]
    baseline = data[
        (data["model"] == args.model) & (data["stage"] == "baseline")
    ]["mean_accuracy"].iloc[0]
    max_gen_size = int(curves["gen_size"].max())
    rf = data[
        (data["model"] == "rf") & (data["stage"] == "baseline")
    ]["mean_accuracy"].iloc[0]

    figure, axis = plt.subplots(figsize=(7.5, 5.5))
    plotted = []
    curve_min = baseline
    for generator, (label, color, marker) in GENERATORS.items():
        curve = curves[curves["generator"] == generator].sort_values("gen_size")
        if curve.empty:
            continue
        curve_min = min(curve_min, curve["mean_accuracy"].min())
        start_x = 1 if args.log_x else 0
        plot_data = pd.concat(
            [
                pd.DataFrame({"gen_size": [start_x], "mean_accuracy": [baseline]}),
                curve[["gen_size", "mean_accuracy"]],
            ],
            ignore_index=True,
        )
        axis.plot(
            plot_data["gen_size"],
            plot_data["mean_accuracy"],
            color=color,
            linewidth=2.8,
            marker=marker,
            markersize=5,
            markeredgewidth=0.7,
        )
        last_x = curve["gen_size"].iloc[-1]
        last_y = curve["mean_accuracy"].iloc[-1]
        value_offset = {"uniform": 8, "kde": 8, "rf_labelled_test": 8}[generator]
        axis.annotate(
            f"{last_y:.3f}",
            (last_x, last_y),
            xytext=(12, value_offset),
            textcoords="offset points",
            ha="center",
            color=color,
            fontsize=13,
        )
        plotted.append((generator, label, color, last_x, last_y))

    axis.axhline(
        baseline,
        color=model_color,
        linestyle="--",
        linewidth=1.1,
    )
    axis.axhline(
        rf,
        color="#0072B2",
        linestyle=":",
        linewidth=1.5,
    )

    step = 0.005
    lower = math.floor((curve_min - 1e-12) / step) * step
    upper = math.ceil(rf / step) * step
    upper = upper + step if upper <= rf else upper
    axis.set_ylim(lower, upper)
    axis.set_xlabel("number of generated points", fontsize=13)
    title = (
        "Better distribution matching speeds decision-tree\n"
        "convergence to random-forest performance"
        if args.model == "dt_pruned"
        else "Distribution approximation quality is decisive for the\n"
        "shallow decision tree accuracy"
    )
    axis.set_title(title, fontsize=18, pad=32, loc="left")
    axis.tick_params(axis="both", labelsize=11)
    axis.grid(axis="y", alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(True)
    axis.spines["bottom"].set_visible(True)

    label_x = max_gen_size * (1.25 if args.log_x else 1.12)
    for generator, label, color, last_x, last_y in plotted:
        if generator == "rf_labelled_test":
            axis.annotate(
                label,
                xy=(last_x, last_y),
                xytext=(-12, 12),
                textcoords="offset points",
                color=color,
                va="center",
                ha="right",
                fontsize=13,
                annotation_clip=False,
            )
        else:
            axis.annotate(
                label,
                xy=(last_x, last_y),
                xytext=(label_x, last_y),
                color=color,
                va="center",
                ha="left",
                fontsize=13,
                annotation_clip=False,
            )
    axis.annotate(
        f"{baseline:.3f}",
        (max_gen_size, baseline),
        xytext=(12, 2),
        textcoords="offset points",
        ha="center",
        color=model_color,
        fontsize=13,
    )
    axis.annotate(
        "no augmentation",
        xy=(max_gen_size, baseline),
        xytext=(label_x, baseline - 0.003),
        color=model_color,
        va="center",
        ha="left",
        fontsize=13,
        annotation_clip=False,
    )
    axis.annotate(
        f"{rf:.3f}",
        (max_gen_size, rf),
        xytext=(12, 8),
        textcoords="offset points",
        ha="center",
        color="#0072B2",
        fontsize=13,
    )
    axis.annotate(
        "random forest",
        xy=(max_gen_size, rf),
        xytext=(label_x, rf),
        color="#0072B2",
        va="center",
        ha="left",
        fontsize=13,
        annotation_clip=False,
    )

    if args.log_x:
        axis.set_xscale("log")
        axis.set_xlim(1, max_gen_size * 1.45)
        axis.xaxis.set_major_locator(LogLocator(base=10))
    else:
        axis.set_xlim(0, max_gen_size * 1.3)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
