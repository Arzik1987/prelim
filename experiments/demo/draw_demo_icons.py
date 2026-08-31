"""Create schematic icons for large and small artificial neural networks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch


EDGE_COLOR = "#8A8F98"
NODE_COLOR = "#FFFFFF"
NODE_EDGE_COLOR = "#243447"
INPUT_COLOR = "#0072B2"
HIDDEN_COLOR = "#009E73"
OUTPUT_COLOR = "#D55E00"
BOUNDARY_COLOR = "#20252B"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=here / "icons",
        help="Directory in which to save the icons.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def layer_positions(layer_sizes: tuple[int, ...]) -> list[list[tuple[float, float]]]:
    """Return evenly spaced node positions, centered vertically per layer."""
    positions = []
    for layer_index, size in enumerate(layer_sizes):
        if size == 1:
            ys = [0.5]
        else:
            ys = [0.1 + 0.8 * i / (size - 1) for i in range(size)]
        x = layer_index / (len(layer_sizes) - 1)
        positions.append([(x, y) for y in ys])
    return positions


def draw_icon(layer_sizes: tuple[int, ...], output: Path, dpi: int) -> None:
    positions = layer_positions(layer_sizes)
    figure, axis = plt.subplots(figsize=(3.5, 3.5))
    figure.patch.set_alpha(0)
    axis.set_facecolor("none")

    for left_layer, right_layer in zip(positions, positions[1:]):
        for left_x, left_y in left_layer:
            for right_x, right_y in right_layer:
                axis.plot(
                    [left_x, right_x], [left_y, right_y],
                    color=EDGE_COLOR, linewidth=1.4, alpha=0.42,
                    solid_capstyle="round", zorder=1,
                )

    for layer_index, layer in enumerate(positions):
        color = (
            INPUT_COLOR if layer_index == 0
            else OUTPUT_COLOR if layer_index == len(positions) - 1
            else HIDDEN_COLOR
        )
        radius = 0.045 if len(layer) >= 6 else 0.06
        for x, y in layer:
            axis.add_patch(
                Circle(
                    (x, y), radius=radius, facecolor=NODE_COLOR,
                    edgecolor=color, linewidth=1.8, zorder=2,
                )
            )

    axis.set_xlim(-0.08, 1.08)
    axis.set_ylim(-0.04, 1.04)
    axis.set_aspect("equal")
    axis.axis("off")
    figure.savefig(output, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def draw_interpretable_icon(output: Path, dpi: int) -> None:
    """Draw an explicit shallow binary decision-tree icon."""
    figure, axis = plt.subplots(figsize=(3.5, 3.5))
    figure.patch.set_alpha(0)
    axis.set_facecolor("none")
    root = (0.5, 0.86)
    internal = [(0.28, 0.57), (0.72, 0.57)]
    leaves = [(0.14, 0.18), (0.38, 0.18), (0.62, 0.18), (0.86, 0.18)]
    for parent, children in ((root, internal), (internal[0], leaves[:2]), (internal[1], leaves[2:])):
        for child in children:
            axis.plot([parent[0], child[0]], [parent[1] - 0.055, child[1] + 0.075],
                      color=BOUNDARY_COLOR, linewidth=3.2, solid_capstyle="round", zorder=1)
    def node(center: tuple[float, float], width: float, height: float, color: str, label: str) -> None:
        x, y = center
        axis.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                       boxstyle="round,pad=0.025", facecolor=color,
                                       edgecolor="white", linewidth=2.4, zorder=2))
        axis.text(x, y, label, ha="center", va="center", fontsize=10,
                  color="white" if color != "white" else BOUNDARY_COLOR,
                  fontweight="bold", zorder=3)
    node(root, 0.22, 0.12, HIDDEN_COLOR, "split")
    for center in internal:
        node(center, 0.18, 0.11, HIDDEN_COLOR, "split")
    for index, center in enumerate(leaves):
        node(center, 0.16, 0.11, INPUT_COLOR if index % 2 == 0 else OUTPUT_COLOR,
             "")
    axis.set_xlim(0.02, 0.98)
    axis.set_ylim(0.08, 0.96)
    axis.set_aspect("equal")
    axis.axis("off")
    figure.savefig(output, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


def make_donut_points(count: int, rng: np.random.Generator, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate annulus points, split into two classes by a horizontal line."""
    angles = rng.uniform(0, 2 * np.pi, count)
    radii = np.sqrt(rng.uniform(0.58**2, 1.0**2, count))
    points = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    if noise:
        points += rng.normal(0.0, noise, points.shape)
    labels = (np.sin(angles) >= 0).astype(int)
    return points, labels


def draw_donut_scatter(points: np.ndarray, labels: np.ndarray, output: Path, dpi: int, boundary: tuple[np.ndarray, np.ndarray]) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 5.5))
    for label, color in enumerate((INPUT_COLOR, OUTPUT_COLOR)):
        selected = labels == label
        axis.scatter(points[selected, 0], points[selected, 1], s=36 if len(points) > 100 else 60,
                     color=color, edgecolors="white", linewidths=0.35, alpha=0.86)
    axis.plot(boundary[0], boundary[1], color=BOUNDARY_COLOR, linewidth=4.4, zorder=3)
    axis.set_aspect("equal")
    axis.set_xlim(-1.22, 1.22)
    axis.set_ylim(-1.22, 1.22)
    axis.axis("off")
    figure.savefig(output, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


def draw_donut_figures(output_dir: Path, dpi: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    clean_points, clean_labels = make_donut_points(100, rng)
    clean_x = np.linspace(-1.0, 1.0, 200)
    draw_donut_scatter(clean_points, clean_labels, output_dir / "donut_clean_100.png", dpi,
                       (clean_x, np.zeros_like(clean_x)))

    noisy_points, _ = make_donut_points(1000, rng, noise=0.12)
    boundary_x = np.linspace(-1.15, 1.15, 45)
    boundary_y = np.cumsum(rng.normal(0.0, 0.018, len(boundary_x)))
    boundary_y -= boundary_y.mean()
    boundary_y = np.clip(boundary_y, -0.18, 0.18)
    noisy_labels = (noisy_points[:, 1] >= np.interp(noisy_points[:, 0], boundary_x, boundary_y)).astype(int)
    draw_donut_scatter(noisy_points, noisy_labels, output_dir / "donut_noisy_1000.png", dpi,
                       (boundary_x, boundary_y))
def draw_generator_icon(output: Path, dpi: int) -> None:
    """Draw a schematic probability-density generator icon."""
    x = np.linspace(-3.2, 3.2, 400)
    density = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    figure, axis = plt.subplots(figsize=(3.5, 3.5))
    figure.patch.set_alpha(0)
    axis.set_facecolor("none")
    axis.fill_between(x, 0, density, color=HIDDEN_COLOR, alpha=0.22)
    axis.plot(x, density, color=HIDDEN_COLOR, linewidth=4.4)
    axis.set_xlim(-3.2, 3.2)
    axis.set_ylim(0, density.max() * 1.12)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(True)
    axis.spines["bottom"].set_visible(True)
    axis.tick_params(axis="both", which="both", bottom=False, left=False,
                     labelbottom=False, labelleft=False)
    figure.savefig(output, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)

def draw_interpretable_compact_icon(output: Path, dpi: int) -> None:
    """Draw a three-leaf binary decision-tree icon."""
    figure, axis = plt.subplots(figsize=(3.5, 3.5))
    figure.patch.set_alpha(0)
    axis.set_facecolor("none")
    root = (0.5, 0.86)
    direct_leaf = (0.20, 0.50)
    internal = (0.72, 0.55)
    leaves = [(0.59, 0.18), (0.86, 0.18)]
    for parent, child in ((root, direct_leaf), (root, internal), (internal, leaves[0]), (internal, leaves[1])):
        axis.plot([parent[0], child[0]], [parent[1] - 0.055, child[1] + 0.075],
                  color=BOUNDARY_COLOR, linewidth=3.2, solid_capstyle="round", zorder=1)
    def node(center: tuple[float, float], width: float, height: float, color: str, label: str) -> None:
        x, y = center
        axis.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                       boxstyle="round,pad=0.025", facecolor=color,
                                       edgecolor="white", linewidth=2.4, zorder=2))
        axis.text(x, y, label, ha="center", va="center", fontsize=10, color="white",
                  fontweight="bold", zorder=3)
    node(root, 0.22, 0.12, HIDDEN_COLOR, "split")
    node(internal, 0.18, 0.11, HIDDEN_COLOR, "split")
    node(direct_leaf, 0.16, 0.11, INPUT_COLOR, "")
    node(leaves[0], 0.16, 0.11, OUTPUT_COLOR, "")
    node(leaves[1], 0.16, 0.11, INPUT_COLOR, "")
    axis.set_xlim(0.02, 0.98)
    axis.set_ylim(0.08, 0.96)
    axis.set_aspect("equal")
    axis.axis("off")
    figure.savefig(output, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_icon((4, 6, 6, 2), args.output_dir / "large_ann.png", args.dpi)
    draw_icon((4, 3, 2), args.output_dir / "small_ann.png", args.dpi)
    draw_donut_figures(args.output_dir, args.dpi, args.seed)
    draw_generator_icon(args.output_dir / "generator_icon.png", args.dpi)
    draw_interpretable_icon(args.output_dir / "interpretable_model_icon.png", args.dpi)
    draw_interpretable_compact_icon(args.output_dir / "interpretable_model_compact_icon.png", args.dpi)
    print(f"Wrote {args.output_dir / 'large_ann.png'}")
    print(f"Wrote {args.output_dir / 'small_ann.png'}")
    print(f"Wrote {args.output_dir / 'donut_clean_100.png'}")
    print(f"Wrote {args.output_dir / 'donut_noisy_1000.png'}")
    print(f"Wrote {args.output_dir / 'generator_icon.png'}")
    print(f"Wrote {args.output_dir / 'interpretable_model_icon.png'}")
    print(f"Wrote {args.output_dir / 'interpretable_model_compact_icon.png'}")

if __name__ == "__main__":
    main()
