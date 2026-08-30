"""Produce a reproducible turbine learning curve."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.data.loader import load_data  # noqa: E402


DATASET = "turbine"
N_REPETITIONS = 10
SPLIT_SEED = 2020
RF_MAX_FEATURES = [2, "sqrt", None]
DTVAL_MAX_LEAVES = [2**power for power in range(1, 8)]
PRUNING_CRITERION = "gini"
MAX_PRUNING_ALPHAS = 12
MODEL_NAMES = {"rf", "dt", "dtc", "dtval", "dt_pruned"}
RESULT_FIELDS = [
    "dataset", "train_size", "repetition", "model", "accuracy",
    "train_accuracy", "cv_accuracy", "fit_seconds", "n_features",
    "selected_max_features", "selected_max_leaf_nodes", "selected_ccp_alpha",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--step", type=int, default=50)
    parser.add_argument("--max-train-size", type=int, default=1000)
    parser.add_argument(
        "--train-sizes",
        type=str,
        default=None,
        help="Comma-separated explicit training sizes; overrides step and max-train-size.",
    )
    return parser.parse_args()


def make_partitions(y: np.ndarray, train_size: int) -> list[np.ndarray]:
    """Create ten shuffled, evenly spaced training windows."""
    if train_size > len(y):
        raise ValueError(
            f"Training size {train_size} exceeds the dataset size {len(y)}"
        )
    for attempt in range(1000):
        rng = np.random.RandomState(SPLIT_SEED + train_size + attempt)
        permutation = rng.permutation(len(y))
        starts = np.linspace(
            0, len(y) - train_size, num=N_REPETITIONS, endpoint=True, dtype=int
        )
        partitions = [
            permutation[start : start + train_size].copy() for start in starts
        ]
        if all(np.unique(y[index]).size == 2 for index in partitions):
            return partitions
    raise RuntimeError(f"Could not construct valid partitions for size {train_size}")

def prepare_split(X, y, train_indices):
    test_mask = np.ones(len(y), dtype=bool)
    test_mask[train_indices] = False
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_mask], y[test_mask]
    variable_mask = X_train.max(axis=0) != X_train.min(axis=0)
    X_train, X_test = X_train[:, variable_mask], X_test[:, variable_mask]
    scaler = StandardScaler()
    return (
        scaler.fit_transform(X_train), y_train,
        scaler.transform(X_test), y_test, variable_mask,
    )


def fit_and_score(model, X_train, y_train, X_test, y_test):
    started = time.perf_counter()
    model.fit(X_train, y_train)
    return {
        "accuracy": float(model.score(X_test, y_test)),
        "train_accuracy": float(model.score(X_train, y_train)),
        "cv_accuracy": "",
        "fit_seconds": time.perf_counter() - started,
        "selected_max_features": "",
        "selected_max_leaf_nodes": "",
        "selected_ccp_alpha": "",
    }


def fit_rf(X_train, y_train, X_test, y_test):
    search = GridSearchCV(
        RandomForestClassifier(random_state=SPLIT_SEED),
        {"max_features": RF_MAX_FEATURES}, cv=5,
    )
    result = fit_and_score(search, X_train, y_train, X_test, y_test)
    result["cv_accuracy"] = float(search.best_score_)
    result["selected_max_features"] = str(search.best_params_["max_features"])
    return result


def fit_dtval(X_train, y_train, X_test, y_test):
    search = GridSearchCV(
        DecisionTreeClassifier(), {"max_leaf_nodes": DTVAL_MAX_LEAVES}, cv=5,
    )
    result = fit_and_score(search, X_train, y_train, X_test, y_test)
    result["cv_accuracy"] = float(search.best_score_)
    result["selected_max_leaf_nodes"] = int(search.best_params_["max_leaf_nodes"])
    return result


def pruning_alpha_candidates(X_train, y_train):
    path = DecisionTreeClassifier(
        criterion=PRUNING_CRITERION
    ).cost_complexity_pruning_path(X_train, y_train)
    alphas = np.unique(path.ccp_alphas[:-1])
    if len(alphas) <= MAX_PRUNING_ALPHAS:
        return alphas
    indices = np.linspace(0, len(alphas) - 1, MAX_PRUNING_ALPHAS, dtype=int)
    return alphas[np.unique(indices)]


def fit_dt_pruned(X_train, y_train, X_test, y_test):
    search = GridSearchCV(
        DecisionTreeClassifier(),
        {"criterion": [PRUNING_CRITERION],
         "ccp_alpha": pruning_alpha_candidates(X_train, y_train)},
        cv=5,
    )
    result = fit_and_score(search, X_train, y_train, X_test, y_test)
    result["cv_accuracy"] = float(search.best_score_)
    result["selected_ccp_alpha"] = float(search.best_params_["ccp_alpha"])
    return result


def write_result(writer, size, repetition, model, result, n_features):
    writer.writerow({
        "dataset": DATASET, "train_size": size, "repetition": repetition,
        "model": model, "accuracy": result["accuracy"],
        "train_accuracy": result["train_accuracy"],
        "cv_accuracy": result["cv_accuracy"],
        "fit_seconds": result["fit_seconds"], "n_features": n_features,
        "selected_max_features": result["selected_max_features"],
        "selected_max_leaf_nodes": result["selected_max_leaf_nodes"],
        "selected_ccp_alpha": result["selected_ccp_alpha"],
    })


def main() -> None:
    args = parse_args()
    if args.train_sizes is not None:
        try:
            sizes = tuple(sorted({
                int(value.strip())
                for value in args.train_sizes.split(",")
                if value.strip()
            }))
        except ValueError as error:
            raise ValueError(
                "--train-sizes must be a comma-separated list of integers"
            ) from error
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError("--train-sizes must contain positive integers")
    else:
        if args.step <= 0 or args.max_train_size < args.step:
            raise ValueError("Require 0 < --step <= --max-train-size")
        sizes = tuple(range(args.step, args.max_train_size + 1, args.step))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.csv"
    manifest_path = output_dir / "manifest.json"
    mean_path = output_dir / "mean_results.csv"
    if not args.overwrite and not args.resume and any(
        path.exists() for path in (result_path, manifest_path, mean_path)
    ):
        raise FileExistsError("Output exists; use --overwrite or --resume")

    X, y = load_data(DATASET)
    existing = pd.DataFrame()
    completed_sizes = set()
    if args.resume and result_path.exists():
        existing = pd.read_csv(result_path)
        counts = existing[existing["model"].isin(MODEL_NAMES)].groupby(
            ["train_size", "model"]
        ).size()
        completed_sizes = {
            size for size in sizes
            if all(counts.get((size, model), 0) >= N_REPETITIONS
                   for model in MODEL_NAMES)
        }
    sizes_to_run = tuple(size for size in sizes if size not in completed_sizes)
    manifest_sizes = sorted(
        set(sizes) | set(existing.get("train_size", pd.Series(dtype=int)))
    )
    naive_accuracy = float(max(np.mean(y), 1.0 - np.mean(y)))
    manifest = {
        "dataset": DATASET, "n_rows": int(len(y)),
        "train_sizes": manifest_sizes, "n_repetitions": N_REPETITIONS,
        "split_seed": SPLIT_SEED,
        "partition_policy": "ten shuffled training sets per size; disjoint when possible, otherwise evenly spaced windows; test is complement",
        "naive_accuracy": naive_accuracy,
        "models": {
            "rf": {"max_features": RF_MAX_FEATURES, "cv": 5},
            "dt": {"min_samples_split": 10},
            "dtc": {"max_leaf_nodes": 8},
            "dtval": {"max_leaf_nodes": DTVAL_MAX_LEAVES, "cv": 5},
            "dt_pruned": {
                "criterion": PRUNING_CRITERION,
                "ccp_alpha": "up to 12 pruning-path values",
                "cv": 5,
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    mode = "a" if args.resume else "w"
    with result_path.open(mode, newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_FIELDS)
        if not args.resume:
            writer.writeheader()
            for repetition in range(N_REPETITIONS):
                write_result(writer, 0, repetition, "naive", {
                    "accuracy": naive_accuracy, "train_accuracy": "",
                    "cv_accuracy": "", "fit_seconds": "",
                    "selected_max_features": "", "selected_max_leaf_nodes": "",
                    "selected_ccp_alpha": "",
                }, X.shape[1])
        for train_size in sizes_to_run:
            for repetition, indices in enumerate(make_partitions(y, train_size)):
                X_train, y_train, X_test, y_test, variable_mask = prepare_split(
                    X, y, indices
                )
                models = {
                    "rf": fit_rf(X_train, y_train, X_test, y_test),
                    "dt": fit_and_score(DecisionTreeClassifier(min_samples_split=10), X_train, y_train, X_test, y_test),
                    "dtc": fit_and_score(DecisionTreeClassifier(max_leaf_nodes=8), X_train, y_train, X_test, y_test),
                    "dtval": fit_dtval(X_train, y_train, X_test, y_test),
                    "dt_pruned": fit_dt_pruned(X_train, y_train, X_test, y_test),
                }
                for model, result in models.items():
                    write_result(writer, train_size, repetition, model, result, int(variable_mask.sum()))
                result_file.flush()

    results = pd.read_csv(result_path)
    summary = (
        results[results["model"] != "naive"]
        .groupby(["train_size", "model"], as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"),
             mean_fit_seconds=("fit_seconds", "mean"),
             mean_cv_accuracy=("cv_accuracy", "mean"))
    )
    summary = pd.concat([
        pd.DataFrame([{"train_size": 0, "model": "naive", "mean_accuracy": naive_accuracy}]),
        summary,
    ], ignore_index=True, sort=False)
    summary.to_csv(mean_path, index=False)
    print(f"Wrote demo results to {output_dir}")


if __name__ == "__main__":
    main()
