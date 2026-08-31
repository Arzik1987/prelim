"""Evaluate generator-assisted decision-tree learning curves on the turbine demo."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.data.loader import load_data
from experiments.data.partitioner import DataSplitter



DATASET = "turbine"
SPLIT_SEED = 2020
RF_MAX_FEATURES = [2, "sqrt", None]
DTVAL_MAX_LEAVES = [2**power for power in range(1, 8)]
RESULT_FIELDS = [
    "train_size",
    "repetition",
    "stage",
    "generator",
    "gen_size",
    "model",
    "accuracy",
    "train_accuracy",
    "fit_seconds",
    "selected_max_leaf_nodes",
]


def parse_csv_ints(raw: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted({int(value.strip()) for value in raw.split(",") if value.strip()}))
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated list of integers") from error
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generator_learning_tasks",
    )
    parser.add_argument("--train-size", type=int, default=600)
    parser.add_argument(
        "--gen-sizes",
        default="100,500,1000,2000,5000,10000",
        help="Comma-separated generated/test-cut sizes.",
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def prepare_split(X, y, indices):
    test_mask = np.ones(len(y), dtype=bool)
    test_mask[indices] = False
    X_train, y_train = X[indices], y[indices]
    X_test, y_test = X[test_mask], y[test_mask]
    variable_mask = X_train.max(axis=0) != X_train.min(axis=0)
    X_train = X_train[:, variable_mask]
    X_test = X_test[:, variable_mask]
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), y_train, scaler.transform(X_test), y_test


def fit_score(model, X_train, y_train, X_test, y_test):
    started = time.perf_counter()
    model.fit(X_train, y_train)
    return (
        float(model.score(X_test, y_test)),
        float(model.score(X_train, y_train)),
        time.perf_counter() - started,
        getattr(model, "best_params_", {}).get("max_leaf_nodes", ""),
    )


def fit_rf(X_train, y_train, X_test, y_test):
    search = GridSearchCV(
        RandomForestClassifier(random_state=SPLIT_SEED, n_jobs=1),
        {"max_features": RF_MAX_FEATURES},
        cv=5,
        n_jobs=1,
    )
    return search, fit_score(search, X_train, y_train, X_test, y_test)


def pruning_alphas(X, y):
    path = DecisionTreeClassifier(criterion="gini").cost_complexity_pruning_path(X, y)
    alphas = np.unique(path.ccp_alphas[:-1])
    if len(alphas) <= 12:
        return alphas
    indices = np.linspace(0, len(alphas) - 1, 12, dtype=int)
    return alphas[np.unique(indices)]


def fit_pruned(X_train, y_train, X_test, y_test):
    search = GridSearchCV(
        DecisionTreeClassifier(),
        {"criterion": ["gini"], "ccp_alpha": pruning_alphas(X_train, y_train)},
        cv=5,
        n_jobs=1,
    )
    return search, fit_score(search, X_train, y_train, X_test, y_test)


def fit_dtc(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(max_leaf_nodes=8)
    return model, fit_score(model, X_train, y_train, X_test, y_test)


def result_row(train_size, repetition, stage, generator, gen_size, model, score):
    accuracy, train_accuracy, fit_seconds, selected_leaves = score
    return {
        "train_size": train_size,
        "repetition": repetition,
        "stage": stage,
        "generator": generator,
        "gen_size": gen_size,
        "model": model,
        "accuracy": accuracy,
        "train_accuracy": train_accuracy,
        "fit_seconds": fit_seconds,
        "selected_max_leaf_nodes": selected_leaves,
    }


def start_task_file(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        result_file.flush()
        os.fsync(result_file.fileno())


def append_rows(path: Path, rows: list[dict]) -> None:
    with path.open("a", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=RESULT_FIELDS)
        writer.writerows(rows)
        result_file.flush()
        os.fsync(result_file.fileno())


def row_key(row: dict) -> tuple:
    return (
        int(row["train_size"]),
        int(row["repetition"]),
        row["stage"],
        row["generator"],
        int(row["gen_size"]),
        row["model"],
    )


def read_existing_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    try:
        with path.open(newline="", encoding="utf-8") as result_file:
            return {row_key(row) for row in csv.DictReader(result_file)}
    except (OSError, KeyError, TypeError, ValueError, csv.Error):
        return set()


def append_missing(path: Path, rows: list[dict], existing_keys: set[tuple]) -> None:
    missing = []
    for row in rows:
        key = row_key(row)
        if key not in existing_keys:
            missing.append(row)
            existing_keys.add(key)
    if missing:
        append_rows(path, missing)


def expected_keys(train_size: int, repetition: int, test_size: int, gen_sizes: tuple[int, ...]) -> set[tuple]:
    keys = {
        (train_size, repetition, "baseline", "none", 0, model)
        for model in ("rf", "dt_pruned", "dtc")
    }
    for generator in ("uniform", "kde"):
        for gen_size in gen_sizes:
            keys.update(
                (train_size, repetition, "generated", generator, gen_size, model)
                for model in ("dt_pruned", "dtc")
            )
    for gen_size in gen_sizes:
        if test_size - gen_size > 5000:
            keys.update(
                (train_size, repetition, "test_cut", "rf_labelled_test", gen_size, model)
                for model in ("dt_pruned", "dtc")
            )
    return keys


def expected_rows(test_size: int, gen_sizes: tuple[int, ...]) -> int:
    test_cut_count = sum(test_size - gen_size > 5000 for gen_size in gen_sizes)
    return 3 + 4 * len(gen_sizes) + 2 * test_cut_count


def task_complete(
    path: Path,
    train_size: int,
    repetition: int,
    test_size: int,
    gen_sizes: tuple[int, ...],
) -> bool:
    return expected_keys(train_size, repetition, test_size, gen_sizes).issubset(
        read_existing_keys(path)
    )


def run_repetition(X, y, train_size, repetition, repetitions, gen_sizes, path):
    from prelim.generators.kde import Gen_kdebw
    from prelim.generators.rand import Gen_randu
    
    
    splitter = DataSplitter(seed=SPLIT_SEED)
    splitter.fit(X, y)
    splitter.configure(repetitions, train_size)
    X_raw, y_train = splitter.get_train(repetition)
    X_test_raw, y_test = splitter.get_test(repetition)
    variable_mask = X_raw.max(axis=0) != X_raw.min(axis=0)
    X_raw = X_raw[:, variable_mask]
    X_test_raw = X_test_raw[:, variable_mask]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_raw)
    X_test = scaler.transform(X_test_raw)
    if not path.exists() or path.stat().st_size == 0:
        start_task_file(path)
    existing_keys = read_existing_keys(path)

    rf, score = fit_rf(X_train, y_train, X_test, y_test)
    append_missing(path, [result_row(train_size, repetition, "baseline", "none", 0, "rf", score)], existing_keys)
    _, score = fit_pruned(X_train, y_train, X_test, y_test)
    append_missing(path, [result_row(train_size, repetition, "baseline", "none", 0, "dt_pruned", score)], existing_keys)
    _, score = fit_dtc(X_train, y_train, X_test, y_test)
    append_missing(path, [result_row(train_size, repetition, "baseline", "none", 0, "dtc", score)], existing_keys)

    uniform = Gen_randu(seed=SPLIT_SEED + repetition)
    kde = Gen_kdebw(method="silverman", seed=SPLIT_SEED + repetition)
    uniform.fit(X_train, y_train)
    kde.fit(X_train, y_train)
    for name, generator in (("uniform", uniform), ("kde", kde)):
        for gen_size in gen_sizes:
            X_generated = generator.sample(gen_size)
            y_generated = rf.predict(X_generated)
            X_augmented = np.concatenate([X_train, X_generated])
            y_augmented = np.concatenate([y_train, y_generated])
            _, score = fit_pruned(X_augmented, y_augmented, X_test, y_test)
            append_missing(path, [result_row(train_size, repetition, "generated", name, gen_size, "dt_pruned", score)], existing_keys)
            _, score = fit_dtc(X_augmented, y_augmented, X_test, y_test)
            append_missing(path, [result_row(train_size, repetition, "generated", name, gen_size, "dtc", score)], existing_keys)

    for gen_size in gen_sizes:
        if len(X_test) - gen_size <= 5000:
            continue
        X_cut = X_test[:gen_size]
        X_remaining = X_test[gen_size:]
        y_cut = rf.predict(X_cut)
        X_augmented = np.concatenate([X_train, X_cut])
        y_augmented = np.concatenate([y_train, y_cut])
        _, score = fit_pruned(X_augmented, y_augmented, X_remaining, y_test[gen_size:])
        append_missing(path, [result_row(train_size, repetition, "test_cut", "rf_labelled_test", gen_size, "dt_pruned", score)], existing_keys)
        _, score = fit_dtc(X_augmented, y_augmented, X_remaining, y_test[gen_size:])
        append_missing(path, [result_row(train_size, repetition, "test_cut", "rf_labelled_test", gen_size, "dtc", score)], existing_keys)
    return path


def main() -> None:
    args = parse_args()
    if args.train_size <= 0 or args.repetitions <= 0 or args.threads <= 0:
        raise ValueError("train-size, repetitions, and threads must be positive")
    gen_sizes = parse_csv_ints(args.gen_sizes, "--gen-sizes")
    X, y = load_data(DATASET)
    tasks_dir = args.tasks_dir.resolve()
    output_dir = tasks_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [output_dir / f"repetition_{rep:03d}.csv" for rep in range(args.repetitions)]
    if not args.overwrite and not args.resume and any(path.exists() for path in files):
        raise FileExistsError("Experiment files exist; use --overwrite or --resume")
    if args.overwrite:
        for path in files:
            if path.exists():
                path.unlink()
    tasks = [
        (rep, path)
        for rep, path in enumerate(files)
        if not (args.resume and task_complete(path, args.train_size, rep, len(y) - args.train_size, gen_sizes))
    ]
    print(f"Running {len(tasks)} repetitions with {args.threads} worker threads")
    with ThreadPoolExecutor(max_workers=args.threads, thread_name_prefix="generator-demo") as pool:
        futures = [
            pool.submit(run_repetition, X, y, args.train_size, rep, args.repetitions, gen_sizes, path)
            for rep, path in tasks
        ]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            print(f"Completed {completed}/{len(futures)}")
    manifest = {
        "dataset": DATASET,
        "train_size": args.train_size,
        "gen_sizes": gen_sizes,
        "repetitions": args.repetitions,
        "split_seed": SPLIT_SEED,
        "test_cut_condition": "test_size - gen_size > 5000",
        "kde": "Gen_kdebw with Silverman bandwidth",
        "labeler": "hard RF predict labels",
        "task_format": "one CSV per outer repetition",
    }
    (tasks_dir.parent / "generator_learning_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()



