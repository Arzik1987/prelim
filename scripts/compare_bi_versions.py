import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "docs" / "assets" / "bi_comparison_10000.csv"
HPO_ROWS = 100
FIT_ROWS = 10000


def load_module(name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


loader_mod = load_module("loader_mod", "experiments/data/loader.py")
helpers_mod = load_module("helpers_mod", "experiments/evaluation/helpers.py")
bi_old_mod = load_module("bi_old_mod", "src/prelim/sd/bi_slow.py")
bi_new_mod = load_module("bi_new_mod", "src/prelim/sd/bi.py")

load_data = loader_mod.load_data
get_bi_param = helpers_mod.get_bi_param
opt_param = helpers_mod.opt_param
BI_OLD = bi_old_mod.BI
BI_NEW = bi_new_mod.BI

DATASETS = (
    "ccpp",
    "occupancy",
    "electricity",
    "higgs7",
    "htru",
    "seoul",
    "shuttle",
    "turbine",
    "avila",
    "gt",
    "wine",
    "ees",
    "dry",
    "parkinson",
    "pendata",
    "ring",
    "higgs21",
    "jm1",
    "stocks",
    "anuran",
    "cc",
    "sensorless",
    "ml",
    "saac2",
    "bankruptcy",
    "sylva",
    "nomao",
    "gas",
    "clean2",
    "seizure",
)

FIELDNAMES = (
    "dataset",
    "hpo_rows",
    "fit_rows",
    "cols",
    "depth_grid",
    "old_cv_depth",
    "new_cv_depth",
    "old_refit_depth",
    "new_refit_depth",
    "old_score",
    "new_score",
    "old_nrestr",
    "new_nrestr",
    "old_fit_runtime_s",
    "new_fit_runtime_s",
    "old_hpo_runtime_s",
    "new_hpo_runtime_s",
    "box_same",
    "error",
)


def prepare_data(dataset):
    X, y = load_data(dataset)
    variable_mask = X.max(axis=0) != X.min(axis=0)
    X = X[:, variable_mask]
    return X, y


def cap_rows(X, y, limit):
    if X.shape[0] > limit:
        return X[:limit].copy(), y[:limit].copy()
    return X.copy(), y.copy()


def scale_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def run_bi_hpo(model_cls, X, y, depth_grid):
    start = time.perf_counter()
    cv = GridSearchCV(model_cls(), {"depth": depth_grid}, refit=False).fit(X, y).cv_results_
    cv_scores = opt_param(cv, len(depth_grid))
    best_depth = int(depth_grid[int(np.argmax(cv_scores))])
    model = model_cls(depth=best_depth)
    model.fit(X, y)
    elapsed = time.perf_counter() - start
    return model, best_depth, elapsed


def fit_bi_once(model_cls, X, y, depth):
    model = model_cls(depth=depth)
    start = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - start
    return model, elapsed


def main():
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        handle.flush()

        for dataset in DATASETS:
            row = {
                "dataset": dataset,
                "hpo_rows": "",
                "fit_rows": "",
                "cols": "",
                "depth_grid": "",
                "old_cv_depth": "",
                "new_cv_depth": "",
                "old_refit_depth": "",
                "new_refit_depth": "",
                "old_score": "",
                "new_score": "",
                "old_nrestr": "",
                "new_nrestr": "",
                "old_fit_runtime_s": "",
                "new_fit_runtime_s": "",
                "old_hpo_runtime_s": "",
                "new_hpo_runtime_s": "",
                "box_same": "",
                "error": "",
            }

            try:
                X_raw, y_raw = prepare_data(dataset)
                X_hpo, y_hpo = cap_rows(X_raw, y_raw, HPO_ROWS)
                X_fit, y_fit = cap_rows(X_raw, y_raw, FIT_ROWS)

                X_hpo = scale_data(X_hpo)
                X_fit = scale_data(X_fit)

                depth_grid = np.asarray(get_bi_param(5, X_fit.shape[1]), dtype=int)
                old_hpo_model, old_cv_depth, old_hpo_elapsed = run_bi_hpo(BI_OLD, X_hpo, y_hpo, depth_grid)
                new_hpo_model, new_cv_depth, new_hpo_elapsed = run_bi_hpo(BI_NEW, X_hpo, y_hpo, depth_grid)

                old_refit_depth = int(old_hpo_model.get_nrestr())
                new_refit_depth = int(new_hpo_model.get_nrestr())

                old_model, old_fit_elapsed = fit_bi_once(BI_OLD, X_fit, y_fit, old_refit_depth)
                new_model, new_fit_elapsed = fit_bi_once(BI_NEW, X_fit, y_fit, new_refit_depth)

                row.update(
                    {
                        "hpo_rows": int(X_hpo.shape[0]),
                        "fit_rows": int(X_fit.shape[0]),
                        "cols": int(X_fit.shape[1]),
                        "depth_grid": json.dumps(depth_grid.tolist()),
                        "old_cv_depth": old_cv_depth,
                        "new_cv_depth": new_cv_depth,
                        "old_refit_depth": old_refit_depth,
                        "new_refit_depth": new_refit_depth,
                        "old_score": old_model.score(X_fit, y_fit),
                        "new_score": new_model.score(X_fit, y_fit),
                        "old_nrestr": old_model.get_nrestr(),
                        "new_nrestr": new_model.get_nrestr(),
                        "old_fit_runtime_s": old_fit_elapsed,
                        "new_fit_runtime_s": new_fit_elapsed,
                        "old_hpo_runtime_s": old_hpo_elapsed,
                        "new_hpo_runtime_s": new_hpo_elapsed,
                        "box_same": bool(np.allclose(old_model.box_, new_model.box_, equal_nan=True)),
                    }
                )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"

            writer.writerow(row)
            handle.flush()


if __name__ == "__main__":
    main()
