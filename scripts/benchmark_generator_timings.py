from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import statistics
import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class BenchmarkResult:
    generator: str
    n_rows: int
    n_features: int
    generated_rows: int
    repeat: int
    fit_seconds: float
    sample_seconds: float
    total_seconds: float
    sample_shape: tuple[int, int]
    status: str = "ok"
    error: str = ""


def _ensure_local_package():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_root = os.path.join(repo_root, "src")
    prelim_root = os.path.join(src_root, "prelim")
    generators_root = os.path.join(prelim_root, "generators")

    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    if "prelim" not in sys.modules:
        prelim_spec = importlib.util.spec_from_loader("prelim", loader=None, is_package=True)
        prelim_module = importlib.util.module_from_spec(prelim_spec)
        prelim_module.__path__ = [prelim_root]
        sys.modules["prelim"] = prelim_module

    if "prelim.generators" not in sys.modules:
        generators_spec = importlib.util.spec_from_loader("prelim.generators", loader=None, is_package=True)
        generators_module = importlib.util.module_from_spec(generators_spec)
        generators_module.__path__ = [generators_root]
        sys.modules["prelim.generators"] = generators_module

    return generators_root


def _load_generator_class(module_basename: str, class_name: str):
    generators_root = _ensure_local_package()
    module_name = f"prelim.generators.{module_basename}"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        module_path = os.path.join(generators_root, f"{module_basename}.py")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return getattr(module, class_name)


def build_dataset(n_rows: int, n_features: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    latent_dim = max(2, min(8, n_features // 2))
    latent = rng.normal(size=(n_rows, latent_dim))
    weights = rng.normal(scale=0.6, size=(latent_dim, n_features))
    noise = rng.normal(scale=0.15, size=(n_rows, n_features))
    data = latent @ weights + noise
    data += rng.normal(scale=0.1, size=(1, n_features))
    return data.astype(float)


def build_generators(seed: int):
    Gen_gibbs = _load_generator_class("gibbs", "Gen_gibbs")
    Gen_ctgan = _load_generator_class("ctgan", "Gen_ctgan")
    Gen_tabgan = _load_generator_class("tabgan", "Gen_tabgan")
    return {
        "gibbs": lambda: Gen_gibbs(
            model_kwargs={
                "hidden_dim": 64,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1,
                "mixture_components": 5,
            },
            train_kwargs={
                "epochs": 10,
                "batch_size": 128,
                "lr": 3e-4,
                "weight_decay": 1e-6,
                "grad_clip": 5.0,
                "device": "cpu",
            },
            sample_kwargs={
                "gibbs_rounds": 3,
                "batch_size": 256,
            },
            seed=seed,
        ),
        "ctgan": lambda: Gen_ctgan(
            model_kwargs={
                "epochs": 10,
                "verbose": False,
            },
            seed=seed,
        ),
        "tabgan": lambda: Gen_tabgan(
            generator_kwargs={
                "gen_x_times": 1.1,
            },
            seed=seed,
        ),
    }


def benchmark_one(generator_name: str, n_rows: int, n_features: int, generated_rows: int, repeat: int, seed: int):
    X = build_dataset(n_rows=n_rows, n_features=n_features, seed=seed + repeat)
    generator = build_generators(seed=seed + repeat)[generator_name]()

    fit_started = time.perf_counter()
    generator.fit(X)
    fit_seconds = time.perf_counter() - fit_started

    sample_started = time.perf_counter()
    try:
        sample = generator.sample(n_samples=generated_rows)
        sample_seconds = time.perf_counter() - sample_started
        sample_shape = tuple(sample.shape)
        status = "ok"
        error = ""
    except Exception as exc:
        sample_seconds = time.perf_counter() - sample_started
        sample_shape = (0, 0)
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    return BenchmarkResult(
        generator=generator_name,
        n_rows=n_rows,
        n_features=n_features,
        generated_rows=generated_rows,
        repeat=repeat,
        fit_seconds=fit_seconds,
        sample_seconds=sample_seconds,
        total_seconds=fit_seconds + sample_seconds,
        sample_shape=sample_shape,
        status=status,
        error=error,
    )


def summarize(results: list[BenchmarkResult]):
    grouped = {}
    for result in results:
        key = (result.generator, result.n_rows, result.n_features, result.generated_rows)
        grouped.setdefault(key, []).append(result)

    lines = []
    for key in sorted(grouped):
        runs = grouped[key]
        fit_values = [run.fit_seconds for run in runs]
        sample_values = [run.sample_seconds for run in runs]
        total_values = [run.total_seconds for run in runs]
        statuses = [run.status for run in runs]
        errors = [run.error for run in runs if run.error]
        lines.append(
            {
                "generator": key[0],
                "n_rows": key[1],
                "n_features": key[2],
                "generated_rows": key[3],
                "repeats": len(runs),
                "fit_seconds_mean": statistics.mean(fit_values),
                "fit_seconds_stdev": statistics.pstdev(fit_values) if len(fit_values) > 1 else 0.0,
                "sample_seconds_mean": statistics.mean(sample_values),
                "sample_seconds_stdev": statistics.pstdev(sample_values) if len(sample_values) > 1 else 0.0,
                "total_seconds_mean": statistics.mean(total_values),
                "total_seconds_stdev": statistics.pstdev(total_values) if len(total_values) > 1 else 0.0,
                "status": "ok" if all(status == "ok" for status in statuses) else "error",
                "errors": errors,
            }
        )
    return lines


def print_summary(summary_rows: list[dict]):
    header = (
        f"{'generator':<8} {'rows':>5} {'feat':>5} {'gen':>5} "
        f"{'fit_mean_s':>12} {'sample_mean_s':>14} {'total_mean_s':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['generator']:<8} {row['n_rows']:>5} {row['n_features']:>5} {row['generated_rows']:>5} "
            f"{row['fit_seconds_mean']:>12.3f} {row['sample_seconds_mean']:>14.3f} {row['total_seconds_mean']:>13.3f}"
        )
        if row["status"] != "ok":
            print(f"  status={row['status']} errors={row['errors']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="100,400")
    parser.add_argument("--features", default="4,8,16,32")
    parser.add_argument("--generators", default="gibbs,ctgan,tabgan")
    parser.add_argument("--generated-rows", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    args = parse_args()
    row_values = [int(value.strip()) for value in args.rows.split(",") if value.strip()]
    feature_values = [int(value.strip()) for value in args.features.split(",") if value.strip()]
    generator_names = [value.strip() for value in args.generators.split(",") if value.strip()]

    results = []
    for n_rows in row_values:
        generated_rows = n_rows if args.generated_rows is None else args.generated_rows
        for n_features in feature_values:
            for generator_name in generator_names:
                for repeat in range(args.repeats):
                    print(
                        f"running generator={generator_name} rows={n_rows} features={n_features} "
                        f"repeat={repeat + 1}/{args.repeats}",
                        flush=True,
                    )
                    results.append(
                        benchmark_one(
                            generator_name=generator_name,
                            n_rows=n_rows,
                            n_features=n_features,
                            generated_rows=generated_rows,
                            repeat=repeat,
                            seed=args.seed,
                        )
                    )

    summary = summarize(results)
    print()
    print_summary(summary)
    if args.output_json:
        payload = {
            "results": [asdict(result) for result in results],
            "summary": summary,
        }
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
