import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"


def _load_read_results_module(monkeypatch):
    fake_stats = types.SimpleNamespace(wilcoxon=lambda values, alternative="greater": (0.0, 0.5))
    fake_scipy = types.SimpleNamespace(stats=fake_stats)
    monkeypatch.setitem(sys.modules, "scipy", fake_scipy)
    monkeypatch.setitem(sys.modules, "scipy.stats", fake_stats)
    monkeypatch.syspath_prepend(str(EXPERIMENTS_DIR))

    module_name = "read_results_smoke_runner"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, EXPERIMENTS_DIR / "read_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_rows(delta):
    return [
        ["dt", "na", "na", 0.80, 0.70, 3, 0.1, "na", 0.70],
        ["dtc", "na", "na", 0.81, 0.71, 4, 0.1, "na", 0.71],
        ["dtval", "na", "na", 0.82, 0.72, 5, 0.1, "na", 0.72],
        ["dtb", "na", "na", 0.79, 0.69, 3, 0.1, "na", 0.69],
        ["dtcb", "na", "na", 0.80, 0.70, 4, 0.1, "na", 0.70],
        ["dtvalb", "na", "na", 0.81, 0.71, 5, 0.1, "na", 0.71],
        ["ripper", "na", "na", 0.78, 0.68, 2, 0.1, "na", 0.68],
        ["irep", "na", "na", 0.77, 0.67, 2, 0.1, "na", 0.67],
        ["primcv", "na", "na", 0.76, 0.66, 2, 0.1, "na", "na"],
        ["bicv", "na", "na", 0.75, 0.65, 2, 0.1, "na", "na"],
        ["dt", "kdebw", "rf", 0.86, 0.70 + delta, 3, 0.2, 0.80, 0.70 + delta],
        ["dtc", "kdebw", "rf", 0.87, 0.71 + delta, 4, 0.2, 0.81, 0.71 + delta],
        ["dtval", "kdebw", "rf", 0.88, 0.72 + delta, 5, 0.2, 0.82, 0.72 + delta],
        ["ripper", "kdebw", "rf", 0.84, 0.68 + delta, 2, 0.2, 0.79, 0.68 + delta],
        ["irep", "kdebw", "rf", 0.83, 0.67 + delta, 2, 0.2, 0.78, 0.67 + delta],
        ["primcv", "kdebw", "rf", 0.82, 0.66 + delta, 2, 0.2, "na", "na"],
        ["bicv", "kdebw", "rf", 0.81, 0.65 + delta, 2, 0.2, "na", "na"],
        ["dtb", "kdebw", "rfb", 0.85, 0.69 + delta, 3, 0.2, 0.80, 0.69 + delta],
        ["dtcb", "kdebw", "rfb", 0.86, 0.70 + delta, 4, 0.2, 0.81, 0.70 + delta],
        ["dtvalb", "kdebw", "rfb", 0.87, 0.71 + delta, 5, 0.2, 0.82, 0.71 + delta],
    ]


def _meta_rows():
    return [
        ["testprec", 0.50],
        ["trainprec", 0.50],
        ["rffid", 0.55],
        ["rfdtfid", 0.72],
        ["rfdtcfid", 0.73],
        ["rfdtvalfid", 0.74],
        ["rfripperfid", 0.70],
        ["rfirepfid", 0.69],
        ["rfacc", 0.83],
        ["rfbfid", 0.56],
        ["rfbdtbfid", 0.73],
        ["rfbdtcbfid", 0.74],
        ["rfbdtvalbfid", 0.75],
        ["rfbacc", 0.82],
    ]


def _write_shard(raw_dir, dataset_name, delta):
    raw_path = raw_dir / f"{dataset_name}_0_10.csv"
    meta_path = raw_dir / f"{dataset_name}_0_10_meta.csv"
    pd.DataFrame(_raw_rows(delta)).to_csv(raw_path, header=False, index=False)
    pd.DataFrame(_meta_rows()).to_csv(meta_path, header=False, index=False)


def _create_run(tmp_path):
    run_dir = tmp_path / "registry" / "runs" / "post-run"
    raw_dir = run_dir / "raw"
    derived_dir = run_dir / "derived"
    figures_dir = run_dir / "figures"
    raw_dir.mkdir(parents=True)
    derived_dir.mkdir()
    figures_dir.mkdir()
    _write_shard(raw_dir, "alpha", 0.08)
    _write_shard(raw_dir, "beta", -0.03)
    return run_dir


def test_postprocess_run_creates_derived_outputs(monkeypatch, tmp_path):
    module = _load_read_results_module(monkeypatch)
    run_dir = _create_run(tmp_path)

    outputs = module.postprocess_run(str(run_dir), draw_figures=False)

    assert (run_dir / "derived" / "res.csv").exists()
    assert (run_dir / "derived" / "pivot_wil.csv").exists()
    assert (run_dir / "derived" / "pivot_median.csv").exists()

    res = pd.read_csv(run_dir / "derived" / "res.csv")
    pivot_wil = pd.read_csv(run_dir / "derived" / "pivot_wil.csv")
    pivot_median = pd.read_csv(run_dir / "derived" / "pivot_median.csv")

    assert not res.empty
    assert "ora" in res.columns
    assert "orf" in res.columns
    assert not outputs["res_bb"].empty
    assert "BBacc" in outputs["res_bb"].columns
    assert not pivot_wil.empty
    assert not pivot_median.empty
