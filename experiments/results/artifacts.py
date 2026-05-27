import csv
import json
import os
from dataclasses import asdict, dataclass
from itertools import product


RESULT_FIELDS = ("alg", "gen", "met", "tra", "tes", "nle", "tme", "fid", "bac")
META_FIELDS = ("alg", "val")


@dataclass(frozen=True)
class ExperimentResult:
    alg: object
    gen: object
    met: object
    tra: object
    tes: object
    nle: object
    tme: object
    fid: object
    bac: object


@dataclass(frozen=True)
class ExperimentMeta:
    alg: object
    val: object


def result_prefix(config, dataset_name, split_index, dataset_size):
    return os.path.join(config.raw_dir, "%s_%s_%s" % (dataset_name, split_index, dataset_size))


def result_paths(config, dataset_name, split_index, dataset_size):
    prefix = result_prefix(config, dataset_name, split_index, dataset_size)
    return {
        "raw": prefix + ".csv",
        "meta": prefix + "_meta.csv",
        "zeros": prefix + "_zeros.csv",
    }


def shard_is_complete(config, dataset_name, split_index, dataset_size):
    paths = result_paths(config, dataset_name, split_index, dataset_size)
    return os.path.exists(paths["zeros"]) or (
        os.path.exists(paths["raw"]) and os.path.exists(paths["meta"])
    )


def write_result(handle, model_name, gen_name, meta_name, sctrain, sctest, complexity, elapsed, fidelity, bactest):
    row = ExperimentResult(
        alg=model_name,
        gen=gen_name,
        met=meta_name,
        tra=sctrain,
        tes=sctest,
        nle=complexity,
        tme=elapsed,
        fid=fidelity,
        bac=bactest,
    )
    writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, lineterminator="\n")
    writer.writerow(asdict(row))
    handle.flush()


def write_meta(handle, key, value):
    row = ExperimentMeta(alg=key, val=value)
    writer = csv.DictWriter(handle, fieldnames=META_FIELDS, lineterminator="\n")
    writer.writerow(asdict(row))
    handle.flush()


def iter_experiment_args(config):
    return product(config.datasets, config.dataset_sizes, config.split_indices)


def write_manifest(config, status, summary=None):
    manifest = config.to_manifest()
    manifest["status"] = status
    if summary is not None:
        manifest["summary"] = summary
    with open(config.manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def summarize_results(result_list):
    summary = {
        "completed": 0,
        "skipped": 0,
        "zero_class": 0,
        "failed": 0,
    }
    for status, _, _, _, _ in result_list:
        if status == "completed":
            summary["completed"] += 1
        elif status == "skipped":
            summary["skipped"] += 1
        elif status == "zero-class":
            summary["zero_class"] += 1
        else:
            summary["failed"] += 1
    summary["total"] = len(result_list)
    return summary
