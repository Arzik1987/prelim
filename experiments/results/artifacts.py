import json
import os
from itertools import product


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
    handle.write(
        model_name + ",%s,%s,%s,%s,%s,%s,%s,%s\n"
        % (gen_name, meta_name, sctrain, sctest, complexity, elapsed, fidelity, bactest)
    )


def write_meta(handle, key, value):
    handle.write("%s,%s\n" % (key, value))


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
