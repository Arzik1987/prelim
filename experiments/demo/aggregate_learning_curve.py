"""Aggregate per-experiment demo CSV files into analysis tables."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import pandas as pd
MODEL_NAMES={"rf","dt","dtc","dtval","dt_pruned"}
RESULT_FIELDS=["dataset","train_size","repetition","model","accuracy","train_accuracy","cv_accuracy","fit_seconds","n_features","selected_max_features","selected_max_leaf_nodes","selected_ccp_alpha"]
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input-dir",type=Path,default=Path(__file__).resolve().parent/"learning_curve_tasks"); p.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent); args=p.parse_args(); files=sorted(args.input_dir.glob("size_*.csv"));
    if not files: raise FileNotFoundError(f"No experiment CSV files found in {args.input_dir}")
    frames=[]
    for path in files:
        try: frame=pd.read_csv(path)
        except (OSError,pd.errors.ParserError) as e: raise ValueError(f"Could not read {path}: {e}") from e
        if list(frame.columns)!=RESULT_FIELDS or len(frame)!=5 or set(frame["model"])!=MODEL_NAMES: raise ValueError(f"Invalid experiment file: {path}; expected five model rows")
        frames.append(frame)
    results=pd.concat(frames,ignore_index=True); keys=["train_size","repetition","model"]
    if results.duplicated(keys).any(): raise ValueError("Duplicate train-size/repetition/model rows found")
    manifest_path=args.output_dir/"learning_curve_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    naive=float(manifest["naive_accuracy"]); repetitions=int(manifest.get("n_repetitions",10)); dataset=manifest.get("dataset","turbine")
    naive_rows=pd.DataFrame([{"dataset":dataset,"train_size":0,"repetition":r,"model":"naive","accuracy":naive,"train_accuracy":"","cv_accuracy":"","fit_seconds":"","n_features":"","selected_max_features":"","selected_max_leaf_nodes":"","selected_ccp_alpha":""} for r in range(repetitions)])
    results=pd.concat([naive_rows,results],ignore_index=True).sort_values(["train_size","repetition","model"]); args.output_dir.mkdir(parents=True,exist_ok=True); results.to_csv(args.output_dir/"learning_curve_results.csv",index=False)
    summary=results[results.model!="naive"].groupby(["train_size","model"],as_index=False).agg(mean_accuracy=("accuracy","mean"),std_accuracy=("accuracy","std"),mean_fit_seconds=("fit_seconds","mean"),mean_cv_accuracy=("cv_accuracy","mean")); summary=pd.concat([pd.DataFrame([{"train_size":0,"model":"naive","mean_accuracy":naive}]),summary],ignore_index=True,sort=False); summary.to_csv(args.output_dir/"learning_curve_summary.csv",index=False); print(f"Aggregated {len(files)} experiment files into {args.output_dir}")
if __name__=="__main__": main()
