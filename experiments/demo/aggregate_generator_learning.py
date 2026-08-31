"""Aggregate generator-learning repetition files."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
FIELDS=["train_size","repetition","stage","generator","gen_size","model","accuracy","train_accuracy","fit_seconds","selected_max_leaf_nodes"]
def main():
    here=Path(__file__).resolve().parent; p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input-dir",type=Path,default=here/"generator_learning_tasks"); p.add_argument("--output",type=Path,default=here/"generator_learning_results.csv"); a=p.parse_args(); files=sorted(a.input_dir.glob("repetition_*.csv"))
    if not files: raise FileNotFoundError(f"No repetition CSV files found in {a.input_dir}")
    frames=[]
    for f in files:
        frame=pd.read_csv(f)
        if list(frame.columns)!=FIELDS: raise ValueError(f"Unexpected columns in {f}")
        frames.append(frame)
    results=pd.concat(frames,ignore_index=True); results["accuracy"]=pd.to_numeric(results["accuracy"]); results["train_size"]=pd.to_numeric(results["train_size"]); results["gen_size"]=pd.to_numeric(results["gen_size"])
    keys=["train_size","stage","generator","gen_size","model"]
    aggregated=results.groupby(keys,as_index=False).agg(mean_accuracy=("accuracy","mean"),std_accuracy=("accuracy","std"),repetitions=("accuracy","count")).sort_values(keys); a.output.parent.mkdir(parents=True,exist_ok=True); aggregated.to_csv(a.output,index=False); print(f"Wrote {len(aggregated)} aggregated groups to {a.output}")
if __name__=="__main__": main()