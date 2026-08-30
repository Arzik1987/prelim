"""Run independent turbine learning-curve experiments in parallel."""
from __future__ import annotations
import argparse, csv, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from experiments.data.loader import load_data
DATASET="turbine"; DEFAULT_REPETITIONS=10; SPLIT_SEED=2020
RF_MAX_FEATURES=[2,"sqrt",None]; DTVAL_MAX_LEAVES=[2**p for p in range(1,8)]
PRUNING_CRITERION="gini"; MAX_PRUNING_ALPHAS=12
MODEL_NAMES=("rf","dt","dtc","dtval","dt_pruned")
RESULT_FIELDS=["dataset","train_size","repetition","model","accuracy","train_accuracy","cv_accuracy","fit_seconds","n_features","selected_max_features","selected_max_leaf_nodes","selected_ccp_alpha"]
def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parent/"output"); p.add_argument("--overwrite",action="store_true"); p.add_argument("--resume",action="store_true"); p.add_argument("--step",type=int,default=50); p.add_argument("--max-train-size",type=int,default=1000); p.add_argument("--train-sizes",help="Comma-separated explicit training sizes; overrides step and max-train-size."); p.add_argument("--threads", type=int, default=8, help="Maximum concurrent worker threads (default: 8).") ; p.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS, help="Number of splits per train size (default: 10)."); return p.parse_args()
def sizes_for(args):
    if args.train_sizes is not None:
        try: sizes=tuple(sorted({int(v.strip()) for v in args.train_sizes.split(",") if v.strip()}))
        except ValueError as e: raise ValueError("--train-sizes must be a comma-separated list of integers") from e
        if not sizes or any(s<=0 for s in sizes): raise ValueError("--train-sizes must contain positive integers")
        return sizes
    if args.step<=0 or args.max_train_size<args.step: raise ValueError("Require 0 < --step <= --max-train-size")
    return tuple(range(args.step,args.max_train_size+1,args.step))
def make_partitions(y, size, repetitions):
    if size>len(y): raise ValueError(f"Training size {size} exceeds the dataset size {len(y)}")
    for attempt in range(1000):
        rng=np.random.RandomState(SPLIT_SEED+size+attempt); perm=rng.permutation(len(y)); starts=np.linspace(0,len(y)-size,num=repetitions,endpoint=True,dtype=int); parts=[perm[s:s+size].copy() for s in starts]
        if all(np.unique(y[i]).size==2 for i in parts): return parts
    raise RuntimeError(f"Could not construct valid partitions for size {size}")
def prepare_split(X,y,indices):
    mask=np.ones(len(y),dtype=bool); mask[indices]=False; Xtr,ytr=X[indices],y[indices]; Xte,yte=X[mask],y[mask]; variable=Xtr.max(axis=0)!=Xtr.min(axis=0); Xtr,Xte=Xtr[:,variable],Xte[:,variable]; scaler=StandardScaler(); return scaler.fit_transform(Xtr),ytr,scaler.transform(Xte),yte,variable
def fit_and_score(model,Xtr,ytr,Xte,yte):
    start=time.perf_counter(); model.fit(Xtr,ytr); return {"accuracy":float(model.score(Xte,yte)),"train_accuracy":float(model.score(Xtr,ytr)),"cv_accuracy":"","fit_seconds":time.perf_counter()-start,"selected_max_features":"","selected_max_leaf_nodes":"","selected_ccp_alpha":""}
def fit_rf(Xtr,ytr,Xte,yte):
    search=GridSearchCV(RandomForestClassifier(random_state=SPLIT_SEED,n_jobs=1),{"max_features":RF_MAX_FEATURES},cv=5,n_jobs=1); r=fit_and_score(search,Xtr,ytr,Xte,yte); r["cv_accuracy"]=float(search.best_score_); r["selected_max_features"]=str(search.best_params_["max_features"]); return r
def fit_dtval(Xtr,ytr,Xte,yte):
    search=GridSearchCV(DecisionTreeClassifier(),{"max_leaf_nodes":DTVAL_MAX_LEAVES},cv=5,n_jobs=1); r=fit_and_score(search,Xtr,ytr,Xte,yte); r["cv_accuracy"]=float(search.best_score_); r["selected_max_leaf_nodes"]=int(search.best_params_["max_leaf_nodes"]); return r
def pruning_alpha_candidates(Xtr,ytr):
    alphas=np.unique(DecisionTreeClassifier(criterion=PRUNING_CRITERION).cost_complexity_pruning_path(Xtr,ytr).ccp_alphas[:-1]);
    if len(alphas)<=MAX_PRUNING_ALPHAS: return alphas
    return alphas[np.unique(np.linspace(0,len(alphas)-1,MAX_PRUNING_ALPHAS,dtype=int))]
def fit_dt_pruned(Xtr,ytr,Xte,yte):
    search=GridSearchCV(DecisionTreeClassifier(),{"criterion":[PRUNING_CRITERION],"ccp_alpha":pruning_alpha_candidates(Xtr,ytr)},cv=5,n_jobs=1); r=fit_and_score(search,Xtr,ytr,Xte,yte); r["cv_accuracy"]=float(search.best_score_); r["selected_ccp_alpha"]=float(search.best_params_["ccp_alpha"]); return r
def task_path(directory,size,repetition): return directory/f"size_{size:05d}_split_{repetition:03d}.csv"
def complete(path,size,repetition):
    if not path.exists(): return False
    try:
        with path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
        return len(rows)==5 and {r["model"] for r in rows}==set(MODEL_NAMES) and all(int(r["train_size"])==size and int(r["repetition"])==repetition for r in rows)
    except (OSError,KeyError,TypeError,ValueError,csv.Error): return False
def write_task(path,rows):
    tmp=path.with_name(f".{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=RESULT_FIELDS); w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
def run_task(X,y,size,repetition,indices,path):
    Xtr,ytr,Xte,yte,variable=prepare_split(X,y,indices); models={"rf":fit_rf(Xtr,ytr,Xte,yte),"dt":fit_and_score(DecisionTreeClassifier(min_samples_split=10),Xtr,ytr,Xte,yte),"dtc":fit_and_score(DecisionTreeClassifier(max_leaf_nodes=8),Xtr,ytr,Xte,yte),"dtval":fit_dtval(Xtr,ytr,Xte,yte),"dt_pruned":fit_dt_pruned(Xtr,ytr,Xte,yte)}; rows=[]
    for model,r in models.items(): rows.append({"dataset":DATASET,"train_size":size,"repetition":repetition,"model":model,"accuracy":r["accuracy"],"train_accuracy":r["train_accuracy"],"cv_accuracy":r["cv_accuracy"],"fit_seconds":r["fit_seconds"],"n_features":int(variable.sum()),"selected_max_features":r["selected_max_features"],"selected_max_leaf_nodes":r["selected_max_leaf_nodes"],"selected_ccp_alpha":r["selected_ccp_alpha"]})
    write_task(path,rows); return path
def print_progress(completed, total):
    width = 30
    filled = int(width * completed / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r[{bar}] {completed}/{total} experiments")
    sys.stdout.flush()
    if completed == total:
        sys.stdout.write("\n")
def main():
    args=parse_args();
    if args.threads<=0: raise ValueError("--threads must be positive")
    if args.repetitions<=0: raise ValueError("--repetitions must be positive")
    sizes=sizes_for(args); output=args.output_dir.resolve(); tasks_dir=output/"experiments"; output.mkdir(parents=True,exist_ok=True); tasks_dir.mkdir(parents=True,exist_ok=True)
    if list(tasks_dir.glob("size_*.csv")) and not args.overwrite and not args.resume: raise FileExistsError("Experiment files exist; use --overwrite or --resume")
    if args.overwrite:
        for size in sizes:
            for path in tasks_dir.glob(f"size_{size:05d}_split_*.csv"):
                path.unlink()
    X,y=load_data(DATASET); manifest={"dataset":DATASET,"n_rows":int(len(y)),"train_sizes":list(sizes),"n_repetitions":args.repetitions,"split_seed":SPLIT_SEED,"partition_policy":"requested shuffled training sets per size; disjoint when possible, otherwise evenly spaced windows; test is complement","task_format":"one CSV per train size and repetition, containing five model rows","naive_accuracy":float(max(np.mean(y),1-np.mean(y))),"models":{"rf":{"max_features":RF_MAX_FEATURES,"cv":5},"dt":{"min_samples_split":10},"dtc":{"max_leaf_nodes":8},"dtval":{"max_leaf_nodes":DTVAL_MAX_LEAVES,"cv":5},"dt_pruned":{"criterion":PRUNING_CRITERION,"ccp_alpha":"up to 12 pruning-path values","cv":5}}}; (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    tasks=[]
    for size in sizes:
        for repetition,indices in enumerate(make_partitions(y,size,args.repetitions)):
            path=task_path(tasks_dir,size,repetition)
            if args.resume and complete(path,size,repetition): continue
            tasks.append((size,repetition,indices,path))
    total = len(tasks)
    print(f"Running {total} experiments with {args.threads} worker threads")
    with ThreadPoolExecutor(max_workers=args.threads,thread_name_prefix="demo") as pool:
        futures=[pool.submit(run_task,X,y,size,rep,indices,path) for size,rep,indices,path in tasks]
        for n,future in enumerate(as_completed(futures),1):
            future.result()
            print_progress(n, total)
    if total == 0:
        print("All requested experiments are already complete")
    print(f"Wrote experiment files to {tasks_dir}")
if __name__=="__main__": main()
