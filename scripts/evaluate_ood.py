#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,pickle,statistics,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.gnarl.evaluation.evaluate import evaluate_gnarl
from src.gnarl.models.gnarl_msc import GNARLMSC

def load_checkpoint(checkpoint,device):
    model=GNARLMSC(hidden_dim=64,message_passing_rounds=4,pooling="mean")
    state=torch.load(checkpoint,map_location=device); sd=state["model"] if isinstance(state,dict) and "model" in state else state
    model.load_state_dict(sd,strict=True); model.eval(); return model

def load_pickle(path):
    with path.open("rb") as f:return pickle.load(f)
    
SIZES = [16,128,512,1024]

def main():
    p=argparse.ArgumentParser(); p.add_argument("checkpoint"); p.add_argument("--dataset-root",default="dataset/pdnar_ood"); p.add_argument("--output-dir",default="runs/evaluation_ood"); p.add_argument("--device",default="cpu"); p.add_argument("--b",nargs="+",type=int,default=[3,8]); p.add_argument("--sizes",nargs="+",type=int,default=SIZES); p.add_argument("--seeds",type=int,default=10); args=p.parse_args()
    model=load_checkpoint(args.checkpoint,args.device); root=Path(args.dataset_root); out=Path(args.output_dir); seed_rows=[]; aggregate=[]
    for b in args.b:
      for n in args.sizes:
        metrics=[]
        for seed in range(args.seeds):
          path=root/f"b_{b}"/f"n_{n}"/f"test_50_seed_{seed}.pkl"
          if not path.exists(): raise FileNotFoundError(f"Missing OOD dataset: {path}")
          r=evaluate_gnarl(model,load_pickle(path),args.device,out/f"b_{b}"/f"n_{n}"/f"seed_{seed}",title_prefix=f"GNARL-MSC OOD b={b}, n={n}, seed={seed}")
          metrics.append(r)
          seed_rows.append({"b":b,"n":n,"seed":seed,"instances":r.instances,"objective_mean":r.mean_objective,"objective_std":r.std_objective,"optimal_ratio_mean":r.mean_optimal_ratio,"optimal_ratio_std":r.std_optimal_ratio,"classical_pd_ratio_mean":r.mean_classical_pd_ratio,"classical_pd_ratio_std":r.std_classical_pd_ratio,"steps_mean":r.mean_steps,"steps_std":r.std_steps})
        # Aggregate over ALL individual test instances across all seeds, not merely the ten seed means.
        total=sum(m.instances for m in metrics)
        def pooled(field):
          means=[getattr(m,"mean_"+field) for m in metrics]; vars_=[getattr(m,"std_"+field)**2 for m in metrics]; ns=[m.instances for m in metrics]
          mean=sum(n*x for n,x in zip(ns,means))/total
          ss=sum(max(n-1,0)*v+n*(x-mean)**2 for n,v,x in zip(ns,vars_,means))
          return mean,(ss/(total-1))**0.5 if total>1 else 0.0
        vals={};
        for field in ("objective","optimal_ratio","classical_pd_ratio","steps"): vals[field]=pooled(field)
        row={"b":b,"n":n,"seeds":len(metrics),"instances":total}
        for field,(mean,std) in vals.items(): row[field+"_mean"]=mean; row[field+"_std"]=std
        aggregate.append(row)
        print(f"OOD aggregate b={b:2d} n={n:4d} | opt={row['optimal_ratio_mean']:.6f}±{row['optimal_ratio_std']:.6f} | pd={row['classical_pd_ratio_mean']:.6f}±{row['classical_pd_ratio_std']:.6f}")
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in (("seed_summary.csv",seed_rows),("summary.csv",aggregate)):
      with (out/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(f"Saved OOD results to {out}")
if __name__=="__main__":main()
