#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,pickle,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_pdnar import load_checkpoint, load_pickle, evaluate_pdnar
SIZES = [16,128,512,1024]
def main():
    p=argparse.ArgumentParser(); p.add_argument("checkpoint"); p.add_argument("--dataset-root",default="dataset/pdnar_ood"); p.add_argument("--output-dir",default="runs/evaluation_pdnar_ood"); p.add_argument("--device",default="cpu"); p.add_argument("--b",nargs="+",type=int,default=[3,8]); p.add_argument("--sizes",nargs="+",type=int,default=SIZES); p.add_argument("--seeds",type=int,default=10); p.add_argument("--hidden-dim",type=int,default=32); p.add_argument("--eps",action="store_true"); args=p.parse_args()
    model=load_checkpoint(args.checkpoint,args.device,args.hidden_dim,args.eps); root=Path(args.dataset_root); out=Path(args.output_dir); seed_rows=[]; aggregate=[]
    for b in args.b:
      for n in args.sizes:
        all_rows=[]
        for seed in range(args.seeds):
          path=root/f"b_{b}"/f"n_{n}"/f"test_50_seed_{seed}.pkl"
          if not path.exists(): raise FileNotFoundError(path)
          seed_out=out/f"b_{b}"/f"n_{n}"/f"seed_{seed}"; count,s=evaluate_pdnar(model,load_pickle(path),args.device,seed_out,f"PDNAR-MSC OOD b={b}, n={n}, seed={seed}")
          row={"b":b,"n":n,"seed":seed,"instances":count,**{k.replace("mean_"," ").strip():v for k,v in s.items()}}
          # Keep explicit stable column names.
          row={"b":b,"n":n,"seed":seed,"instances":count,"objective_mean":s["mean_objective"],"objective_std":s["std_objective"],"optimal_ratio_mean":s["mean_optimal_ratio"],"optimal_ratio_std":s["std_optimal_ratio"],"classical_pd_ratio_mean":s["mean_classical_pd_ratio"],"classical_pd_ratio_std":s["std_classical_pd_ratio"],"steps_mean":s["mean_steps"],"steps_std":s["std_steps"]}
          seed_rows.append(row); all_rows.append((count,s))
        total=sum(c for c,_ in all_rows)
        agg={"b":b,"n":n,"seeds":len(all_rows),"instances":total}
        for field in ("objective","optimal_ratio","classical_pd_ratio","steps"):
          means=[s[f"mean_{field}"] for _,s in all_rows]; stds=[s[f"std_{field}"] for _,s in all_rows]; ns=[c for c,_ in all_rows]; mean=sum(c*x for c,x in zip(ns,means))/total; ss=sum(max(c-1,0)*sd*sd+c*(x-mean)**2 for c,sd,x in zip(ns,stds,means)); agg[f"{field}_mean"]=mean; agg[f"{field}_std"]=(ss/(total-1))**0.5 if total>1 else 0.0
        aggregate.append(agg); print(f"OOD aggregate b={b:2d} n={n:4d} | opt={agg['optimal_ratio_mean']:.6f}±{agg['optimal_ratio_std']:.6f} | pd={agg['classical_pd_ratio_mean']:.6f}±{agg['classical_pd_ratio_std']:.6f}")
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in (("seed_summary.csv",seed_rows),("summary.csv",aggregate)):
      with (out/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
if __name__=="__main__":main()
