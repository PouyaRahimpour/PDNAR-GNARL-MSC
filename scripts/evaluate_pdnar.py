#!/usr/bin/env python3
"""Evaluate an original PDNAR Lightning checkpoint with GNARL metrics."""
from __future__ import annotations
import argparse,csv,pickle,sys,time
from pathlib import Path
import torch
from torch_geometric.data import Batch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.gnarl.evaluation.evaluate import classical_pd_cost
from src.gnarl.evaluation.evaluate import _mean_std, write_results, plot_results
from src.model.graph_executor import GraphNeuralExecutor

SIZES=[16,32,64,128,256,512,1024]

def load_checkpoint(path,device,hidden_dim=32,eps=False):
    model=GraphNeuralExecutor(hidden_dim=hidden_dim,eps=eps).to(device)
    ckpt=torch.load(path,map_location=device)
    sd=ckpt.get("state_dict",ckpt)
    cleaned={k[len("model."):]:v for k,v in sd.items() if k.startswith("model.")}
    if not cleaned: cleaned=sd
    missing,unexpected=model.load_state_dict(cleaned,strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Could not load PDNAR checkpoint. Missing={missing}, unexpected={unexpected}")
    model.eval(); return model

def load_pickle(path):
    with open(path,"rb") as f:return pickle.load(f)

def evaluate_pdnar(model,data,device,out,title):
    objectives=[]; opt_ratios=[]; pd_ratios=[]; steps=[]; rows=[]; total=len(data); t0=time.perf_counter()
    for i,record in enumerate(data,1):
        start=time.perf_counter()
        batch=Batch.from_data_list([record],follow_batch=["x","y"]).to(device)
        result=model.test(batch)
        pred=result["pred_set"].squeeze(-1).bool()
        weights=record.x[:,0].flatten().float().to(device)
        objective=float(weights[pred].sum().item())
        optimal=float(record.primal_optimal_weight.flatten()[0].item())
        classical=classical_pd_cost(record)
        runtime=time.perf_counter()-start
        # PDNAR's own output is the numerator; stored x_mask is the classical PD solution.
        oratio=objective/optimal; pratio=objective/classical
        # test() performs its corrective stage until coverage, so this is the actual episode length unavailable from its API.
        # Count selected sets as a robust model-independent step metric.
        step=int(pred.sum().item())
        objectives.append(objective); opt_ratios.append(oratio); pd_ratios.append(pratio); steps.append(step)
        rows.append({"instance":i,"objective":objective,"optimal":optimal,"classical_pd":classical,"optimal_ratio":oratio,"classical_pd_ratio":pratio,"steps":step,"runtime_seconds":runtime})
        elapsed=time.perf_counter()-t0; eta=elapsed/i*(total-i)
        print(f"\rTEST {i:4d}/{total:<4d} | n={record.x.shape[0]:4d} | obj={objective:9.3f} | opt={oratio:7.4f} | pd={pratio:7.4f} | t={runtime:7.3f}s | ETA={eta:7.1f}s",end="",flush=True)
    print(); write_results(rows,out); plot_results(rows,out,title)
    vals={}
    for name,xs in (("objective",objectives),("optimal_ratio",opt_ratios),("classical_pd_ratio",pd_ratios),("steps",steps)): vals[f"mean_{name}"],vals[f"std_{name}"]=_mean_std(xs)
    return len(data),vals

def main():
    p=argparse.ArgumentParser(); p.add_argument("checkpoint"); p.add_argument("--device",default="cpu"); p.add_argument("--test-root",default="dataset/set_cover"); p.add_argument("--output-dir",default="runs/evaluation_pdnar"); p.add_argument("--sizes",nargs="+",type=int,default=SIZES); p.add_argument("--hidden-dim",type=int,default=32); p.add_argument("--eps",action="store_true"); args=p.parse_args()
    model=load_checkpoint(args.checkpoint,args.device,args.hidden_dim,args.eps); out=Path(args.output_dir); summary=[]
    for n in args.sizes:
        path=Path(args.test_root)/f"test_100_{n}.pkl"
        if not path.exists(): raise FileNotFoundError(path)
        data=load_pickle(path); result_dir=out/f"n_{n}"; count,s=evaluate_pdnar(model,data,args.device,result_dir,f"PDNAR-MSC n={n}")
        row={"n":n,"instances":count,"objective_mean":s["mean_objective"],"objective_std":s["std_objective"],"optimal_ratio_mean":s["mean_optimal_ratio"],"optimal_ratio_std":s["std_optimal_ratio"],"classical_pd_ratio_mean":s["mean_classical_pd_ratio"],"classical_pd_ratio_std":s["std_classical_pd_ratio"],"steps_mean":s["mean_steps"],"steps_std":s["std_steps"]}; summary.append(row)
        print(f"n={n:4d} | obj={row['objective_mean']:.4f}±{row['objective_std']:.4f} | opt={row['optimal_ratio_mean']:.6f}±{row['optimal_ratio_std']:.6f} | pd={row['classical_pd_ratio_mean']:.6f}±{row['classical_pd_ratio_std']:.6f}")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"summary.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=summary[0].keys());w.writeheader();w.writerows(summary)
    print(f"Saved PDNAR evaluation to {out}")
if __name__=="__main__":main()
