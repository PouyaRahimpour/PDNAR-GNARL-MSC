#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gnarl.evaluation.evaluate import evaluate_gnarl
from src.gnarl.models.gnarl_msc import GNARLMSC
from src.gnarl.training.data import load_msc_split

SIZES = [16, 32, 64, 128, 256, 512, 1024]

def load_checkpoint(checkpoint, device):
    model = GNARLMSC(hidden_dim=64, message_passing_rounds=4, pooling="mean")
    state = torch.load(checkpoint, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict, strict=True); model.eval(); return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint"); p.add_argument("--device", default="cpu")
    p.add_argument("--test-root", default="dataset/set_cover"); p.add_argument("--output-dir", default="runs/evaluation")
    p.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    args = p.parse_args(); model = load_checkpoint(args.checkpoint, args.device)
    out = Path(args.output_dir); summary=[]
    for size in args.sizes:
        data=load_msc_split("test",100,size,args.test_root); result_dir=out/f"n_{size}"
        r=evaluate_gnarl(model,data,args.device,result_dir)
        row={"n":size,"instances":r.instances,"objective_mean":r.mean_objective,"objective_std":r.std_objective,
             "optimal_ratio_mean":r.mean_optimal_ratio,"optimal_ratio_std":r.std_optimal_ratio,
             "classical_pd_ratio_mean":r.mean_classical_pd_ratio,"classical_pd_ratio_std":r.std_classical_pd_ratio,
             "steps_mean":r.mean_steps,"steps_std":r.std_steps}
        summary.append(row)
        print(f"n={size:4d} | obj={r.mean_objective:.4f}±{r.std_objective:.4f} | opt={r.mean_optimal_ratio:.6f}±{r.std_optimal_ratio:.6f} | pd={r.mean_classical_pd_ratio:.6f}±{r.std_classical_pd_ratio:.6f} | steps={r.mean_steps:.2f}±{r.std_steps:.2f}")
    out.mkdir(parents=True,exist_ok=True)
    with (out/"summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    print(f"Saved evaluation results to {out}")
if __name__=="__main__": main()
