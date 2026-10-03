"""Turn results/data/*.json into markdown tables for the report."""
from common import *
import json, os

def first_below(h, thr=0.05):
    return next((s for s, r in zip(h["step"], h["rmse"]) if r < thr), None)

out = []
def sec(title): out.append(f"\n### {title}\n")
def table(header, rows):
    out.append("| " + " | ".join(header) + " |"); out.append("|" + "---|" * len(header))
    for r in rows: out.append("| " + " | ".join(str(x) for x in r) + " |")

def fmt(x): return "-" if x is None else (f"{x:,}" if isinstance(x, int) else f"{x:.4f}")

if os.path.exists(os.path.join(DATA, "exp_paper.json")):
    R = load_json("exp_paper")
    sec("Paper reproduction (exp_paper)")
    rows = []
    for k, h in R.items():
        rows.append([k, fmt(h["rmse"][-1]), fmt(h["err1"][-1]), fmt(first_below(h)), f"{h['step'][-1]:,}"])
    table(["run", "final oracle RMSE", "final 1-step |err|", "steps to RMSE<.05", "steps/stream"], rows)

if os.path.exists(os.path.join(DATA, "exp_history_baseline.json")):
    R = load_json("exp_history_baseline")
    sec("History-only baselines (exp_history_baseline)")
    table(["world / k", "final next-obs oracle RMSE", "final |err|"], [[k, fmt(h["rmse"][-1]), fmt(h["err1"][-1])] for k, h in R.items()])

for name, title in [("exp_grid", "Bit-to-bit gridworld (exp_grid)"), ("exp_big", "Larger / richer / stochastic worlds (exp_big)")]:
    if os.path.exists(os.path.join(DATA, name + ".json")):
        R = load_json(name)
        sec(title)
        rows = []
        for k, h in R.items():
            bd = h.get("by_depth", {})
            rows.append([k, h.get("env", ""), h.get("N", ""), fmt(h["rmse"][-1]), fmt(h["err1"][-1]),
                         " ".join(f"{d}:{v:.3f}" for d, v in bd.items())])
        table(["run", "env", "nodes", "final oracle RMSE", "final 1-step |err|", "RMSE by question depth"], rows)

if os.path.exists(os.path.join(DATA, "exp_grow.json")):
    R = load_json("exp_grow")
    sec("Growing vs fixed (exp_grow)")
    rows = []
    for k, h in R.items():
        rows.append([k, h["n"][-1], fmt(h["rmse"][-1]), fmt(h["d1"][-1]), f"{h['step'][-1]:,}"])
    table(["run", "final nodes", "final all-node RMSE", "final next-obs RMSE", "steps/stream"], rows)
    for k, h in R.items():
        if "structure" in h:
            out.append(f"\n**{k}: final question network**\n```\n{h['structure']}\n```")
            out.append("rounds: " + "; ".join(f"{r['step']//1000}k:N={r['n_active']}(+{len(r['new'])},-{len(r['pruned'])})" for r in h["rounds"]))

text = "\n".join(out)
open(os.path.join(DATA, "summary.md"), "w").write(text)
print(text)
