"""Bit-to-bit gridworld (IJCAI'05 Sec. 5): egocentric agent, 1-bit observation (wall ahead),
actions forward / turn-right, 26 open cells x 4 headings = 104 states.
Sweeps: question depth x history, lambda, weight sets vs history features.  Reports oracle RMSE
over all nodes and per question depth."""
from common import *
from tdnet import *
import sys

B, SEED = 32, 0
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
log = logger()
env = gridworld(MAPS["ijcai26"], name="bit2bit104")
results = {}


def run(q, lam, D, hist_f=0, hist_w=1, alpha=0.5, steps=STEPS, tag="", model="linear", hidden=32, opt="sgd"):
    cfg = LearnerConfig(alpha=alpha, lam=lam, hist_f=hist_f, hist_w=hist_w, max_depth=D, model=model, hidden=hidden, optimizer=opt)
    ex, h = run_experiment(env, q, cfg, batch=B, steps=steps, chunk=20_000, seed=SEED, log=None)
    depths = q.depths()
    by_depth = {int(d): float(np.sqrt(h["node_mse"][-1][depths == d].mean())) for d in range(1, depths.max() + 1)}
    log(f"{tag:34s} N={q.n_active:3d} lam={lam} hf={hist_f} hw={hist_w} {model}: RMSE {h['rmse'][-1]:.4f} "
        f"1-step {h['err1'][-1]:.4f} wrong {h['wrong'][-1]:.4f} by depth {' '.join(f'{d}:{v:.3f}' for d, v in by_depth.items())}")
    h["by_depth"] = by_depth
    return h


# 1. depth x history (lambda = 1; weight set per (a,o) plus k-step history features)
curves = []
for depth in [2, 3, 4, 5]:
    for hf in [0, 2, 3]:
        h = run(symmetric_tree(2, depth), 1.0, depth, hist_f=hf, tag=f"sym depth {depth}")
        results[f"d{depth}_hf{hf}_lam1"] = h
        if hf in (0, 3):
            curves.append((f"depth {depth}, history {hf}", h["step"], h["rmse"]))
plot_curves(curves, "Bit-to-bit gridworld (104 states): depth x history, lambda=1", "grid_depth_history.png", ymax=0.5)

# 2. lambda at depth 4, history 2
curves = []
for lam in [0.0, 0.5, 1.0]:
    h = run(symmetric_tree(2, 4), lam, 4, hist_f=2, tag="sym depth 4")
    results[f"d4_hf2_lam{lam}"] = h; curves.append((f"lambda={lam}", h["step"], h["rmse"]))
plot_curves(curves, "Bit-to-bit gridworld: effect of lambda (depth 4, history 2)", "grid_lambda.png", ymax=0.5)

# 3. weight sets (C++ default) vs history features, and an MLP answer network
curves = []
h = results["d4_hf0_lam1"]; curves.append(("weight set per (a,o)", h["step"], h["rmse"]))
h = results["d4_hf2_lam1"]; curves.append(("weight set per (a,o) + history-2 features", h["step"], h["rmse"]))
h = run(symmetric_tree(2, 4), 1.0, 4, hist_f=0, hist_w=2, tag="sym depth 4 weight-set 2-hist"); results["d4_hw2"] = h
curves.append(("weight set per 2-step history (16 sets)", h["step"], h["rmse"]))
h = run(symmetric_tree(2, 4), 1.0, 4, hist_f=2, hist_w=0, model="mlp", hidden=64, alpha=0.1, tag="sym depth 4 MLP-64"); results["d4_hf2_mlp"] = h
curves.append(("history-2 features, MLP-64, no weight sets", h["step"], h["rmse"]))
h = run(symmetric_tree(2, 4), 1.0, 4, hist_f=2, hist_w=0, model="mlp", hidden=64, alpha=1e-3, opt="adam", tag="sym depth 4 MLP-64 adam"); results["d4_hf2_mlp_adam"] = h
curves.append(("history-2 features, MLP-64 + Adam", h["step"], h["rmse"]))
plot_curves(curves, "Bit-to-bit gridworld: answer-network variants (depth 4, lambda=1)", "grid_answer_nets.png", ymax=0.5)

save_json("exp_grid", {k: {kk: vv for kk, vv in v.items() if kk != "node_mse"} for k, v in results.items()})
log("done")
