"""Follow-up: the larger fixed trees diverge at alpha = 0.5 with 32 summed streams (RMSE > 0.4 for
the deepest nodes).  Re-run the big configurations on the bit-to-bit world with alpha = 0.1 / 0.05."""
from common import *
from tdnet import *
import sys

B, SEED = 32, 0
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
log = logger()
env = gridworld(MAPS["ijcai26"], name="bit2bit104")
results = {}


def run(q, lam, D, hist_f=0, hist_w=1, alpha=0.1, tag="", model="linear", hidden=64, opt="sgd"):
    cfg = LearnerConfig(alpha=alpha, lam=lam, hist_f=hist_f, hist_w=hist_w, max_depth=D, model=model, hidden=hidden, optimizer=opt)
    ex, h = run_experiment(env, q, cfg, batch=B, steps=STEPS, chunk=20_000, seed=SEED, log=None)
    depths = q.depths()
    by_depth = {int(d): float(np.sqrt(h["node_mse"][-1][depths == d].mean())) for d in range(1, depths.max() + 1)}
    log(f"{tag:30s} N={q.n_active:3d} lam={lam} hf={hist_f} hw={hist_w} alpha={alpha} {model}/{opt}: RMSE {h['rmse'][-1]:.4f} "
        f"1-step {h['err1'][-1]:.4f} by depth {' '.join(f'{d}:{v:.3f}' for d, v in by_depth.items())}")
    h["by_depth"] = by_depth; h["N"] = q.n_active
    return h


curves = []
for depth, hf, alpha in [(4, 0, 0.1), (5, 0, 0.1), (5, 3, 0.1), (6, 0, 0.1), (6, 0, 0.05), (6, 3, 0.05)]:
    h = run(symmetric_tree(2, depth), 1.0, depth, hist_f=hf, alpha=alpha, tag=f"sym depth {depth}")
    results[f"d{depth}_hf{hf}_a{alpha}"] = h; curves.append((f"depth {depth}, history {hf}, alpha {alpha}", h["step"], h["rmse"]))
plot_curves(curves, "Bit-to-bit gridworld: deeper fixed trees at smaller step sizes", "grid_alpha.png", ymax=0.5)

curves = []
h = run(symmetric_tree(2, 4), 0.0, 4, hist_f=2, alpha=0.1, tag="sym depth 4 TD(0)"); results["d4_hf2_lam0_a0.1"] = h
curves.append(("lambda 0", h["step"], h["rmse"]))
h = run(symmetric_tree(2, 4), 0.5, 4, hist_f=2, alpha=0.1, tag="sym depth 4 TD(.5)"); results["d4_hf2_lam0.5_a0.1"] = h
curves.append(("lambda 0.5", h["step"], h["rmse"]))
h = run(symmetric_tree(2, 4), 1.0, 4, hist_f=2, alpha=0.1, tag="sym depth 4 TD(1)"); results["d4_hf2_lam1_a0.1"] = h
curves.append(("lambda 1", h["step"], h["rmse"]))
plot_curves(curves, "Bit-to-bit gridworld: lambda at alpha 0.1 (depth 4, history 2)", "grid_lambda_alpha0.1.png", ymax=0.5)

for depth in (4, 5):
    h = run(symmetric_tree(2, depth), 1.0, depth, hist_f=2, hist_w=0, model="mlp", hidden=128, alpha=1e-3, opt="adam", tag=f"sym depth {depth} MLP-128 adam")
    results[f"d{depth}_hf2_mlp128_adam"] = h
save_json("exp_alpha", {k: {kk: vv for kk, vv in v.items() if kk != "node_mse"} for k, v in results.items()})
log("done")
