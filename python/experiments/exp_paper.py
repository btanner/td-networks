"""Reproduce the headline results of Sutton & Tanner (NIPS'04), Tanner & Sutton (ICML'05, IJCAI'05):
  (a) 5-state ring, symmetric depth-3 net: all lambda learn, larger lambda faster
  (b) 8-state ring, sparse 16-node net: TD(0) fails, lambda>0 succeeds
  (c) 6-state cycle, 5-chain: TD(0) fails without history; lambda>0 or history fixes it
NOTE: the papers describe x_t = (bias, one-hot(a_{t-1}, o_t), y_{t-1}) with a single weight matrix.
That cannot represent action-gated updates (see representability check in the report), and the
original C++ actually used a separate weight set per (a_{t-1}, o_t) (hist_w=1).  We use that as the
baseline and include the single-matrix variant as a documented negative result."""
from common import *
from tdnet import *

B, SEED = 16, 0
log = logger()
results = {}


def run(env, q, lam, steps, D, hist_f=0, hist_w=1, alpha=0.5, batch=B, chunk=5000, tag=""):
    cfg = LearnerConfig(alpha=alpha, lam=lam, hist_f=hist_f, hist_w=hist_w, max_depth=D)
    ex, h = run_experiment(env, q, cfg, batch=batch, steps=steps, chunk=chunk, seed=SEED, log=None)
    log(f"{env.name:8s} {tag:18s} lam={lam:<4} hf={hist_f} hw={hist_w} B={batch}: final RMSE {h['rmse'][-1]:.4f}  "
        f"1-step {h['err1'][-1]:.4f}  first step with RMSE<.05: {next((s for s, r in zip(h['step'], h['rmse']) if r < .05), None)}")
    return h


# (a) 5-state ring, depth-3 symmetric tree
curves = []
for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
    h = run(ring_world(5), symmetric_tree(2, 3), lam, 200_000, 3, chunk=2500, tag="sym d=3")
    results[f"ring5_lam{lam}"] = h; curves.append((f"lambda={lam}", h["step"], h["rmse"]))
plot_curves(curves, "5-state ring world, symmetric depth-3 question net (16 streams)", "paper_ring5_lambda.png", ref=0.05, ymax=0.5)

# representation check: single weight matrix with paper features fails; MLP with the same features works
curves = [("weight set per (a,o), linear", results["ring5_lam1.0"]["step"], results["ring5_lam1.0"]["rmse"])]
h = run(ring_world(5), symmetric_tree(2, 3), 1.0, 200_000, 3, hist_f=1, hist_w=0, chunk=2500, tag="paper features, linear")
results["ring5_paperfeat_linear"] = h; curves.append(("paper features, single matrix", h["step"], h["rmse"]))
cfg = LearnerConfig(alpha=0.1, lam=1.0, hist_f=1, hist_w=0, max_depth=3, model="mlp", hidden=32)
ex, h = run_experiment(ring_world(5), symmetric_tree(2, 3), cfg, batch=B, steps=200_000, chunk=2500, seed=SEED, log=None)
log(f"ring5 paper features, MLP-32 alpha=0.1: final RMSE {h['rmse'][-1]:.4f}")
results["ring5_paperfeat_mlp"] = h; curves.append(("paper features, MLP-32", h["step"], h["rmse"]))
plot_curves(curves, "5-state ring: the answer network must gate on the action", "paper_ring5_representation.png", ref=0.05, ymax=0.5)

# single stream, lambda=1, to compare with the paper's '< 10k steps' claim
h = run(ring_world(5), symmetric_tree(2, 3), 1.0, 60_000, 3, batch=1, chunk=1000, tag="sym d=3 B=1")
results["ring5_lam1_B1"] = h
h0 = run(ring_world(5), symmetric_tree(2, 3), 0.0, 300_000, 3, batch=1, chunk=5000, tag="sym d=3 B=1")
results["ring5_lam0_B1"] = h0
plot_curves([("lambda=1", h["step"], h["rmse"]), ("lambda=0", h0["step"], h0["rmse"])],
            "5-state ring world, single stream", "paper_ring5_single_stream.png", ref=0.05, ymax=0.5)

# (b) 8-state ring, sparse 16-node network (ICML'05 Fig. 6)
curves = []
for lam in [0.0, 0.5, 0.9, 1.0]:
    h = run(ring_world(8), ring_sparse(8), lam, 600_000, 8, chunk=10_000, tag="sparse d=8")
    results[f"ring8_lam{lam}"] = h; curves.append((f"lambda={lam}", h["step"], h["rmse"]))
plot_curves(curves, "8-state ring world, sparse 16-node question net (16 streams)", "paper_ring8_lambda.png", ref=0.05, ymax=0.5)

# (c) 6-state cycle world, 5-chain
curves = []
for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
    h = run(cycle_world(6), chain(5), lam, 200_000, 5, hist_f=1, hist_w=0, chunk=2500, tag="chain d=5")
    results[f"cycle6_lam{lam}"] = h; curves.append((f"lambda={lam}", h["step"], h["rmse"]))
plot_curves(curves, "6-state cycle world, 5-chain question net (16 streams)", "paper_cycle6_lambda.png", ref=0.05, ymax=0.5)

curves = []
for hf in [1, 2, 3, 4]:
    h = run(cycle_world(6), chain(5), 0.0, 200_000, 5, hist_f=hf, hist_w=0, chunk=2500, tag=f"chain d=5 TD(0)")
    results[f"cycle6_lam0_hf{hf}"] = h; curves.append((f"history {hf}", h["step"], h["rmse"]))
plot_curves(curves, "6-state cycle world, TD(0) with history features (IJCAI'05)", "paper_cycle6_history.png", ref=0.05, ymax=0.5)

save_json("exp_paper", {k: {kk: vv for kk, vv in v.items() if kk != "node_mse"} for k, v in results.items()})
log("done")
