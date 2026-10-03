"""Growing / pruning question networks vs fixed symmetric trees.
Metric that is comparable across structures: oracle RMSE of the depth-1 (next-observation) nodes,
i.e. how good the learned state is for the primary prediction task.  Also report all-node RMSE and
the number of nodes over time."""
from common import *
from tdnet import *
from tdnet.grow import Grower
from tdnet.runner import Experiment
import sys

B, SEED = 32, 0
log = logger()
results = {}


def depth1_rmse(h, q):
    d = q.depths()
    return [float(np.sqrt(m[d == 1].mean())) for m in h["node_mse"]]


def run_fixed(env, q, steps, chunk, D, tag, alpha=0.5):
    cfg = LearnerConfig(alpha=alpha, lam=1.0, hist_w=1, max_depth=D)
    ex, h = run_experiment(env, q, cfg, batch=B, steps=steps, chunk=chunk, seed=SEED, log=None)
    h["d1"] = depth1_rmse(h, q); h["n"] = [q.n_active] * len(h["step"])
    log(f"{env.name} fixed {tag}: N={q.n_active} all-node RMSE {h['rmse'][-1]:.4f}  depth-1 RMSE {h['d1'][-1]:.4f}")
    return h


def run_grown(env, capacity, max_depth, rounds, round_steps, tag, alpha=0.5, **gkw):
    q = level0_only(env.n_actions, env.n_obs_bits, capacity=capacity)
    cfg = LearnerConfig(alpha=alpha, lam=1.0, hist_w=1, max_depth=max_depth)
    ex = Experiment(env, q, cfg, batch=B, seed=SEED)
    g = Grower(ex, max_depth, log=log, **gkw)
    h = dict(step=[], rmse=[], d1=[], n=[], err1=[])
    for r in range(rounds):
        hh = ex.run(ex.steps_done + round_steps, chunk=round_steps)
        h["step"] += hh["step"]; h["rmse"] += hh["rmse"]; h["err1"] += hh["err1"]
        h["d1"] += depth1_rmse(hh, ex.q); h["n"] += [ex.q.n_active] * len(hh["step"])
        g.round()
    log(f"{env.name} grown {tag}: final N={ex.q.n_active} depth {ex.q.max_depth()} all-node RMSE {h['rmse'][-1]:.4f} depth-1 RMSE {h['d1'][-1]:.4f}")
    log("final question network:\n" + ex.q.describe())
    h["structure"] = ex.q.describe(); h["rounds"] = g.history
    return h


def plot_grow(curves_d1, curves_n, title, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    for k, (label, xs, ys) in enumerate(curves_d1):
        axes[0].plot(xs, ys, color=SERIES[k % 8], label=label)
    axes[0].set_xscale("log"); axes[0].set_ylim(0, 0.5); axes[0].set_xlabel("time steps per stream")
    axes[0].set_ylabel("oracle RMSE of next-observation nodes"); axes[0].legend(fontsize=8); axes[0].set_title(title, loc="left", fontsize=11)
    for k, (label, xs, ys) in enumerate(curves_n):
        axes[1].plot(xs, ys, color=SERIES[k % 8], label=label)
    axes[1].set_xscale("log"); axes[1].set_xlabel("time steps per stream"); axes[1].set_ylabel("active nodes"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, fname)); plt.close(fig)


# ---- 8-state ring: can growth discover a sufficient sparse network? ----
env = ring_world(8)
c1, c2 = [], []
for depth in [4, 6]:
    h = run_fixed(env, symmetric_tree(2, depth), 800_000, 20_000, depth, f"sym depth {depth}")
    results[f"ring8_sym{depth}"] = h; c1.append((f"fixed symmetric depth {depth} ({h['n'][0]} nodes)", h["step"], h["d1"])); c2.append((c1[-1][0], h["step"], h["n"]))
h = run_fixed(env, ring_sparse(8), 800_000, 20_000, 8, "sparse 16"); results["ring8_sparse"] = h
c1.append(("fixed sparse chains (16 nodes)", h["step"], h["d1"])); c2.append((c1[-1][0], h["step"], h["n"]))
h = run_grown(env, capacity=64, max_depth=8, rounds=40, round_steps=20_000, tag="cap 64"); results["ring8_grown"] = h
c1.append(("grown + pruned (cap 64)", h["step"], h["d1"])); c2.append((c1[-1][0], h["step"], h["n"]))
h = run_grown(env, capacity=64, max_depth=8, rounds=40, round_steps=20_000, tag="cap 64 no prune", prune=False); results["ring8_grown_noprune"] = h
c1.append(("grown, no pruning (cap 64)", h["step"], h["d1"])); c2.append((c1[-1][0], h["step"], h["n"]))
plot_grow(c1, c2, "8-state ring world", "grow_ring8.png")
save_json("exp_grow", {k: {kk: vv for kk, vv in v.items() if kk not in ("node_mse",)} for k, v in results.items()})

# ---- symmetric empty room: PSR rank 7, easy ----
env = gridworld(MAPS["room4"], name="room4")
c1, c2 = [], []
for depth in [2, 3, 4]:
    h = run_fixed(env, symmetric_tree(2, depth), 600_000, 20_000, depth, f"sym depth {depth}", alpha=0.1)
    results[f"room4_sym{depth}"] = h; c1.append((f"fixed symmetric depth {depth} ({h['n'][0]} nodes)", h["step"], h["d1"])); c2.append((c1[-1][0], h["step"], h["n"]))
h = run_grown(env, capacity=48, max_depth=6, rounds=30, round_steps=20_000, tag="cap 48", alpha=0.1); results["room4_grown"] = h
c1.append(("grown + pruned (cap 48)", h["step"], h["d1"])); c2.append((c1[-1][0], h["step"], h["n"]))
plot_grow(c1, c2, "Empty 4x4 room (64 states, PSR rank 7)", "grow_room4.png")
save_json("exp_grow", {k: {kk: vv for kk, vv in v.items() if kk not in ("node_mse",)} for k, v in results.items()})

# ---- bit-to-bit gridworld: PSR rank 104, needs tests of length 8 ----
env = gridworld(MAPS["ijcai26"], name="bit2bit104")
c1, c2 = [], []
for depth, steps in [(4, 800_000), (5, 800_000), (6, 800_000), (7, 400_000)]:
    h = run_fixed(env, symmetric_tree(2, depth), steps, 40_000, depth, f"sym depth {depth}", alpha=0.1)
    results[f"grid_sym{depth}"] = h; c1.append((f"fixed symmetric depth {depth} ({h['n'][0]} nodes)", h["step"], h["d1"])); c2.append((c1[-1][0], h["step"], h["n"]))
h = run_grown(env, capacity=160, max_depth=8, rounds=50, round_steps=40_000, tag="cap 160 depth<=8", alpha=0.1); results["grid_grown"] = h
c1.append(("grown + pruned (cap 160, depth <= 8)", h["step"], h["d1"])); c2.append((c1[-1][0], h["step"], h["n"]))
h = run_grown(env, capacity=160, max_depth=8, rounds=50, round_steps=40_000, tag="cap 160 no prune", alpha=0.1, prune=False); results["grid_grown_noprune"] = h
c1.append(("grown, no pruning (cap 160)", h["step"], h["d1"])); c2.append((c1[-1][0], h["step"], h["n"]))
plot_grow(c1, c2, "Bit-to-bit gridworld (104 states)", "grow_grid.png")

save_json("exp_grow", {k: {kk: vv for kk, vv in v.items() if kk not in ("node_mse",)} for k, v in results.items()})
log("done")
