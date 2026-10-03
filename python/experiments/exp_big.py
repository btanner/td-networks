"""Beyond the papers: larger maps, multi-bit observations, more actions, stochastic dynamics and
noisy sensors.  Oracle = exact belief-state predictions, so RMSE 0 is attainable in principle for
every variant."""
from common import *
from tdnet import *
import sys

B, SEED = 32, 0
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
log = logger()
results = {}


def run(env, q, lam=1.0, D=None, hist_f=0, hist_w=1, alpha=0.5, steps=STEPS, tag="", model="linear", hidden=64, opt="sgd", chunk=20_000):
    D = D or q.max_depth()
    cfg = LearnerConfig(alpha=alpha, lam=lam, hist_f=hist_f, hist_w=hist_w, max_depth=D, model=model, hidden=hidden, optimizer=opt)
    ex, h = run_experiment(env, q, cfg, batch=B, steps=steps, chunk=chunk, seed=SEED, log=None)
    depths = q.depths()
    by_depth = {int(d): float(np.sqrt(h["node_mse"][-1][depths == d].mean())) for d in range(1, depths.max() + 1)}
    log(f"{env.name:26s} {tag:24s} N={q.n_active:3d} S={env.n_states} A={env.n_actions} O={env.n_obs} lam={lam} hf={hist_f} hw={hist_w} {model}: "
        f"RMSE {h['rmse'][-1]:.4f} 1-step {h['err1'][-1]:.4f} by depth {' '.join(f'{d}:{v:.3f}' for d, v in by_depth.items())}")
    h["by_depth"] = by_depth; h["env"] = env.name; h["N"] = q.n_active; h["S"] = env.n_states
    return h


# ---- 1. larger maps, 1 bit, F/R ----
curves = []
for mapname in ["ijcai26", "tworooms", "maze12"]:
    env = gridworld(MAPS[mapname], name=mapname)
    for depth, hf in [(3, 0), (4, 0), (4, 3)]:
        h = run(env, symmetric_tree(2, depth), hist_f=hf, tag=f"1bit F/R depth {depth}")
        results[f"{mapname}_1bit_d{depth}_hf{hf}"] = h
        if (depth, hf) == (4, 3):
            curves.append((f"{mapname} ({env.n_states} states)", h["step"], h["rmse"]))
plot_curves(curves, "Larger 1-bit maps: depth-4 tree, 3-step history, lambda=1", "big_maps_1bit.png", ymax=0.5)

# ---- 2. richer observations and more actions on the 26-cell map ----
curves = []
env = gridworld(MAPS["ijcai26"], sensor="flr", name="ijcai26_3bit")
for depth in [2, 3]:
    h = run(env, symmetric_tree(2, depth, n_obs_bits=3), tag=f"3bit F/R depth {depth}"); results[f"ijcai26_3bit_d{depth}"] = h
    curves.append((f"3 bits, F/R, depth {depth}", h["step"], h["rmse"]))
env = gridworld(MAPS["ijcai26"], actions=("F", "R", "L", "B"), sensor="flr", name="ijcai26_3bit_4act")
h = run(env, symmetric_tree(4, 2, n_obs_bits=3), tag="3bit FRLB depth 2"); results["ijcai26_3bit_4act_d2"] = h
curves.append(("3 bits, F/R/L/B, depth 2", h["step"], h["rmse"]))
env = gridworld(MAPS["ijcai26"], actions=("F", "R", "L", "B"), sensor="front", name="ijcai26_1bit_4act")
h = run(env, symmetric_tree(4, 3), tag="1bit FRLB depth 3"); results["ijcai26_1bit_4act_d3"] = h
curves.append(("1 bit, F/R/L/B, depth 3", h["step"], h["rmse"]))
plot_curves(curves, "26-cell map: multi-bit sensors and four actions (lambda=1)", "big_obs_actions.png", ymax=0.5)

# ---- 3. stochastic dynamics / noisy sensors ----
curves = []
variants = [
    ("slip 0.1 (+0.02 double)", dict(slip=0.1, slip_twice=0.02)),
    ("noisy sensor .925/.9", dict(p_wall_seen=0.925, p_open_seen=0.9)),
    ("slip + noisy sensor", dict(slip=0.1, slip_twice=0.02, p_wall_seen=0.925, p_open_seen=0.9)),
]
for label, kw in variants:
    env = gridworld(MAPS["ijcai26"], **kw)
    for depth, hf, lam in [(3, 0, 1.0), (3, 2, 1.0), (3, 2, 0.5), (4, 2, 1.0)]:
        h = run(env, symmetric_tree(2, depth), lam=lam, hist_f=hf, tag=f"{label} d{depth}")
        results[f"{env.name}_d{depth}_hf{hf}_lam{lam}"] = h
        if (depth, hf, lam) == (3, 2, 1.0):
            curves.append((label, h["step"], h["rmse"]))
    h = run(env, symmetric_tree(2, 3), hist_f=2, hist_w=0, model="mlp", hidden=64, alpha=1e-3, opt="adam", tag=f"{label} d3 mlp")
    results[f"{env.name}_d3_hf2_mlp"] = h
h = results["ijcai26_1bit_d3_hf0"]; curves.append(("deterministic (reference)", h["step"], h["rmse"]))
plot_curves(curves, "26-cell map with stochastic motion / noisy sensor (depth 3, history 2, lambda=1)", "big_stochastic.png", ymax=0.5)

save_json("exp_big", {k: {kk: vv for kk, vv in v.items() if kk != "node_mse"} for k, v in results.items()})
log("done")
