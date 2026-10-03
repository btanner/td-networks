"""Does stochasticity itself hurt?  Use maps that ARE solved deterministically (empty room: PSR
rank 7, two rooms) and add slip / sensor noise.  Oracle = exact belief-state prediction."""
from common import *
from tdnet import *

B, SEED, STEPS = 32, 0, 400_000
log = logger()
results = {}
variants = [("deterministic", {}), ("slip 0.1 (+0.02)", dict(slip=0.1, slip_twice=0.02)),
            ("noisy sensor .925/.9", dict(p_wall_seen=0.925, p_open_seen=0.9)),
            ("slip + noisy", dict(slip=0.1, slip_twice=0.02, p_wall_seen=0.925, p_open_seen=0.9))]
for mapname, depth in [("room4", 4), ("tworooms", 4)]:
    curves = []
    for label, kw in variants:
        env = gridworld(MAPS[mapname], name=f"{mapname} {label}", **kw)
        for model, alpha, opt in [("linear", 0.1, "sgd"), ("mlp", 1e-3, "adam")]:
            cfg = LearnerConfig(alpha=alpha, lam=1.0, hist_w=1 if model == "linear" else 0, hist_f=0 if model == "linear" else 2,
                                max_depth=depth, model=model, hidden=64, optimizer=opt)
            ex, h = run_experiment(env, symmetric_tree(2, depth), cfg, batch=B, steps=STEPS, chunk=20_000, seed=SEED, log=None)
            log(f"{env.name:28s} depth {depth} {model:6s}: oracle RMSE {h['rmse'][-1]:.4f}  1-step |err| {h['err1'][-1]:.4f}")
            results[f"{mapname}|{label}|{model}"] = {k: v for k, v in h.items() if k != "node_mse"}
            curves.append((f"{label}, {model}", h["step"], h["rmse"]))
    plot_curves(curves, f"{mapname}: stochastic variants of a solvable map (depth {depth})", f"stochastic_{mapname}.png", ymax=0.5)
save_json("exp_stochastic_easy", results)
log("done")
