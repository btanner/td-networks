"""History-only tabular predictors of the next observation bit, for comparison with the depth-1
nodes of the TD networks (IJCAI'05 Figures 4 and 6)."""
from common import *
from tdnet import *
from tdnet.baselines import run_history_baseline

log = logger()
results = {}
for env, ks, steps in [(ring_world(5), [1, 2, 3, 4, 6], 200_000), (ring_world(8), [1, 2, 4, 6, 8], 600_000),
                       (cycle_world(6), [1, 2, 3, 4, 5], 200_000), (gridworld(MAPS["ijcai26"], name="bit2bit104"), [1, 2, 3, 4, 5, 6, 8], 1_000_000)]:
    curves = []
    for k in ks:
        h = run_history_baseline(env, k, steps, batch=32, chunk=steps // 20)
        results[f"{env.name}_k{k}"] = h
        log(f"{env.name:12s} history k={k}: final 1-step oracle RMSE {h['rmse'][-1]:.4f}  |err| {h['err1'][-1]:.4f}  table rows {(env.n_actions*env.n_obs)**k}")
        curves.append((f"history k={k}", h["step"], h["rmse"]))
    plot_curves(curves, f"{env.name}: history-only next-observation predictors", f"history_{env.name}.png", ylabel="oracle RMSE (next-observation predictions)", ymax=0.5)
save_json("exp_history_baseline", results)
log("done")
