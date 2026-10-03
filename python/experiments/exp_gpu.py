"""GPU experiments.  Needs a CUDA jax:  pip install -U "jax[cuda12]"
Run:  python3 exp_gpu.py bench        # steps/sec for network sizes x stream counts
      python3 exp_gpu.py fullrank     # the depth-8 tree (510 nodes, PSR rank 104) on bit-to-bit
      python3 exp_gpu.py grow         # grower with capacity 512, depth <= 8
      python3 exp_gpu.py all
Everything is written to results/data/exp_gpu_*.json and results/figs/gpu_*.png."""
import jax
jax.config.update("jax_default_matmul_precision", "highest")   # TF32 matmuls would add noise to sigmoid-output nets
from common import *
from tdnet import *
from tdnet.runner import Experiment
from tdnet.grow import Grower
import sys, time

log = logger()
log(f"jax {jax.__version__} devices: {jax.devices()}")
what = sys.argv[1] if len(sys.argv) > 1 else "all"
env = gridworld(MAPS["ijcai26"], name="bit2bit104")


def bench():
    rows = []
    for depth in [5, 6, 7, 8]:
        for B in [32, 128, 512]:
            q = symmetric_tree(2, depth)
            ex = Experiment(env, q, LearnerConfig(alpha=0.1, lam=1.0, hist_w=1, max_depth=depth), batch=B, seed=0)
            ex.run(200, chunk=200)                                   # compile
            t0 = time.time(); ex.run(ex.steps_done + 2000, chunk=2000); dt = time.time() - t0
            rows.append(dict(nodes=q.n_active, B=B, ms_per_step=1e3 * dt / 2000, env_steps_per_s=2000 * B / dt))
            log(f"bench nodes={q.n_active:4d} B={B:4d}: {1e3 * dt / 2000:7.2f} ms/step  {2000 * B / dt:10,.0f} env-steps/s")
    save_json("exp_gpu_bench", rows)


def fullrank(steps=3_000_000, B=128):
    """Depth-8 symmetric tree: the first fixed network whose questions separate all 104 states."""
    results, curves = {}, []
    for name, q, cfg in [
        ("depth 8 linear a=0.05", symmetric_tree(2, 8), LearnerConfig(alpha=0.05, lam=1.0, hist_w=1, max_depth=8)),
        ("depth 8 linear a=0.02", symmetric_tree(2, 8), LearnerConfig(alpha=0.02, lam=1.0, hist_w=1, max_depth=8)),
        ("depth 8 MLP-256 adam", symmetric_tree(2, 8), LearnerConfig(alpha=1e-3, lam=1.0, hist_f=2, hist_w=0, max_depth=8, model="mlp", hidden=256, optimizer="adam")),
        ("depth 6 MLP-256 adam", symmetric_tree(2, 6), LearnerConfig(alpha=1e-3, lam=1.0, hist_f=2, hist_w=0, max_depth=6, model="mlp", hidden=256, optimizer="adam")),
    ]:
        ex = Experiment(env, q, cfg, batch=B, seed=0)
        h = ex.run(steps, chunk=50_000, log=log)
        d = q.depths(); h["by_depth"] = {int(k): float(np.sqrt(h["node_mse"][-1][d == k].mean())) for k in range(1, d.max() + 1)}
        log(f"FULLRANK {name}: N={q.n_active} oracle RMSE {h['rmse'][-1]:.4f} next-obs {h['by_depth'][1]:.4f} 1-step |err| {h['err1'][-1]:.4f}")
        results[name] = {k: v for k, v in h.items() if k != "node_mse"}; curves.append((name, h["step"], h["rmse"]))
        save_json("exp_gpu_fullrank", results)
    plot_curves(curves, "Bit-to-bit gridworld: full-rank depth-8 question network (GPU)", "gpu_fullrank.png", ymax=0.5)


def grow(rounds=80, round_steps=50_000, B=128):
    q = level0_only(2, 1, capacity=512)
    ex = Experiment(env, q, LearnerConfig(alpha=0.05, lam=1.0, hist_w=1, max_depth=8), batch=B, seed=0)
    g = Grower(ex, 8, grow_k=8, log=log)
    h = dict(step=[], rmse=[], d1=[], n=[])
    for r in range(rounds):
        hh = ex.run(ex.steps_done + round_steps, chunk=round_steps)
        d = ex.q.depths()
        h["step"] += hh["step"]; h["rmse"] += hh["rmse"]; h["n"] += [ex.q.n_active]; h["d1"] += [float(np.sqrt(hh["node_mse"][-1][d == 1].mean()))]
        log(f"GROW round {r}: step {ex.steps_done:,} N={ex.q.n_active} depth {ex.q.max_depth()} RMSE {hh['rmse'][-1]:.4f} next-obs {h['d1'][-1]:.4f}")
        g.round()
    h["structure"] = ex.q.describe(); h["rounds"] = g.history
    save_json("exp_gpu_grow", h)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=130)
    axes[0].plot(h["step"], h["d1"], color=SERIES[0]); axes[0].set_xscale("log"); axes[0].set_ylim(0, .5); axes[0].set_ylabel("next-obs oracle RMSE"); axes[0].set_xlabel("steps per stream")
    axes[1].plot(h["step"], h["n"], color=SERIES[1]); axes[1].set_xscale("log"); axes[1].set_ylabel("active nodes"); axes[1].set_xlabel("steps per stream")
    fig.suptitle("Bit-to-bit gridworld: grown network, capacity 512, depth <= 8 (GPU)", x=0.01, ha="left", fontsize=11); fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "gpu_grow.png")); plt.close(fig)


if what in ("bench", "all"): bench()
if what in ("fullrank", "all"): fullrank()
if what in ("grow", "all"): grow()
log("done")
