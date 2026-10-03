"""Train networks on the bit-to-bit map, roll one stream out, and build the egocentric
prediction-map viewer (results/viz/bit2bit_viewer.html) plus a PNG strip of frames."""
from common import *
from tdnet import *
from tdnet.runner import Experiment
from tdnet.viz import resolve, rollout, frames, draw_frame
import json, os, sys

def conv(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
VIZ = os.path.join(ROOT, "results", "viz"); os.makedirs(VIZ, exist_ok=True)
log = logger()
datasets = {}

configs = [
    ("1-bit sensor, depth-4 tree, MLP-64 answer net", dict(sensor="front"), symmetric_tree(2, 4),
     LearnerConfig(alpha=1e-3, lam=1.0, hist_f=2, hist_w=0, max_depth=4, model="mlp", hidden=64, optimizer="adam")),
    ("1-bit sensor, depth-4 tree, linear answer net", dict(sensor="front"), symmetric_tree(2, 4),
     LearnerConfig(alpha=0.1, lam=1.0, hist_w=1, max_depth=4)),
    ("3-bit sensor (front/left/right), depth-3 tree, linear", dict(sensor="flr"), symmetric_tree(2, 3, n_obs_bits=3),
     LearnerConfig(alpha=0.1, lam=1.0, hist_w=1, max_depth=3)),
]
for name, envkw, q, cfg in configs:
    env = gridworld(MAPS["ijcai26"], name="bit2bit104", **envkw)
    ex = Experiment(env, q, cfg, batch=32, seed=0)
    h = ex.run(STEPS, chunk=50_000)
    log(f"{name}: trained {STEPS:,} steps, oracle RMSE {h['rmse'][-1]:.3f}, 1-step |err| {h['err1'][-1]:.3f}")
    table, pose_of = resolve(env, q)
    rec = rollout(ex, 160)
    fr = frames(env, q, rec, table, pose_of)
    datasets[name] = dict(frames=fr, rmse=h["rmse"][-1])
    # PNG strip of 8 frames
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8), dpi=120)
    for ax, k in zip(axes.flat, range(0, 160, 20)):
        draw_frame(ax, fr[k])
    fig.suptitle(name, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "viz_" + name.replace(", ", "_").replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "") + ".png")); plt.close(fig)

tpl = open(os.path.join(ROOT, "python", "tdnet", "viz_template.html")).read()
blob = json.dumps(datasets, separators=(",", ":"), default=conv)
open(os.path.join(VIZ, "frames.json"), "w").write(blob)
page = tpl.replace("/*DATA*/{}", blob)
open(os.path.join(VIZ, "bit2bit_viewer_body.html"), "w").write(page)
open(os.path.join(VIZ, "bit2bit_viewer.html"), "w").write('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">\n' + page + "\n</head></html>" if False else '<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"></head><body>\n' + page + "\n</body></html>")
log(f"wrote viewer with {sum(len(d['frames']) for d in datasets.values())} frames, {os.path.getsize(os.path.join(VIZ, 'bit2bit_viewer.html'))/1e6:.1f} MB")
