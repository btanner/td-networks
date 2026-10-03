"""Shared helpers for experiment scripts: result persistence and plotting (palette from the dataviz
reference: fixed categorical order, thin lines, recessive chrome, one axis)."""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "results", "data")
FIGS = os.path.join(ROOT, "results", "figs")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "legend.frameon": False, "lines.linewidth": 1.6,
})


def save_json(name, obj):
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.floating, np.integer)): return o.item()
        raise TypeError(type(o))
    with open(os.path.join(DATA, name + ".json"), "w") as f:
        json.dump(obj, f, default=conv)


def load_json(name):
    with open(os.path.join(DATA, name + ".json")) as f:
        return json.load(f)


def plot_curves(curves, title, fname, ylabel="oracle RMSE (all nodes)", logx=True, ymax=None, ref=None):
    """curves: list of (label, steps, values).  Saves PNG to results/figs."""
    fig, ax = plt.subplots(figsize=(7, 4), dpi=130)
    for k, (label, xs, ys) in enumerate(curves):
        ax.plot(xs, ys, color=SERIES[k % len(SERIES)], label=label)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=INK2, va="center")
    if ref is not None:
        ax.axhline(ref, color=MUTED, lw=0.8, ls="--")
        ax.annotate(f"RMSE = {ref}", (ax.get_xlim()[0] if not logx else min(c[1][0] for c in curves), ref),
                    xytext=(2, 3), textcoords="offset points", fontsize=8, color=MUTED)
    if logx: ax.set_xscale("log")
    ax.set_ylim(0, ymax)
    ax.set_xlabel("time steps per stream"); ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, fname)); plt.close(fig)


def logger(prefix=""):
    t0 = time.time()
    def log(msg):
        print(f"{prefix}{msg}", flush=True)
    return log
