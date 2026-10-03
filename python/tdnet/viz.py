"""Egocentric prediction maps for gridworld TD networks.

Every question is an action sequence ending in a sensor bit.  Resolving the sequence through the
true (deterministic, or most-likely) kinematics from the agent's true pose gives a pose, and the
sensor bit at that pose names one *wall segment*: the edge between that cell and the cell the
sensor looks at.  So every prediction is a claim "there is a wall on edge e", and the whole vector
y_t can be drawn as a map of edges whose darkness is the predicted probability.

When several questions resolve to the same edge (different paths; or paths that collapse because
a blocked move leaves the pose unchanged, which only ground truth knows) we aggregate: the edge
shows the mean prediction, its weight the number of questions, and a marker when they disagree.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from .envs import DIRS, env_step


def rot_rel(dr, dc, d):
    """Rotate a displacement so the agent's heading d points 'up' (-row)."""
    if d == 0: return (dr, dc)
    if d == 1: return (-dc, dr)
    if d == 2: return (-dr, -dc)
    return (dc, -dr)


def resolve(env, q):
    """For every active node i and state s: (final pose, sensed cell, edge key, path of poses).
    Uses the most likely transition (argmax of T), i.e. the deterministic kinematics."""
    S = env.n_states
    cells = env.cells
    pose_of = {s: (cells[s // 4][0], cells[s // 4][1], s % 4) for s in range(S)}
    T_mode = env.T.argmax(-1)                                   # (A, S)
    seqs = q.action_sequences()
    table = {}
    for i in np.flatnonzero(q.active):
        seq, bit = seqs[i]
        for s in range(S):
            path, cur = [pose_of[s]], s
            for a in seq:
                cur = int(T_mode[a, cur]); path.append(pose_of[cur])
            r, c, d = pose_of[cur]
            look = d if bit == 0 else ((d - 1) % 4 if bit == 1 else (d + 1) % 4)
            r2, c2 = r + DIRS[look][0], c + DIRS[look][1]
            edge = tuple(sorted([(r, c), (int(r2), int(c2))]))
            table[(int(i), s)] = dict(pose=(r, c, d), look=int(look), cell2=(int(r2), int(c2)), edge=edge, path=path)
    return table, pose_of


def rollout(ex, steps, stream=0, seed=1):
    """Run the batched env+learner for `steps` more steps, recording one stream:
    true state, action, observation, predictions y_t (N,), oracle values (N,)."""
    arrs = ex.arrs
    step_env = jax.jit(lambda s, b, k: env_step(arrs, s, b, k))
    key = jax.random.PRNGKey(seed)
    V = np.asarray(ex.V)
    rec = []
    for t in range(steps):
        key, k = jax.random.split(key)
        s_true = int(np.asarray(ex.s)[stream])
        y = np.asarray(ex.state["y"])[stream]
        a, s2, o, belief = step_env(ex.s, ex.belief, k)
        rec.append(dict(s=s_true, y=y.copy(), oracle=V[:, s_true].copy(), a=int(np.asarray(a)[stream]), o=int(np.asarray(o)[stream])))
        ex.state = ex.learner.step(ex.state, a, o)
        ex.s, ex.belief = s2, belief
        ex.steps_done += 1
    return rec


def frames(env, q, rec, table, pose_of):
    """Turn a rollout into drawable frames: per step the agent pose, the true observation, and
    one entry per edge with aggregated predictions in egocentric coordinates."""
    seqs = q.action_sequences()
    names = {int(i): "".join("FRLB"[a] if env.n_actions <= 4 else str(a) for a in seqs[i][0]) + ("" if seqs[i][1] == 0 else ["", "<", ">"][seqs[i][1]])
             for i in np.flatnonzero(q.active)}
    out = []
    for t, r_ in enumerate(rec):
        s = r_["s"]; ar, ac, ad = pose_of[s]
        edges = {}
        for i in np.flatnonzero(q.active):
            e = table[(int(i), s)]
            key = e["edge"]
            (r1, c1), (r2, c2) = key
            rel1, rel2 = rot_rel(r1 - ar, c1 - ac, ad), rot_rel(r2 - ar, c2 - ac, ad)
            d = edges.setdefault(str(key), dict(cells=[rel1, rel2], preds=[], truth=None, questions=[]))
            d["preds"].append(float(r_["y"][i])); d["truth"] = float(r_["oracle"][i]) if d["truth"] is None else d["truth"]
            d["questions"].append(dict(name=names[int(i)], y=round(float(r_["y"][i]), 3), truth=round(float(r_["oracle"][i]), 3),
                                       path=[rot_rel(pr - ar, pc - ac, ad) + (((pd - ad) % 4),) for pr, pc, pd in e["path"]]))
        for d in edges.values():
            p = np.array(d["preds"]); d["mean"] = round(float(p.mean()), 3); d["min"] = round(float(p.min()), 3); d["max"] = round(float(p.max()), 3)
            d["n"] = int(len(p)); del d["preds"]
        # true walls in egocentric frame (for the faint ground-truth layer)
        walls = []
        R, C = env.grid.shape
        for (r, c) in env.cells:
            for dd in range(4):
                r2, c2 = r + DIRS[dd][0], c + DIRS[dd][1]
                if not (0 <= r2 < R and 0 <= c2 < C and env.grid[r2, c2]):
                    walls.append([rot_rel(r - ar, c - ac, ad), rot_rel(r2 - ar, c2 - ac, ad)])
        out.append(dict(t=t, pose=[int(ar), int(ac), int(ad)], obs=r_["o"], action="FRLB"[r_["a"]], edges=list(edges.values()), walls=walls,
                        rmse=round(float(np.sqrt(np.mean((r_["y"][q.active] - r_["oracle"][q.active]) ** 2))), 3)))
    return out


# ---------------- static matplotlib rendering ----------------
def draw_frame(ax, frame, radius=4, show_truth=True):
    import matplotlib.pyplot as plt
    ax.set_aspect("equal"); ax.set_xlim(-radius - .5, radius + .5); ax.set_ylim(radius + .5, -radius - .5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    def seg(c1, c2):
        (r1, c1_), (r2, c2_) = c1, c2
        if r1 == r2:  # vertical edge between horizontally adjacent cells
            x = (c1_ + c2_) / 2; return [(x, r1 - .5), (x, r1 + .5)]
        y = (r1 + r2) / 2; return [(c1_ - .5, y), (c1_ + .5, y)]
    if show_truth:
        for w in frame["walls"]:
            (x1, y1), (x2, y2) = seg(tuple(w[0]), tuple(w[1])); ax.plot([x1, x2], [y1, y2], color="#c9cdd2", lw=4, solid_capstyle="butt", zorder=1)
    for e in frame["edges"]:
        (x1, y1), (x2, y2) = seg(tuple(e["cells"][0]), tuple(e["cells"][1]))
        p = e["mean"]
        ax.plot([x1, x2], [y1, y2], color=(0.11, 0.33, 0.70, max(p, 0.04)), lw=2 + 1.2 * np.log1p(e["n"]), solid_capstyle="butt", zorder=3)
        if e["max"] - e["min"] > 0.25:
            ax.plot((x1 + x2) / 2, (y1 + y2) / 2, marker="o", ms=5, color="#e07a1f", zorder=4)
    ax.plot(0, 0, marker=(3, 0, 0), ms=14, color="#1b2a3b", zorder=5)
    ax.set_title(f"t={frame['t']}  o={frame['obs']}  next action {frame['action']}  RMSE {frame['rmse']}", fontsize=9, loc="left")
