"""Growing and pruning the question network online, using only oracle-free statistics kept by the
learner:

  node_err  EMA of the squared lambda-return error of each node (is the question answerable yet?)
  node_cnt  EMA of how often the node had a valid target
  ym_h/ysq_h EMA of the prediction's mean / 2nd moment given the weight-set context (a_{t-1}, o_t):
            a node whose prediction is constant given the context carries no information beyond
            the input features (e.g. "turn right then turn left" = the current observation)
  yy        EMA of y y^T: two nodes with |corr| ~ 1 ask equivalent questions (e.g. LLL == RR on a
            5-ring), keep the older one

Grow rule : a leaf gets its |A| children (breadth-first, at most grow_k parents per round, never
            deeper than max_depth) when it is either *answered* (error below tau_grow: its children
            can bootstrap from it) or *stuck* (error improved by less than tau_plateau relative to
            the previous round: the current predictive state is insufficient, deeper questions may
            supply the missing state).
Prune rule: a leaf older than min_age that is (a) constant in context, (b) redundant with an older
            node, or (c) still unlearnable after min_age_bad rounds, is removed and its (parent,
            action) is tomb-stoned so it is not re-grown.  Level-1 nodes are never pruned.
"""
from __future__ import annotations
import numpy as np
from .question import QuestionNet, OBS


class Grower:
    def __init__(self, ex, max_depth, grow_k=4, tau_grow=0.005, tau_plateau=0.05, tau_const=1e-3, tau_corr=0.98,
                 tau_bad=0.05, min_age=2, min_age_bad=6, prune=True, log=None):
        self.ex, self.max_depth, self.grow_k = ex, max_depth, grow_k
        self.tau_grow, self.tau_plateau, self.tau_const, self.tau_corr, self.tau_bad = tau_grow, tau_plateau, tau_const, tau_corr, tau_bad
        self.prev_err = np.full(ex.q.capacity, np.inf)
        self.min_age, self.min_age_bad, self.prune, self.log = min_age, min_age_bad, prune, log
        self.age = np.zeros(ex.q.capacity, dtype=int)
        self.tombstones = set()
        self.history = []

    def stats(self):
        st = self.ex.state
        err = np.asarray(st["node_err"]); cnt = np.asarray(st["node_cnt"])
        var_h = np.asarray(st["ysq_h"] - st["ym_h"] ** 2).mean(0)
        m = np.asarray(st["node_mean"]); C = np.asarray(st["yy"]) - np.outer(m, m)
        sd = np.sqrt(np.clip(np.diag(C), 1e-12, None)); corr = C / np.outer(sd, sd)
        np.fill_diagonal(corr, 0.0)
        return err, cnt, var_h, corr

    def round(self):
        q, ex = self.ex.q, self.ex
        err, cnt, var_h, corr = self.stats()
        active = np.flatnonzero(q.active)
        depths = q.depths()
        leaves = [i for i in active if len(q.children(i)) == 0]
        pruned, reasons = [], {}
        if self.prune:
            for i in sorted(leaves, key=lambda i: -depths[i]):
                if self.age[i] < self.min_age or depths[i] <= 1:
                    continue
                reason = None
                if var_h[i] < self.tau_const:
                    reason = "constant-in-context"
                else:
                    others = [j for j in active if j != i and j not in pruned and (self.age[j] > self.age[i] or (self.age[j] == self.age[i] and j < i))]
                    if others and np.abs(corr[i, others]).max() > self.tau_corr:
                        reason = f"redundant(with {others[int(np.abs(corr[i, others]).argmax())]})"
                    elif self.age[i] >= self.min_age_bad and err[i] > self.tau_bad:
                        reason = "unlearnable"
                if reason:
                    pruned.append(i); reasons[i] = reason
            for i in pruned:
                self.tombstones.add((int(q.parent[i]), int(q.cond[i])))
                q.remove(i); self.age[i] = 0
        # grow
        grown, new_nodes = [], []
        improvement = (self.prev_err - err) / np.maximum(self.prev_err, 1e-9)
        answered = err < self.tau_grow
        stuck = (improvement < self.tau_plateau) & (self.age >= 2)
        cands = [i for i in leaves if i not in pruned and depths[i] < self.max_depth and (answered[i] or stuck[i]) and cnt[i] > 0.05
                 and self.age[i] >= 1 and any((i, a) not in self.tombstones for a in range(q.n_actions))]
        cands.sort(key=lambda i: (depths[i], err[i]))
        self.prev_err = err.copy()
        for i in cands[:self.grow_k]:
            free = int((~q.active).sum())
            kids = [a for a in range(q.n_actions) if (i, a) not in self.tombstones]
            if free < len(kids):
                break
            for a in kids:
                new_nodes.append(q.add(a, i))
            grown.append(i)
        ex.set_question(q, reset_nodes=pruned + new_nodes)
        self.age[q.active] += 1
        info = dict(step=ex.steps_done, n_active=q.n_active, grown=grown, new=new_nodes, pruned=pruned, reasons=reasons,
                    max_depth=q.max_depth(), why={int(i): ("answered" if answered[i] else "stuck") for i in grown})
        self.history.append(info)
        if self.log:
            self.log(f"  round@{ex.steps_done:,}: active {q.n_active} (depth {q.max_depth()}), grew under {[(i, info['why'][i][:4]) for i in grown]} -> {len(new_nodes)} new, "
                     f"pruned {[(i, reasons[i]) for i in pruned]}")
        return info
