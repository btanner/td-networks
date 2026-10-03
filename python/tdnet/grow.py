"""Growing and pruning the question network online, using only oracle-free statistics kept by the
learner:

  node_err  EMA of the squared lambda-return error of each node (is the question answerable yet?)
  node_cnt  EMA of how often the node had a valid target
  ym_h/ysq_h EMA of the mean / 2nd moment of the *input* y_{t-1}^i given the context (a_{t-1}, o_t)
            it is used in: an input that is constant given the context carries no information
            beyond the context features and can be folded exactly into that weight set's bias
  yy        EMA of y y^T: two nodes whose mean squared difference E[(y_i - y_j)^2] ~ 0 ask
            equivalent questions (e.g. LLL == RR on a 5-ring); keep the older one

Grow rule : a leaf gets its |A| children (breadth-first, at most grow_k parents per round, never
            deeper than max_depth) when it is either *answered* (error below tau_grow: its children
            can bootstrap from it) or *stuck* (error improved by less than tau_plateau relative to
            the previous round: the current predictive state is insufficient, deeper questions may
            supply the missing state).
Prune rule: a leaf older than min_age is removed when it is (a) *answered* but constant in
            context, (b) *answered* and redundant with an older answered node, or (c) still
            unlearnable after min_age_bad rounds although its parent is answered.  Unlearned
            nodes are never judged redundant: before convergence all predictions are correlated.
            Pruning preserves the answer network exactly: a redundant node's input weights are
            added onto its twin's, a constant node is folded into the per-context bias.  A
            (parent, action) pruned as constant / redundant is tomb-stoned permanently, one pruned as
            unlearnable for tombstone_rounds rounds.  Level-1 nodes are never pruned.
"""
from __future__ import annotations
import numpy as np
from .question import QuestionNet, OBS


class Grower:
    def __init__(self, ex, max_depth, grow_k=4, tau_grow=0.005, tau_plateau=0.05, tau_const=1e-4, tau_corr=0.98,
                 tau_bad=0.05, min_age=2, min_age_bad=6, tombstone_rounds=10, prune=True, log=None, tau_msd=1e-4):
        self.tau_msd = tau_msd
        self.tombstone_rounds, self.round_no = tombstone_rounds, 0
        self.ex, self.max_depth, self.grow_k = ex, max_depth, grow_k
        self.tau_grow, self.tau_plateau, self.tau_const, self.tau_corr, self.tau_bad = tau_grow, tau_plateau, tau_const, tau_corr, tau_bad
        self.prev_err = np.full(ex.q.capacity, np.inf)
        self.min_age, self.min_age_bad, self.prune, self.log = min_age, min_age_bad, prune, log
        self.age = np.zeros(ex.q.capacity, dtype=int)
        self.tombstones = {}   # (parent, action) -> round pruned
        self.history = []

    def stats(self):
        st = self.ex.state
        err = np.asarray(st["node_err"]); cnt = np.asarray(st["node_cnt"])
        var_h = np.asarray(st["ysq_h"] - st["ym_h"] ** 2).mean(0)
        yy = np.asarray(st["yy"]); d = np.diag(yy)
        msd = d[:, None] + d[None, :] - 2 * yy          # E[(y_i - y_j)^2]
        np.fill_diagonal(msd, np.inf)
        return err, cnt, var_h, msd

    def round(self):
        q, ex = self.ex.q, self.ex
        err, cnt, var_h, msd = self.stats()
        active = [int(i) for i in np.flatnonzero(q.active)]
        depths = q.depths()
        leaves = [int(i) for i in active if len(q.children(i)) == 0]
        pruned, reasons = [], {}
        answered = err < self.tau_grow
        self.round_no += 1
        self.tombstones = {k: r for k, r in self.tombstones.items() if r is None or self.round_no - r < self.tombstone_rounds}
        merges, folds = [], []
        ym_h = np.asarray(self.ex.state["ym_h"])
        if self.prune:
            for i in sorted(leaves, key=lambda i: -depths[i]):
                if self.age[i] < self.min_age or depths[i] <= 1:
                    continue
                reason = None
                if answered[i] and var_h[i] < self.tau_const:
                    reason = "constant-in-context"; folds.append((i, ym_h[:, i].copy()))
                elif answered[i]:
                    others = [j for j in active if j != i and j not in pruned and answered[j]
                              and (self.age[j] > self.age[i] or (self.age[j] == self.age[i] and j < i))]
                    if others and msd[i, others].min() < self.tau_msd:
                        j = others[int(msd[i, others].argmin())]
                        reason = f"redundant(with {j})"; merges.append((i, j))
                elif self.age[i] >= self.min_age_bad and err[i] > self.tau_bad and answered[q.parent[i]]:
                    reason = "unlearnable"
                if reason:
                    pruned.append(int(i)); reasons[int(i)] = reason
            for i in pruned:
                self.tombstones[(int(q.parent[i]), int(q.cond[i]))] = self.round_no if reasons[i] == "unlearnable" else None
                q.remove(i); self.age[i] = 0
        # grow
        grown, new_nodes = [], []
        improvement = (self.prev_err - err) / np.maximum(self.prev_err, 1e-9)
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
                new_nodes.append(int(q.add(a, i)))
            grown.append(int(i))
        ex.set_question(q, reset_nodes=pruned + new_nodes, merges=merges, folds=folds)
        self.age[q.active] += 1
        info = dict(step=ex.steps_done, n_active=q.n_active, grown=grown, new=new_nodes, pruned=pruned, reasons=reasons,
                    max_depth=q.max_depth(), why={int(i): ("answered" if answered[i] else "stuck") for i in grown})
        self.history.append(info)
        if self.log:
            self.log(f"  round@{ex.steps_done:,}: active {q.n_active} (depth {q.max_depth()}), grew under {[(i, info['why'][i][:4]) for i in grown]} -> {len(new_nodes)} new, "
                     f"pruned {[(i, reasons[i]) for i in pruned]}")
        return info
