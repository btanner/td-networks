"""Question networks with a fixed node *capacity* so that growing / pruning never changes array
shapes (important for jit).  A node i asks: "if the next action is cond[i] (-1 = any action),
what will parent[i] predict at t+1?"  parent[i] == -1 means the target is observation bit
obs_bit[i] at t+1.  Inactive nodes have active[i] == False."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

ANY = -1          # action condition: any action
OBS = -1          # parent: the observation bit itself
NONE = -2         # ancestor table: past the observation / inactive


@dataclass
class QuestionNet:
    capacity: int
    n_actions: int
    n_obs_bits: int = 1
    cond: np.ndarray = field(default=None)
    parent: np.ndarray = field(default=None)
    obs_bit: np.ndarray = field(default=None)
    active: np.ndarray = field(default=None)

    def __post_init__(self):
        N = self.capacity
        if self.cond is None:
            self.cond = np.full(N, ANY, dtype=np.int32)
            self.parent = np.full(N, NONE, dtype=np.int32)
            self.obs_bit = np.zeros(N, dtype=np.int32)
            self.active = np.zeros(N, dtype=bool)

    # ---------------- construction ----------------
    def add(self, cond, parent, obs_bit=None) -> int:
        free = np.flatnonzero(~self.active)
        if len(free) == 0:
            raise RuntimeError("question network capacity exhausted")
        i = int(free[0])
        self.cond[i] = cond
        self.parent[i] = parent
        self.obs_bit[i] = self.obs_bit[parent] if (obs_bit is None and parent >= 0) else (obs_bit or 0)
        self.active[i] = True
        return i

    def remove(self, i: int):
        """Prune a leaf.  Children (nodes whose parent is i) must be removed first."""
        if np.any(self.active & (self.parent == i)):
            raise ValueError(f"node {i} still has children")
        self.active[i] = False
        self.parent[i] = NONE
        self.cond[i] = ANY

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    def depth_of(self, i: int) -> int:
        d, p = 1, self.parent[i]
        while p >= 0:
            d += 1; p = self.parent[p]
        return d

    def depths(self) -> np.ndarray:
        return np.array([self.depth_of(i) if self.active[i] else 0 for i in range(self.capacity)])

    def children(self, i: int) -> np.ndarray:
        return np.flatnonzero(self.active & (self.parent == i))

    def max_depth(self) -> int:
        return int(self.depths().max()) if self.n_active else 0

    # ---------------- tables used by the learner ----------------
    def ancestor_table(self, max_depth: int) -> np.ndarray:
        """P[d, i] = p^d(i) for d = 0..max_depth.  OBS (-1) = the observation, NONE (-2) = beyond
        / inactive."""
        N = self.capacity
        P = np.full((max_depth + 1, N), NONE, dtype=np.int32)
        P[0] = np.where(self.active, np.arange(N), NONE)
        for d in range(1, max_depth + 1):
            prev = P[d - 1]
            ok = prev >= 0
            P[d, ok] = self.parent[prev[ok]]
        return P

    def action_sequences(self):
        """For the oracle: list of (action-condition sequence from node to observation, obs_bit)."""
        seqs = []
        for i in range(self.capacity):
            if not self.active[i]:
                seqs.append(([], 0)); continue
            s, j = [], i
            while j >= 0:
                s.append(int(self.cond[j])); j = self.parent[j]
            seqs.append((s, int(self.obs_bit[i])))
        return seqs

    def describe(self) -> str:
        names = []
        for i in range(self.capacity):
            if not self.active[i]:
                continue
            s, b = self.action_sequences()[i]
            names.append(f"{i}: bit{b} after {s}")
        return "\n".join(names)


# ---------------- standard topologies ----------------
def level0_only(n_actions, n_obs_bits=1, capacity=None) -> QuestionNet:
    q = QuestionNet(capacity or n_actions * n_obs_bits, n_actions, n_obs_bits)
    for b in range(n_obs_bits):
        for a in range(n_actions):
            q.add(a, OBS, b)
    return q


def symmetric_tree(n_actions, depth, n_obs_bits=1, capacity=None) -> QuestionNet:
    """Fully action-conditional tree of the given depth (ICML'05 Fig. 1)."""
    n = n_obs_bits * sum(n_actions ** d for d in range(1, depth + 1))
    q = QuestionNet(capacity or n, n_actions, n_obs_bits)

    def hang(p, level):
        for a in range(n_actions):
            me = q.add(a, p)
            if level + 1 < depth:
                hang(me, level + 1)

    for b in range(n_obs_bits):
        for a in range(n_actions):
            root = q.add(a, OBS, b)
            if depth > 1:
                hang(root, 1)
    return q


def chain(depth, n_actions=1, n_obs_bits=1, capacity=None) -> QuestionNet:
    """Unconditional chain: y^0 predicts o_{t+1}, y^k predicts y^{k-1}_{t+1} (cycle world net)."""
    q = QuestionNet(capacity or depth * n_obs_bits, n_actions, n_obs_bits)
    for b in range(n_obs_bits):
        p = OBS
        for _ in range(depth):
            p = q.add(ANY, p, b)
    return q


def ring_sparse(depth, n_obs_bits=1, capacity=None) -> QuestionNet:
    """ICML'05 Fig. 6: per level one L-question and one R-question, each a chain of the same
    action (y^k_L = 'if L then y^{k-1}_L').  2*depth nodes, two actions."""
    q = QuestionNet(capacity or 2 * depth * n_obs_bits, 2, n_obs_bits)
    for b in range(n_obs_bits):
        for a in range(2):
            p = OBS
            for _ in range(depth):
                p = q.add(a, p, b)
    return q
