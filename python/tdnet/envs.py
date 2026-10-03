"""Tabular POMDPs with a batch axis, written so that the environment step runs inside a jitted
lax.scan.  Every environment is (T[a, s, s'], Obs[s, o], obs_bits[o, :]); the oracle computes the
exact value of every question from the true belief state, exactly like the papers' evaluation."""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class TabularPOMDP:
    T: np.ndarray          # (A, S, S) transition probabilities
    Obs: np.ndarray        # (S, O) observation probabilities (observation emitted on arrival)
    obs_bits: np.ndarray   # (O, nbits) bit decomposition of each observation symbol
    name: str = "pomdp"
    policy: np.ndarray = None   # (A,) action probabilities used to generate experience

    def __post_init__(self):
        A, S, _ = self.T.shape
        if self.policy is None:
            self.policy = np.full(A, 1.0 / A)
        assert np.allclose(self.T.sum(-1), 1), "T rows must sum to 1"
        assert np.allclose(self.Obs.sum(-1), 1), "Obs rows must sum to 1"

    @property
    def n_actions(self): return self.T.shape[0]
    @property
    def n_states(self): return self.T.shape[1]
    @property
    def n_obs(self): return self.Obs.shape[1]
    @property
    def n_obs_bits(self): return self.obs_bits.shape[1]

    # ---- oracle: v[i, s] = P(obs bit = 1 | state s, node i's action sequence) ----
    def node_values(self, action_sequences) -> np.ndarray:
        Tbar = np.einsum('a,ass->ss', self.policy, self.T) if False else (self.policy[:, None, None] * self.T).sum(0)
        bit_prob = self.Obs @ self.obs_bits            # (S, nbits): P(bit=1 | arrive in s)
        V = np.zeros((len(action_sequences), self.n_states))
        for i, (seq, b) in enumerate(action_sequences):
            if not seq:
                continue
            v = bit_prob[:, b]
            for a in reversed(seq):                   # seq[0] is the first action to take
                M = Tbar if a < 0 else self.T[a]
                v = M @ v
            # the innermost step is applied last: v = T[seq[0]] ... T[seq[-1]] bit
            V[i] = v
        return V

    def arrays(self):
        return dict(T=jnp.asarray(self.T, jnp.float32), logT=jnp.log(jnp.asarray(self.T) + 1e-30).astype(jnp.float32),
                    Obs=jnp.asarray(self.Obs, jnp.float32), logObs=jnp.log(jnp.asarray(self.Obs) + 1e-30).astype(jnp.float32),
                    obs_bits=jnp.asarray(self.obs_bits, jnp.float32), logpi=jnp.log(jnp.asarray(self.policy)).astype(jnp.float32))


def env_step(arrs, s, belief, key):
    """One batched step: sample a ~ pi, s' ~ T[a, s], o ~ Obs[s'], and update the exact belief.
    s: (B,) int, belief: (B, S)."""
    k1, k2, k3 = jax.random.split(key, 3)
    B = s.shape[0]
    a = jax.random.categorical(k1, jnp.broadcast_to(arrs['logpi'], (B, arrs['logpi'].shape[0])))
    s2 = jax.random.categorical(k2, arrs['logT'][a, s])
    o = jax.random.categorical(k3, arrs['logObs'][s2])
    pred = jnp.einsum('bs,bst->bt', belief, arrs['T'][a])
    post = pred * arrs['Obs'][:, o].T
    belief = post / jnp.maximum(post.sum(-1, keepdims=True), 1e-30)
    return a, s2, o, belief


# ======================= environments =======================
def ring_world(n: int) -> TabularPOMDP:
    """n-state ring, 2 actions (0 = counter-clockwise, 1 = clockwise), bit = 1 only in state 0."""
    T = np.zeros((2, n, n))
    for s in range(n):
        T[0, s, (s - 1) % n] = 1
        T[1, s, (s + 1) % n] = 1
    Obs = np.zeros((n, 2)); Obs[:, 0] = 1; Obs[0] = [0, 1]
    return TabularPOMDP(T, Obs, np.array([[0], [1]]), name=f"ring{n}")


def cycle_world(n: int) -> TabularPOMDP:
    """n-state deterministic cycle, 1 action, bit = 1 only in state 0."""
    T = np.zeros((1, n, n))
    for s in range(n):
        T[0, s, (s + 1) % n] = 1
    Obs = np.zeros((n, 2)); Obs[:, 0] = 1; Obs[0] = [0, 1]
    return TabularPOMDP(T, Obs, np.array([[0], [1]]), name=f"cycle{n}")


# ---- egocentric gridworlds (IJCAI'05 Sec. 5 and the C++ TwoDimensionalPOMDP) ----
MAPS = {
    # IJCAI-style: 26 open cells -> 104 states. '#' wall, '.' open
    "ijcai26": [
        "#########",
        "#...#...#",
        "#.#.#.#.#",
        "#.#...#.#",
        "#.##.##.#",
        "#.......#",
        "####.####",
        "#########",
    ],
    "room4": [
        "######",
        "#....#",
        "#....#",
        "#....#",
        "#....#",
        "######",
    ],
    "corridor": [
        "########",
        "#......#",
        "########",
    ],
    "tworooms": [
        "#############",
        "#.....#.....#",
        "#.....#.....#",
        "#...........#",
        "#.....#.....#",
        "#.....#.....#",
        "#############",
    ],
    "maze12": [
        "##############",
        "#......#.....#",
        "#.####.#.###.#",
        "#.#..#...#...#",
        "#.#..#####.#.#",
        "#.#........#.#",
        "#.######.###.#",
        "#......#.....#",
        "#.####.#.###.#",
        "#....#...#...#",
        "####.#####.#.#",
        "#..........#.#",
        "##############",
    ],
}
# C++ constants
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])   # up, right, down, left (clockwise order)


def gridworld(map_rows, actions=("F", "R"), sensor="front", slip=0.0, slip_twice=0.0,
              p_wall_seen=1.0, p_open_seen=1.0, name=None) -> TabularPOMDP:
    """Egocentric gridworld.  State = (row, col, heading).  Actions subset of
    F (forward), R (turn right 90), L (turn left 90), B (backward).
    sensor: 'front' -> 1 bit (wall directly ahead); 'flr' -> 3 bits (front, left, right),
    8 observation symbols, as the C++ SENSORFUSION mode.
    slip: probability a move/turn fails (stay); slip_twice: probability it happens twice
    (C++ STICKYTILE).  p_wall_seen / p_open_seen: sensor reliability (C++ O_WW / O_OO)."""
    grid = np.array([[c != '#' for c in row] for row in map_rows])
    R, C = grid.shape
    cells = [(r, c) for r in range(R) for c in range(C) if grid[r, c]]
    idx = {(r, c, d): k for k, (r, c) in enumerate(cells) for d in range(4)}
    idx = {}
    for k, (r, c) in enumerate(cells):
        for d in range(4):
            idx[(r, c, d)] = 4 * k + d
    S = 4 * len(cells)

    def blocked(r, c):
        return not (0 <= r < R and 0 <= c < C and grid[r, c])

    def fwd(r, c, d):
        r2, c2 = r + DIRS[d][0], c + DIRS[d][1]
        return (r, c, d) if blocked(r2, c2) else (r2, c2, d)

    def apply(st, act):
        r, c, d = st
        if act == "F": return fwd(r, c, d)
        if act == "B":
            r2, c2 = r - DIRS[d][0], c - DIRS[d][1]
            return (r, c, d) if blocked(r2, c2) else (r2, c2, d)
        if act == "R": return (r, c, (d + 1) % 4)
        if act == "L": return (r, c, (d - 1) % 4)
        raise ValueError(act)

    A = len(actions)
    T = np.zeros((A, S, S))
    p_next = 1 - slip - slip_twice
    for st, s in idx.items():
        for ai, act in enumerate(actions):
            n1 = apply(st, act); n2 = apply(n1, act)
            T[ai, s, idx[n1]] += p_next
            T[ai, s, idx[n2]] += slip_twice
            T[ai, s, s] += slip

    # sensors: wall ahead / left / right of current pose
    def wall_ahead(r, c, d):
        return blocked(r + DIRS[d][0], c + DIRS[d][1])
    bits_true = np.zeros((S, 3), dtype=int)
    for (r, c, d), s in idx.items():
        bits_true[s] = [wall_ahead(r, c, d), wall_ahead(r, c, (d - 1) % 4), wall_ahead(r, c, (d + 1) % 4)]
    nb = 1 if sensor == "front" else 3
    O = 2 ** nb
    obs_bits = np.array([[(o >> b) & 1 for b in range(nb)] for o in range(O)])
    Obs = np.ones((S, O))
    for s in range(S):
        for o in range(O):
            for b in range(nb):
                truth, seen = bits_true[s, b], obs_bits[o, b]
                p = (p_wall_seen if seen else 1 - p_wall_seen) if truth else (p_open_seen if not seen else 1 - p_open_seen)
                Obs[s, o] *= p
    nm = name or f"grid{len(cells)}c{A}a{nb}b" + (f"_slip{slip}" if slip else "") + ("_noisy" if p_wall_seen < 1 else "")
    env = TabularPOMDP(T, Obs, obs_bits, name=nm)
    env.grid, env.cells, env.idx = grid, cells, idx
    return env
