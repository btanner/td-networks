"""Run a TD network against a batched tabular POMDP with exact oracle evaluation, inside a jitted
lax.scan.  Returns learning curves (per chunk): oracle RMSE over all active nodes, per-node MSE,
and the empirical 1-step error |bit - p(bit | action taken)|."""
from __future__ import annotations
import time
import numpy as np
import jax
import jax.numpy as jnp
from .envs import TabularPOMDP, env_step
from .question import QuestionNet
from .learner import TDNetLearner, LearnerConfig


def level0_table(q: QuestionNet, n_actions: int, n_obs_bits: int) -> np.ndarray:
    """L0[b, a] = node predicting bit b under action a (depth 1), or -1."""
    L0 = np.full((n_obs_bits, n_actions), -1, dtype=np.int32)
    for i in range(q.capacity):
        if q.active[i] and q.parent[i] == -1:
            if q.cond[i] >= 0:
                L0[q.obs_bit[i], q.cond[i]] = i
            else:
                L0[q.obs_bit[i], :] = i
    return L0


class Experiment:
    def __init__(self, env: TabularPOMDP, q: QuestionNet, cfg: LearnerConfig, batch: int = 32, seed: int = 0):
        self.env, self.q, self.cfg, self.B = env, q, cfg, batch
        self.learner = TDNetLearner(q, env.n_actions, env.n_obs, env.obs_bits, cfg, batch)
        self.arrs = env.arrays()
        key = jax.random.PRNGKey(seed)
        key, k1, k2 = jax.random.split(key, 3)
        self.key = key
        s0 = jax.random.randint(k1, (batch,), 0, env.n_states)
        self.s = s0
        self.belief = jax.nn.one_hot(s0, env.n_states)
        self.state = self.learner.init(k2)
        self.steps_done = 0
        self.refresh_tables()
        self._run_chunk = jax.jit(self._run_chunk_impl, static_argnums=(0,))

    def refresh_tables(self):
        self.V = jnp.asarray(self.env.node_values(self.q.action_sequences()), jnp.float32)  # (N, S)
        self.L0 = jnp.asarray(level0_table(self.q, self.env.n_actions, self.env.n_obs_bits))

    def set_question(self, q: QuestionNet, reset_nodes=(), merges=(), folds=()):
        self.q = q
        self.state = self.learner.set_question(q, self.state, reset_nodes, merges, folds)
        self.refresh_tables()

    def _run_chunk_impl(self, n_steps, state, s, belief, key, V, L0, tables, alpha):
        learner = self.learner
        state = dict(state, alpha=alpha)
        active = tables["active"]
        n_active = jnp.maximum(active.sum(), 1.0)
        obs_bits = learner.obs_bits

        def body(carry, _):
            state, s, belief, key, acc = carry
            key, k = jax.random.split(key)
            y = state["y"]                                           # y_t (B, N)
            truth = belief @ V.T                                     # (B, N) oracle values
            sq = jnp.mean((y - truth) ** 2, axis=0) * active         # (N,)
            rmse = jnp.sqrt(jnp.sum(sq) / n_active)
            a, s2, o, belief2 = env_step(self.arrs, s, belief, k)
            bits = obs_bits[o]                                       # (B, nbits)
            node = L0[:, a].T                                        # (B, nbits)
            p = jnp.take_along_axis(y, jnp.clip(node, 0, None), axis=1)
            valid = (node >= 0).astype(jnp.float32)
            err1 = jnp.sum(jnp.abs(bits - p) * valid) / jnp.maximum(valid.sum(), 1)
            wrong = jnp.sum((jnp.abs(bits - p) > 0.5) * valid) / jnp.maximum(valid.sum(), 1)
            state = learner._step_impl(state, a, o, tables)
            acc = (acc[0] + rmse, acc[1] + sq, acc[2] + err1, acc[3] + wrong)
            return (state, s2, belief2, key, acc), None

        acc0 = (jnp.float32(0), jnp.zeros(learner.N), jnp.float32(0), jnp.float32(0))
        (state, s, belief, key, acc), _ = jax.lax.scan(body, (state, s, belief, key, acc0), None, length=n_steps)
        return state, s, belief, key, tuple(a / n_steps for a in acc)

    def run(self, steps: int, chunk: int = 5000, alpha_schedule=None, log=None):
        """Run `steps` env steps per batch element.  alpha_schedule(step) -> alpha (optional)."""
        hist = dict(step=[], rmse=[], node_mse=[], err1=[], wrong=[], alpha=[])
        t0 = time.time()
        while self.steps_done < steps:
            n = min(chunk, steps - self.steps_done)
            alpha = self.cfg.alpha if alpha_schedule is None else alpha_schedule(self.steps_done)
            self.state, self.s, self.belief, self.key, acc = self._run_chunk(
                n, self.state, self.s, self.belief, self.key, self.V, self.L0, self.learner.tables, jnp.float32(alpha))
            self.steps_done += n
            rmse, sq, err1, wrong = (np.asarray(a) for a in acc)
            hist["step"].append(self.steps_done); hist["rmse"].append(float(rmse)); hist["node_mse"].append(sq)
            hist["err1"].append(float(err1)); hist["wrong"].append(float(wrong)); hist["alpha"].append(alpha)
            if log:
                log(f"[{self.env.name} N={self.q.n_active}] step {self.steps_done:>9,d}  oracleRMSE {rmse:.4f}  "
                    f"1-step|err| {err1:.4f}  wrong {wrong:.4f}  ({time.time() - t0:.0f}s)")
        hist["node_mse"] = np.array(hist["node_mse"])
        return hist


def run_experiment(env, q, cfg, batch=32, steps=200_000, chunk=5000, seed=0, log=print, alpha_schedule=None):
    ex = Experiment(env, q, cfg, batch, seed)
    hist = ex.run(steps, chunk, alpha_schedule, log)
    return ex, hist
