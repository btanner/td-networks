"""Forward-view TD(lambda) network learner in JAX.

The question network has bounded depth D, so every lambda-return target for the prediction made
at time k is available at time k + D.  We keep ring buffers of the last D+1 steps and apply the
exact forward-view update of Eq. 7 (Tanner & Sutton, ICML 2005) to y_{t-D} at time t:

    v_i = (1-lam) * sum_{n<last} lam^n z(n) + lam^last z(last)        (lambda-return)
    dtheta = alpha * (v_i - y_i) * dy_i/dtheta                        (via jax.grad)

where z(n) is the n-th unrolled target (the n+1-th ancestor's prediction at time k+n+1, or the
observation bit once the chain is grounded) and the chain stops at the first unmet action
condition.  This is trace-free, exact, and vectorised over (batch, node, feature).
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
import jax
import jax.numpy as jnp
from .question import QuestionNet, OBS, NONE, ANY


@dataclass(frozen=True)
class LearnerConfig:
    alpha: float = 0.5
    lam: float = 1.0
    hist_w: int = 1        # (a,o) pairs selecting the weight set.  0 -> one weight set
    hist_f: int = 0        # (a,o) pairs fed as one-hot input features
    model: str = "linear"  # 'linear' (paper) or 'mlp'
    hidden: int = 32
    normalize: bool = False  # divide alpha by #features (the C++ did; it slows learning ~|x| times)
    optimizer: str = "sgd" # 'sgd' or 'adam'
    max_depth: int = 4     # ring-buffer length D (>= deepest question)
    init_scale: float = 0.0
    stored_pred: bool = True  # error uses the stored y_k (faithful to C++); False recomputes y(x_k; theta_now)
    err_decay: float = 0.999  # EMA constant for the per-node lambda-return error statistics (used for growing/pruning)


class TDNetLearner:
    def __init__(self, q: QuestionNet, n_actions: int, n_obs: int, obs_bits: np.ndarray, cfg: LearnerConfig, batch: int):
        self.q, self.A, self.O, self.cfg, self.B = q, n_actions, n_obs, cfg, batch
        self.obs_bits = jnp.asarray(obs_bits, jnp.float32)   # (O, nbits)
        self.N = q.capacity
        self.K = max(cfg.hist_w, cfg.hist_f, 1)              # history symbols kept
        self.n_sym = n_actions * n_obs
        self.H = self.n_sym ** cfg.hist_w
        self.F = self.n_sym ** cfg.hist_f if cfg.hist_f > 0 else 0
        self.M = 1 + self.F + self.N
        self.D = cfg.max_depth
        self.tables = self.make_tables(q)
        self.lam_pow = jnp.asarray(cfg.lam ** np.arange(self.D + 1), jnp.float32)
        self._step = jax.jit(self._step_impl)

    # ------------------------------------------------------------------ tables
    def make_tables(self, q: QuestionNet):
        D = self.D
        assert q.max_depth() <= D, f"question depth {q.max_depth()} > max_depth {D}"
        P = q.ancestor_table(D)                                   # (D+1, N)
        condP = np.where(P >= 0, q.cond[np.clip(P, 0, None)], ANY)  # condition of p^d(i)
        grounded = (P[1:] == OBS)                                 # target at step d is the observation
        tgt = np.clip(P[1:], 0, None)                             # node index of target (if not grounded)
        return dict(P=jnp.asarray(P), condP=jnp.asarray(condP), grounded=jnp.asarray(grounded),
                    tgt=jnp.asarray(tgt), active=jnp.asarray(q.active, jnp.float32),
                    obs_bit=jnp.asarray(q.obs_bit))

    def set_question(self, q: QuestionNet, state, reset_nodes=()):
        """Swap in a grown/pruned question network (same capacity).  Weights of `reset_nodes`
        (their output rows and their input columns) are reset so re-used slots start fresh."""
        self.q = q
        self.tables = self.make_tables(q)
        if len(reset_nodes):
            idx = jnp.asarray(np.array(reset_nodes, dtype=np.int32))
            p = dict(state["params"])
            if self.cfg.model == "linear":
                W = p["W"].at[:, idx, :].set(0.0)
                W = W.at[:, :, 1 + self.F + idx].set(0.0)
                p["W"] = W
            else:
                p["W2"] = p["W2"].at[:, :, idx].set(0.0); p["b2"] = p["b2"].at[:, idx].set(0.0)
                p["W1"] = p["W1"].at[:, 1 + self.F + idx, :].set(0.0)
            state = dict(state, params=p, y=state["y"] * self.tables["active"][None, :],
                         node_err=state["node_err"].at[idx].set(0.0), node_cnt=state["node_cnt"].at[idx].set(0.0),
                         node_mean=state["node_mean"].at[idx].set(0.5), node_var=state["node_var"].at[idx].set(0.0),
                         ym_h=state["ym_h"].at[:, idx].set(0.5), ysq_h=state["ysq_h"].at[:, idx].set(0.25),
                         yy=state["yy"].at[idx, :].set(0.0).at[:, idx].set(0.0))
            if self.cfg.optimizer == "adam":
                state = dict(state, m=jax.tree_util.tree_map(jnp.zeros_like, state["m"]),
                             v=jax.tree_util.tree_map(jnp.zeros_like, state["v"]))
        return state

    # ------------------------------------------------------------------ params / state
    def init(self, key):
        cfg, H, M, N = self.cfg, self.H, self.M, self.N
        k1, k2 = jax.random.split(key)
        if cfg.model == "linear":
            params = {"W": cfg.init_scale * jax.random.normal(k1, (H, N, M))}
        else:
            hid = cfg.hidden
            params = {"W1": jax.random.normal(k1, (H, M, hid)) * (1.0 / np.sqrt(M)), "b1": jnp.zeros((H, hid)),
                      "W2": jax.random.normal(k2, (H, hid, N)) * (cfg.init_scale / np.sqrt(hid)), "b2": jnp.zeros((H, N))}
        B, D = self.B, self.D
        state = dict(params=params, y=jnp.zeros((B, N)), hist=jnp.zeros((B, self.K), jnp.int32),
                     buf_x=jnp.zeros((D + 1, B, M)), buf_y=jnp.zeros((D + 1, B, N)),
                     buf_a=jnp.zeros((D + 1, B), jnp.int32), buf_o=jnp.zeros((D + 1, B), jnp.int32),
                     buf_h=jnp.zeros((D + 1, B), jnp.int32), t=jnp.int32(0), alpha=jnp.float32(cfg.alpha),
                     node_err=jnp.zeros(N), node_cnt=jnp.zeros(N), node_var=jnp.zeros(N), node_mean=jnp.full(N, 0.5),
                     ym_h=jnp.full((self.H, N), 0.5), ysq_h=jnp.full((self.H, N), 0.25), yy=jnp.zeros((N, N)))
        if cfg.optimizer == "adam":
            state["m"] = jax.tree_util.tree_map(jnp.zeros_like, params)
            state["v"] = jax.tree_util.tree_map(jnp.zeros_like, params)
        return state

    # ------------------------------------------------------------------ model
    def forward(self, params, x, h):
        """x: (B, M), h: (B,) weight-set index -> predictions (B, N)."""
        if self.cfg.model == "linear":
            s = jnp.einsum("bnm,bm->bn", params["W"][h], x)
        else:
            z = jnp.tanh(jnp.einsum("bmk,bm->bk", params["W1"][h], x) + params["b1"][h])
            s = jnp.einsum("bkn,bk->bn", params["W2"][h], z) + params["b2"][h]
        return jax.nn.sigmoid(s)

    def _encode(self, hist, n):
        if n == 0:
            return jnp.zeros(hist.shape[0], jnp.int32)
        mult = self.n_sym ** jnp.arange(n)
        return (hist[:, :n] * mult).sum(1).astype(jnp.int32)

    def features(self, y, hist, tables):
        parts = [jnp.ones((y.shape[0], 1))]
        if self.F:
            parts.append(jax.nn.one_hot(self._encode(hist, self.cfg.hist_f), self.F))
        parts.append(y * tables["active"][None, :])
        return jnp.concatenate(parts, axis=1)

    # ------------------------------------------------------------------ lambda-return
    def lambda_return(self, buf_y, buf_a, buf_o, tb):
        """Targets for the prediction made at slot D (time k = t-D).  Returns (v, has_target) (B, N)."""
        D, lam = self.D, self.cfg.lam
        B = buf_a.shape[1]
        alive = tb["active"][None, :] > 0
        v = jnp.zeros((B, self.N)); remaining = jnp.ones((B, self.N))
        for j in range(D):
            a_j = buf_a[D - 1 - j]                                   # action taken at time k+j
            cj = tb["condP"][j][None, :]
            alive = alive & ((cj == ANY) | (cj == a_j[:, None]))
            if j == 0:
                has_target = alive
            y_next, o_next = buf_y[D - 1 - j], buf_o[D - 1 - j]       # outcome at time k+j+1
            z_pred = jnp.take_along_axis(y_next, jnp.broadcast_to(tb["tgt"][j][None, :], (B, self.N)), axis=1)
            z_obs = self.obs_bits[o_next][:, tb["obs_bit"]]           # (B, N) bit of the observation
            z = jnp.where(tb["grounded"][j][None, :], z_obs, z_pred)
            nxt_alive = alive & ~tb["grounded"][j][None, :]
            if j + 1 < D:
                cn = tb["condP"][j + 1][None, :]
                nxt_alive = nxt_alive & ((cn == ANY) | (cn == buf_a[D - 2 - j][:, None]))
            else:
                nxt_alive = jnp.zeros_like(nxt_alive)
            last = alive & ~nxt_alive
            w = jnp.where(last, remaining, (1 - lam) * self.lam_pow[j])
            v = v + jnp.where(alive, w * z, 0.0)
            remaining = remaining - jnp.where(alive, (1 - lam) * self.lam_pow[j], 0.0)
            alive = nxt_alive
        return v, has_target

    # ------------------------------------------------------------------ step
    def _step_impl(self, state, a, o, tb=None):
        """Consume (a_{t-1}, o_t): compute y_t, then update the prediction made D steps ago."""
        cfg, D = self.cfg, self.D
        tb = self.tables if tb is None else tb
        hist = jnp.concatenate([(a * self.O + o)[:, None], state["hist"][:, :-1]], axis=1)
        h = self._encode(hist, cfg.hist_w)
        x = self.features(state["y"], hist, tb)
        y = self.forward(state["params"], x, h) * tb["active"][None, :]
        roll = lambda buf, v: jnp.concatenate([v[None], buf[:-1]], axis=0)
        buf_x, buf_y = roll(state["buf_x"], x), roll(state["buf_y"], y)
        buf_a, buf_o, buf_h = roll(state["buf_a"], a), roll(state["buf_o"], o), roll(state["buf_h"], h)
        t = state["t"] + 1

        v, has_target = self.lambda_return(buf_y, buf_a, buf_o, tb)
        xk, hk, yk = buf_x[D], buf_h[D], buf_y[D]
        mask = has_target.astype(jnp.float32) * tb["active"][None, :] * (t > D).astype(jnp.float32)

        def loss_fn(params):
            yhat = self.forward(params, xk, hk)
            target_err = v - (yk if cfg.stored_pred else yhat)
            # d/dtheta of 0.5*(v - yhat)^2 with the error evaluated at yk if stored_pred
            # summed (not averaged) over the batch: each stream gets a per-sample update of size alpha,
            # like the single-stream algorithm in the papers
            return -jnp.sum(mask * jax.lax.stop_gradient(target_err) * yhat)

        grads = jax.grad(loss_fn)(state["params"])
        # per-node statistics (oracle-free): EMA of the squared lambda-return error, of how often the node
        # has a target, and of the mean / variance of its own prediction (a constant node carries no state)
        dcy = cfg.err_decay
        err2 = jnp.sum(mask * (v - yk) ** 2, axis=0) / jnp.maximum(mask.sum(0), 1.0)
        had = (mask.sum(0) > 0).astype(jnp.float32)
        node_err = jnp.where(had > 0, dcy * state["node_err"] + (1 - dcy) * err2, state["node_err"])
        node_cnt = dcy * state["node_cnt"] + (1 - dcy) * had
        ym = jnp.mean(y, axis=0)
        node_mean = dcy * state["node_mean"] + (1 - dcy) * ym
        node_var = dcy * state["node_var"] + (1 - dcy) * jnp.mean((y - state["node_mean"][None, :]) ** 2, axis=0)
        # context-conditional statistics: mean / second moment of y given the weight-set index h, and the
        # second-moment matrix y y^T (for redundancy detection when growing / pruning)
        cnt_h = jnp.zeros((self.H,)).at[h].add(1.0)
        sum_h = jnp.zeros((self.H, self.N)).at[h].add(y)
        sq_h = jnp.zeros((self.H, self.N)).at[h].add(y * y)
        seen = (cnt_h > 0)[:, None]
        ym_h = jnp.where(seen, dcy * state["ym_h"] + (1 - dcy) * sum_h / jnp.maximum(cnt_h, 1)[:, None], state["ym_h"])
        ysq_h = jnp.where(seen, dcy * state["ysq_h"] + (1 - dcy) * sq_h / jnp.maximum(cnt_h, 1)[:, None], state["ysq_h"])
        yy = dcy * state["yy"] + (1 - dcy) * (y.T @ y) / y.shape[0]
        scale = state["alpha"] / (self.M if cfg.normalize else 1.0)
        params, new_state = state["params"], {}
        if cfg.optimizer == "sgd":
            params = jax.tree_util.tree_map(lambda p, g: p - scale * g, params, grads)
        else:
            b1, b2, eps = 0.9, 0.999, 1e-8
            m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state["m"], grads)
            vv = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, state["v"], grads)
            tt = t.astype(jnp.float32)
            params = jax.tree_util.tree_map(
                lambda p, m_, v_: p - state["alpha"] * (m_ / (1 - b1 ** tt)) / (jnp.sqrt(v_ / (1 - b2 ** tt)) + eps),
                params, m, vv)
            new_state.update(m=m, v=vv)
        new_state.update(params=params, y=y, hist=hist, buf_x=buf_x, buf_y=buf_y, buf_a=buf_a, buf_o=buf_o,
                         buf_h=buf_h, t=t, alpha=state["alpha"], node_err=node_err, node_cnt=node_cnt,
                         node_mean=node_mean, node_var=node_var, ym_h=ym_h, ysq_h=ysq_h, yy=yy)
        return new_state

    def input_usage(self, params):
        """How much each node's prediction is used as an input by the answer network: mean |weight| on its
        input column across weight sets and output nodes (linear) or hidden units (mlp)."""
        col = slice(1 + self.F, 1 + self.F + self.N)
        if self.cfg.model == "linear":
            return jnp.mean(jnp.abs(params["W"][:, :, col]), axis=(0, 1))
        return jnp.mean(jnp.abs(params["W1"][:, col, :]), axis=(0, 2))

    def step(self, state, a, o):
        return self._step(state, a, o, self.tables)
