"""History-only baseline (IJCAI'05 Sec. 3): a separate tabular estimate of P(o_{t+1} | a_t, h) for
each unique k-step window h = (a_{t-k} o_{t-k+1} ... a_{t-1} o_t).  Evaluated against the same exact
oracle as the TD networks, on the next-observation-bit questions (one per action)."""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from .envs import TabularPOMDP, env_step


def run_history_baseline(env: TabularPOMDP, k: int, steps: int, batch: int = 32, chunk: int = 20_000, seed: int = 0, log=None):
    arrs = env.arrays()
    A, O, nb = env.n_actions, env.n_obs, env.n_obs_bits
    n_sym = A * O
    Hk = n_sym ** k
    # oracle values of the depth-1 questions: P(bit b = 1 | s, action a)  -> (nb, A, S)
    bit_prob = env.Obs @ env.obs_bits                                   # (S, nb)
    V = np.stack([np.stack([env.T[a] @ bit_prob[:, b] for a in range(A)]) for b in range(nb)])
    V = jnp.asarray(V, jnp.float32)
    obs_bits = jnp.asarray(env.obs_bits, jnp.float32)
    mult = jnp.asarray(n_sym ** np.arange(k)) if k > 0 else None

    key = jax.random.PRNGKey(seed)
    key, k1 = jax.random.split(key)
    s = jax.random.randint(k1, (batch,), 0, env.n_states)
    belief = jax.nn.one_hot(s, env.n_states)
    hist = jnp.zeros((batch, max(k, 1)), jnp.int32)
    counts = jnp.zeros((Hk, A, nb))        # sum of bits seen
    totals = jnp.zeros((Hk, A))

    def encode(h):
        return jnp.zeros(h.shape[0], jnp.int32) if k == 0 else (h[:, :k] * mult).sum(1).astype(jnp.int32)

    @partial(jax.jit, static_argnums=0)
    def run_chunk(n, s, belief, hist, counts, totals, key):
        def body(carry, _):
            s, belief, hist, counts, totals, key, acc = carry
            key, kk = jax.random.split(key)
            hidx = encode(hist)                                           # (B,)
            tot = totals[hidx]                                            # (B, A)
            pred = jnp.where(tot[..., None] > 0, counts[hidx] / jnp.maximum(tot[..., None], 1), 0.5)  # (B, A, nb)
            truth = jnp.einsum('bs,nas->ban', belief, V)                  # (B, A, nb)
            rmse = jnp.sqrt(jnp.mean((pred - truth) ** 2))
            a, s2, o, belief2 = env_step(arrs, s, belief, kk)
            bits = obs_bits[o]                                            # (B, nb)
            p_taken = pred[jnp.arange(pred.shape[0]), a]                  # (B, nb)
            err1 = jnp.mean(jnp.abs(bits - p_taken))
            counts = counts.at[hidx, a].add(bits)
            totals = totals.at[hidx, a].add(1.0)
            hist = jnp.concatenate([(a * O + o)[:, None], hist[:, :-1]], axis=1)
            return (s2, belief2, hist, counts, totals, key, (acc[0] + rmse, acc[1] + err1)), None
        (s, belief, hist, counts, totals, key, acc), _ = jax.lax.scan(body, (s, belief, hist, counts, totals, key, (0.0, 0.0)), None, length=n)
        return s, belief, hist, counts, totals, key, (acc[0] / n, acc[1] / n)

    out = dict(step=[], rmse=[], err1=[])
    done = 0
    while done < steps:
        n = min(chunk, steps - done)
        s, belief, hist, counts, totals, key, (rmse, err1) = run_chunk(n, s, belief, hist, counts, totals, key)
        done += n
        out["step"].append(done); out["rmse"].append(float(rmse)); out["err1"].append(float(err1))
        if log:
            log(f"[history k={k} {env.name}] step {done:,} 1-step oracle RMSE {float(rmse):.4f} |err| {float(err1):.4f}")
    return out
