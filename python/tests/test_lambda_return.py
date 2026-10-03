"""Brute-force check of the vectorised lambda-return against the per-trace definition
(ICML'05 Eq. 6): v = (1-lam) sum_{n<k} lam^n z(n) + lam^k z(k-1), chain stops at the first
unmet action condition or when grounded in the observation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, jax, jax.numpy as jnp
from tdnet.question import symmetric_tree, chain, OBS, ANY
from tdnet.learner import TDNetLearner, LearnerConfig


def naive(q, lam, D, buf_y, buf_a, buf_o, obs_bits):
    B, N = buf_y.shape[1], q.capacity
    v = np.zeros((B, N)); has = np.zeros((B, N), bool)
    for b in range(B):
        for i in range(N):
            if not q.active[i]:
                continue
            node, targets = i, []
            for j in range(D):
                a_j = buf_a[D - 1 - j, b]
                if not (q.cond[node] == ANY or q.cond[node] == a_j):
                    break
                p = q.parent[node]
                if p == OBS:
                    targets.append(obs_bits[buf_o[D - 1 - j, b], q.obs_bit[i]]); break
                targets.append(buf_y[D - 1 - j, b, p]); node = p
            if not targets:
                continue
            has[b, i] = True
            k = len(targets)
            v[b, i] = sum((1 - lam) * lam ** n * targets[n] for n in range(k - 1)) + lam ** (k - 1) * targets[-1]
    return v, has


def run_case(q, lam, D, A, O, nbits, seed):
    rng = np.random.default_rng(seed)
    B = 5
    obs_bits = np.array([[(o >> b) & 1 for b in range(nbits)] for o in range(O)])
    L = TDNetLearner(q, A, O, obs_bits, LearnerConfig(lam=lam, max_depth=D), B)
    buf_y = rng.random((D + 1, B, q.capacity)); buf_a = rng.integers(0, A, (D + 1, B)); buf_o = rng.integers(0, O, (D + 1, B))
    v, has = L.lambda_return(jnp.asarray(buf_y, jnp.float32), jnp.asarray(buf_a), jnp.asarray(buf_o), L.tables)
    v_ref, has_ref = naive(q, lam, D, buf_y, buf_a, buf_o, obs_bits)
    assert np.array_equal(np.asarray(has), has_ref), "has_target mismatch"
    assert np.allclose(np.asarray(v) * has_ref, v_ref, atol=1e-5), f"v mismatch max {np.abs(np.asarray(v)*has_ref - v_ref).max()}"


def test_all():
    for seed in range(3):
        for lam in (0.0, 0.5, 1.0):
            run_case(symmetric_tree(2, 3), lam, 3, 2, 2, 1, seed)
            run_case(symmetric_tree(2, 3, capacity=20), lam, 5, 2, 2, 1, seed)   # spare capacity + deeper buffer
            run_case(symmetric_tree(3, 2, n_obs_bits=3), lam, 2, 3, 8, 3, seed)
            run_case(chain(5), lam, 5, 1, 2, 1, seed)
    print("lambda-return matches brute force")


if __name__ == "__main__":
    test_all()
