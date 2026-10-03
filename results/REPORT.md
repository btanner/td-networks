# TD networks revisited: JAX port, reproductions, and growing question networks

*Working notes from an overnight run. Everything here is reproducible from `python/experiments/`;
raw curves are in `results/data/*.json`, figures in `results/figs/`.*

## 1. Summary

**[filled in at the end]**

## 2. Getting the 2005 C++ running

Two problems, both small. Missing `<cstdlib>` / `<cstdio>` includes that old compilers pulled in
transitively, and real heap corruption: with `historyFeatureLength = 0` the per-node history weight
vector was empty but still indexed at slot 0 on every prediction and update
(`src/Weights.cpp`, `setupHistoryWeights`). It worked in 2005 by writing into allocator slack. With
a one-line guard the default experiment (5-state ring, depth-3 tree, Monte-Carlo traces) runs to
zero wrong predictions in 10 s. Branch `fix-build`, commit `8041b19`.

## 3. The bug in the ICML'05 trace algorithm (Figure 3)

Two things in the pseudo-code disagree with Equations 6-7 of the paper:

1. `z ← y_{t-1}[p^{t-k}(i)]` must be `y_t[p^{t-k}(i)]`: the newly available target for the trace
   started at time *k* is the ancestor's prediction *at the current time t*. The C++ does this
   correctly (it passes `y_t` into `TraceNode::learn`).
2. The gradient factor `p(1-p)` uses *p = the previous intermediate target*. Equation 7 needs the
   derivative of the prediction being corrected, `y_k^i (1 - y_k^i)`. The C++ reproduces the paper's
   version (`inverseActivate(lastValue)`); the stored `origValue` that would fix it is never read.
   On deterministic worlds this only perturbs the effective step size (both factors are in (0, ¼]),
   so it does not break the published results, but it is not the algorithm the equations describe.

The JAX port avoids the issue entirely: it never uses traces (Section 5).

## 4. Two findings about the published setup

### 4.1 The paper's feature vector cannot represent the ring world

All three papers describe the answer network as `y_t = σ(W x_t)` with
`x_t = (1, one-hot(a_{t-1}, o_t), y_{t-1})` and a *single* weight matrix. That cannot express
action-conditional state updates. For the node "bit after L" on the 5-ring, the correct update is
`y_t = [a=L]·y_{t-1}^{LL} + [a=R]·y_{t-1}^{RL}`: the action *selects* which previous predictions
feed the node, an XOR-like interaction that one linear-sigmoid layer cannot produce (the two
"positive" state sets are not nested, so no single scoring direction with a per-action bias
separates both). Confirmed by direct supervised fitting of the oracle transition:

| answer network | max |error| after fitting the oracle map on ring-5 |
|---|---|
| single matrix, paper features | 0.50 |
| one weight set per (a_{t-1}, o_t) | 0.00 |

and by learning: with paper features and a single matrix the oracle RMSE on ring-5 stays at 0.29
after 200k steps at every λ; with a weight set per (a,o) (which is what Brian's C++ actually does,
`historyWeightSetLength = 2`) it reaches 0.002; an MLP on the paper's features also solves it
(0.0015). Figure: `figs/paper_ring5_representation.png`.

### 4.2 Do not divide α by the number of features

The C++ scales every update by `1/(2 + #prediction weights)`. On the 5-ring that is α/16, and the
single-stream run needs ~10× more data than the ICML paper reports. Without the normalisation the
paper's number reproduces exactly:

| ring-5, depth 3, λ=1, single stream | steps to oracle RMSE < 0.05 |
|---|---|
| α = 0.5, normalised (C++) | > 60,000 |
| α = 0.5 | 9,000 (paper: "under 10,000") |
| α = 0.25 | 18,000 |
| α = 0.125 | 35,000 |

## 5. The JAX port: forward view instead of traces

The question network has bounded depth D, so every λ-return target for the prediction made at
time k is known at time k+D. The learner keeps ring buffers of the last D+1 steps
(x, y, a, o, weight-set index) and, at time t, computes for *every* (stream, node) at once:

* alive_j: the action conditions of i, p(i), …, p^{j}(i) all matched a_k … a_{k+j}
* z(j): the ancestor p^{j+1}(i)'s prediction at k+j+1, or the observation bit when grounded
* v = (1-λ) Σ_{j<last} λ^j z(j) + λ^last z(last)  (ICML'05 Eq. 6; last = first failed condition or grounding)
* Δθ = α (v - y_k) ∂y_k/∂θ via `jax.grad` of a masked squared error

All of it is gathers along a precomputed ancestor table `P[d, i] = p^d(i)`, masks, one einsum and
one scatter into the selected weight sets, inside a jitted `lax.scan` with the environment. A
brute-force test (`python/tests/test_lambda_return.py`) checks the vectorised λ-return against the
per-trace definition for trees, chains, spare capacity and multi-bit observations.

Cost on this 4-core CPU box, 32 parallel streams: ~20 µs/step for 14-62 nodes, 6 ms/step for 254
nodes, 18 ms/step for 510 nodes. The sequential time loop is irreducible; the win from a GPU is
width (nodes × features × streams), which is exactly the regime the bit-to-bit world needs
(Section 7).

Growing and pruning never change array shapes: nodes live in a fixed *capacity* with an `active`
mask, so the jitted step is never recompiled.

## 6. Reproducing the papers

**[filled in from exp_paper]**

## 7. The bit-to-bit gridworld is a representation problem

Before any learning curve, ask what a fixed symmetric question network *can* represent. For each
world, the number of states its depth-d questions can tell apart and the rank of the outcome
matrix (the dimension a linear PSR needs):

**[rank table]**

## 8. Gridworld experiments

**[filled in from exp_grid / exp_history_baseline]**

## 9. Larger maps, richer sensors, stochastic worlds

**[filled in from exp_big]**

## 10. Growing and pruning the question network

**[filled in from exp_grow]**

## 11. What I would do next

**[filled in at the end]**
