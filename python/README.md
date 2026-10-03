# TD(λ) networks in JAX

A trace-free, batched re-implementation of temporal-difference networks
(Sutton & Tanner 2004; Tanner & Sutton 2005a,b) with history, eligibility-trace
(λ-return) learning, exact oracles for evaluation, and a growing / pruning
question network.

```
pip install "jax[cpu]" numpy matplotlib      # or a CUDA jax build
python3 tests/test_lambda_return.py          # brute-force check of the λ-return
python3 experiments/exp_paper.py             # reproduce the papers (ring / cycle worlds)
python3 experiments/exp_grid.py [steps]      # bit-to-bit gridworld sweeps
python3 experiments/exp_big.py  [steps]      # larger maps, 3-bit sensors, 4 actions, stochastic
python3 experiments/exp_grow.py              # growing / pruning vs fixed trees
```
Results (JSON + PNG) are written to `../results/`.

## How it works

* `tdnet/question.py` – question networks with fixed *capacity* (inactive slots) so growing and
  pruning never change array shapes.  Builders: `symmetric_tree`, `chain`, `ring_sparse`,
  `level0_only`.  `ancestor_table` gives `P[d, i] = p^d(i)`.
* `tdnet/envs.py` – tabular POMDPs `(T[a,s,s'], Obs[s,o], obs_bits[o,:])` stepped in batch inside
  `lax.scan`, with exact belief tracking.  `node_values` computes the true value of every question
  from the model, so RMSE against the oracle is exact.  Ring, cycle, and egocentric gridworlds
  (1-bit or 3-bit sensors, 2 or 4 actions, slip / sticky-tile dynamics, noisy sensors, as in the
  original C++ `TwoDimensionalPOMDP`).
* `tdnet/learner.py` – the learner.  Instead of eligibility traces it uses the **forward view**:
  the question network has bounded depth D, so the λ-return for the prediction made at time k is
  fully known at k+D.  Ring buffers of the last D+1 steps plus gathers along the ancestor table
  give the exact target of ICML'05 Eq. 6/7 for every (stream, node) at once; the weight update is
  `jax.grad` of the masked squared error.  Answer network: a weight set per `(a_{t-1}, o_t)`
  history (`hist_w`), optional k-step history one-hot features (`hist_f`), linear-sigmoid (paper)
  or a small MLP.  Per-node oracle-free statistics (λ-return error, context-conditional variance,
  prediction covariance) feed the grower.
* `tdnet/grow.py` – `Grower`: grow children under leaves that are answered or stuck; prune leaves
  that are constant in context, redundant with an older node (|corr| ≈ 1), or unlearnable.
* `tdnet/runner.py` – jitted `lax.scan` over environment steps with oracle metrics.

## Two things worth knowing

1. **The feature vector in the papers is not sufficient.**  `x_t = (1, onehot(a_{t-1}, o_t), y_{t-1})`
   with a single weight matrix cannot represent the ring world: the action must *select* which
   previous predictions feed a node (an XOR-like interaction).  The original C++ used a separate
   weight set per `(a, o)`; so do we (`hist_w=1`).  An MLP on the paper's features also works.
2. **Don't normalise α by the number of features.**  The C++ divides α by `|x|`; that makes the
   single-stream 5-ring take 10× longer than the paper reports.  Without it, α=0.5 reaches
   RMSE < 0.05 in 9k steps, matching the paper.
