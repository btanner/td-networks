# TD networks revisited: JAX port, reproductions, and growing question networks

*Working notes from an overnight run. Everything here is reproducible from `python/experiments/`;
raw curves are in `results/data/*.json`, figures in `results/figs/`.*

## 1. Summary

* The 2005 C++ builds and runs again (two missing includes, one real heap-corruption bug).
  The ICML'05 pseudo-code has two defects relative to its own equations; the C++ has one of them.
* The JAX port replaces eligibility traces with an exact **forward-view λ-return** (the question
  network has bounded depth, so every target is known D steps later). It is a handful of gathers
  and one einsum per step, runs 32 streams at 20 µs/step on CPU for the paper-sized networks, and
  is verified against a brute-force implementation of Eq. 6. Nodes live in a fixed capacity with
  a mask, so growing and pruning never recompile.
* **All headline results of the three papers reproduce**: λ ordering on the 5-ring (9k single-stream
  steps to RMSE .05 at λ=1, as published), TD(0) failing and λ>0 succeeding on the 8-ring sparse
  net and the 6-cycle, history rescuing TD(0) on the cycle.
* Two corrections to the published setup: the feature vector as written (single weight matrix)
  **cannot represent action-gated updates** and does not learn the ring world; the C++ actually
  used a weight set per (a_{t-1}, o_t). And the C++'s α/|x| normalisation slows learning ~16×.
* **The bit-to-bit gridworld is a representation problem first**: its linear-PSR dimension is 104
  (every state) and separating the states needs tests of length 8. No fixed tree below 510 nodes
  can hold the state, and in practice nothing we tried solves it. Best results: a 64-unit MLP
  answer network on depth-4 questions (next-observation RMSE .197, matching a 65k-row 8-step
  history table with 30 nodes). Linear networks plateau at .24–.30; TD(0) is clearly worse than
  TD(1); deeper fixed trees do not help and diverge at α=.5.
* Richer sensors help a lot (3 bits: 1-step error .04–.05 vs .13), more actions do not; a bigger but
  open map (204 states) is easy, a maze (324) is as hard as the small map. Stochastic motion and
  noisy sensors are handled gracefully against the exact belief-state oracle; the MLP answer
  network roughly halves the linear network's error on every stochastic variant.
* **Growing + pruning works where state is reachable and stalls where it is not.** On the 8-ring it
  discovers the minimal 8-question model (the paper hand-designed 16); on the empty room it keeps
  11 of 33 questions at the accuracy of the 30-node tree. On the bit-to-bit world it fills its
  capacity under "stuck" nodes without finding a core and does not beat a 30-node tree. The
  pruning rules that finally worked (prune only *answered* leaves; test redundancy by mean squared
  difference; condition "constant" statistics on the *input*; make every prune weight-preserving)
  are themselves a result: three simpler versions destroyed working networks.

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

All runs: weight set per (a_{t-1}, o_t), α = 0.5, 16 parallel streams with per-sample updates
(so "steps" means steps per stream; one learner sees 16× that much data). Oracle RMSE over all
nodes, threshold 0.05 as in ICML'05. Script: `exp_paper.py`.

| world / question net | λ | steps to RMSE < 0.05 | RMSE at end | paper says |
|---|---|---|---|---|
| ring-5, symmetric depth 3 | 0 / .25 / .5 / .75 / 1 | 5,000 for all (16 streams) | .0025 → .0019 (monotone in λ) | all solve; higher λ faster |
| ring-5, single stream | 1 | 9,000 | .014 at 60k | < 10,000 |
| ring-5, single stream | 0 | 45,000 | .008 at 300k | > 150,000 |
| ring-8, sparse 16-node (Fig. 6) | 0 | never (.30 at 600k) | | TD(0) never solves it |
| ring-8, sparse 16-node | .5 / .9 / 1 | 50k / 30k / 30k | .002 | all λ > 0 solve it |
| cycle-6, 5-chain | 0 | never (.36) | | never |
| cycle-6, 5-chain | .75 / 1 | 5,000 / 2,500 | .002 | solves, faster with λ |
| cycle-6, 5-chain | .25 / .5 | never with 16 streams; λ=.5 single-stream: 100k; λ=.25 single-stream: not in 600k | | 189k / 32k steps |
| cycle-6, TD(0) + history 1 / 2 / 3 | 0 | never | .36 – .38 | |
| cycle-6, TD(0) + history 4 | 0 | 5,000 | .002 | IJCAI: history fixes the cycle world |

Everything qualitative reproduces: the ordering in λ, the TD(0) failures on the 8-ring and the
cycle world, the rescue by λ > 0 or by history. Two quantitative differences are worth noting.
(1) On the cycle world the intermediate λ values are fragile: λ = 0.5 solves it with a single
stream but not with 16 streams whose gradients are summed, which acts like a 16× step size on a
problem whose solution has to be "stumbled upon" (IJCAI'05 §2). (2) λ = 0.25 did not solve the
cycle world within 600k single-stream steps (paper: 189k); the paper picked the best of four α
values, we used α = 0.5 throughout.

Figures: `figs/paper_ring5_lambda.png`, `paper_ring5_single_stream.png`, `paper_ring8_lambda.png`,
`paper_cycle6_lambda.png`, `paper_cycle6_history.png`, `paper_ring5_representation.png`.

## 7. The bit-to-bit gridworld is a representation problem

Before any learning curve, ask what a fixed symmetric question network *can* represent. For each
world, the number of states its depth-d questions can tell apart and the rank of the outcome
matrix (the dimension a linear PSR needs):

| world | states | depth | nodes | distinguishable states | rank of outcome matrix |
|---|---|---|---|---|---|
| ring5 | 5 | 1 | 2 | 3 | 2 |
| ring5 | 5 | 2 | 6 | 5 | 5 |
| ring8 | 8 | 1 | 2 | 3 | 2 |
| ring8 | 8 | 2 | 6 | 6 | 5 |
| ring8 | 8 | 3 | 14 | 8 | 7 |
| ring8 | 8 | 4 | 30 | 8 | 8 |
| cycle6 | 6 | 1 | 1 | 2 | 1 |
| cycle6 | 6 | 2 | 2 | 3 | 2 |
| cycle6 | 6 | 3 | 3 | 4 | 3 |
| cycle6 | 6 | 4 | 4 | 5 | 4 |
| cycle6 | 6 | 5 | 5 | 6 | 5 |
| cycle6 | 6 | 6 | 6 | 6 | 6 |
| room4 | 64 | 1 | 2 | 4 | 2 |
| room4 | 64 | 2 | 6 | 9 | 5 |
| room4 | 64 | 3 | 14 | 12 | 6 |
| room4 | 64 | 4 | 30 | 16 | 7 |
| room4 | 64 | 5 | 62 | 16 | 7 |
| room4 | 64 | 6 | 126 | 16 | 7 |
| room4 | 64 | 7 | 254 | 16 | 7 |
| room4 | 64 | 8 | 510 | 16 | 7 |
| ijcai26 (bit-to-bit) | 104 | 1 | 2 | 4 | 2 |
| ijcai26 (bit-to-bit) | 104 | 2 | 6 | 22 | 6 |
| ijcai26 (bit-to-bit) | 104 | 3 | 14 | 42 | 14 |
| ijcai26 (bit-to-bit) | 104 | 4 | 30 | 65 | 28 |
| ijcai26 (bit-to-bit) | 104 | 5 | 62 | 84 | 51 |
| ijcai26 (bit-to-bit) | 104 | 6 | 126 | 95 | 85 |
| ijcai26 (bit-to-bit) | 104 | 7 | 254 | 103 | 102 |
| ijcai26 (bit-to-bit) | 104 | 8 | 510 | 104 | 104 |
| tworooms | 204 | 1 | 2 | 4 | 2 |
| tworooms | 204 | 2 | 6 | 16 | 6 |
| tworooms | 204 | 3 | 14 | 34 | 14 |
| tworooms | 204 | 4 | 30 | 51 | 28 |
| tworooms | 204 | 5 | 62 | 63 | 47 |
| tworooms | 204 | 6 | 126 | 77 | 63 |
| tworooms | 204 | 7 | 254 | 90 | 78 |
| tworooms | 204 | 8 | 510 | 99 | 91 |
| maze12 | 324 | 1 | 2 | 4 | 2 |
| maze12 | 324 | 2 | 6 | 19 | 6 |
| maze12 | 324 | 3 | 14 | 71 | 14 |
| maze12 | 324 | 4 | 30 | 147 | 28 |
| maze12 | 324 | 5 | 62 | 227 | 52 |
| maze12 | 324 | 6 | 126 | 275 | 93 |
| maze12 | 324 | 7 | 254 | 306 | 154 |
| maze12 | 324 | 8 | 510 | 316 | 257 |

Three things follow. The 8-ring is fully represented by a depth-4 tree (which is why the fixed
30-node tree learns it; the paper's 16-node sparse net is just cheaper). The empty 4x4 room has
only 16 distinguishable state classes and rank 7, so it is *easy* despite 64 states: the depth-4
tree should solve it and does (Section 10). The 26-cell bit-to-bit map needs tests of length 8 to
separate its 104 states and its outcome matrix has full rank 104: a linear PSR of this world has
dimension 104, and no fixed tree below depth 8 (510 nodes) can hold the state. Every plateau in
Section 8 is a floor, not slow learning. Script: `analysis_rank.py`.

## 8. Gridworld experiments

26-cell map, 1-bit sensor (wall ahead), actions forward / turn-right, 104 states, 32 streams,
400k steps per stream (curves are flat after ~100k). Oracle RMSE over all nodes; "next-obs" is the
RMSE of the depth-1 nodes only, which is the quantity a history predictor can be compared on.
Scripts: `exp_grid.py`, `exp_alpha.py`, `exp_history_baseline.py`.

**Fixed symmetric trees, linear answer network, λ = 1.**

| depth (nodes) | history feats | α | all-node RMSE | next-obs RMSE | 1-step \|err\| |
|---|---|---|---|---|---|
| 2 (6) | 0 / 2 / 3 | .5 | .307 / .297 / .285 | .285 / .278 / .268 | .147 / .139 / .130 |
| 3 (14) | 0 / 2 / 3 | .5 | .335 / .331 / .327 | .296 / .291 / .288 | .130 / .126 / .125 |
| 4 (30) | 0 / 2 / 3 | .5 | .354 / .350 / .344 | .303 / .294 / .291 | .107 / .101 / .101 |
| 4 (30) | 0 | .1 | .295 | .244 | .112 |
| 4 (30) | weight set per 2-step history (16 sets) | .5 | .322 | .268 | .096 |
| 5 (62) | 0 / 2 / 3 | .5 | .479 / .472 / .433 (diverging) | .357 / .309 / .288 | .130 / .098 / .086 |
| 5 (62) | 0 / 3 | .1 | .357 / .344 | .284 / .270 | .123 / .113 |
| 6 (126) | 0 / 3 | .05 | .376 / .360 | .288 / .271 | .123 / .111 |

Deeper fixed trees do not help, as the rank analysis predicts, and above ~60 nodes α = 0.5 with
32 summed streams diverges (deepest nodes worse than 0.5). λ matters here too: at depth 4,
history 2, α = 0.1 the all-node RMSE is .406 (λ=0), .311 (λ=.5), .295 (λ=1).

**Answer network.** Replacing the linear-sigmoid layer by a 64-unit tanh MLP on the same depth-4
questions (history-2 features, Adam 1e-3) gives all-node RMSE **.249**, next-obs **.197**, 1-step
|err| **.079**: the best result on this world by a wide margin. Plain SGD on the MLP at α = 0.1
diverged (.63), so the optimiser matters. Figure: `figs/grid_answer_nets.png`.

**History-only baseline** (tabular P(o | k-step window, a), same oracle, next-obs RMSE):

| k | 1 | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| table rows | 4 | 16 | 64 | 256 | 1,024 | 4,096 | 65,536 |
| next-obs RMSE | .372 | .353 | .310 | .276 | .252 | .230 | .200 |

So a 30-node linear TD network (next-obs .244) sits between history 4 and 5, and the 30-node MLP
TD network (.197) matches an 8-step, 65k-row history table. Neither solves the world; the IJCAI
paper's "suggestive" verdict stands, and Section 7 says why.

## 9. Larger maps, richer sensors, stochastic worlds

Same learner, α = 0.5, λ = 1, 32 streams, 400k steps. Script: `exp_big.py`.

**Bigger maps are not necessarily harder.** The two-rooms map (204 states) is far easier than the
26-cell map because most of its states are in open space where every short test predicts "no
wall": depth 4 gives all-node RMSE .10 and 1-step error .010. The 12x12 maze (324 states, rank
257 at depth 8) is as hard as the small map: .35 / .09.

| map (states) | depth 3 | depth 4 | depth 4 + history 3 |
|---|---|---|---|
| 26-cell (104) | .335 | .354 | .344 |
| two rooms (204) | .155 | .101 | .098 |
| maze 12x12 (324) | .348 | .357 | .346 |

**Richer sensors help more than more actions.** On the 26-cell map, the C++ "sensor fusion"
observation (wall ahead / left / right, 8 symbols) with a depth-2 tree (18 nodes) reaches 1-step
error .054 against .13 for the 1-bit sensor at depth 3; adding backward and turn-left actions
(4 actions, 60 nodes at depth 2) gives .042. Four actions with the 1-bit sensor (84 nodes, depth 3)
stay at .090. Figure: `figs/big_obs_actions.png`.

**Stochastic dynamics and noisy sensors** (C++ constants: slip 0.1 with 0.02 double moves; wall
seen as wall 0.925, open seen as open 0.9). The oracle is the exact belief-state prediction, so
RMSE 0 remains attainable in principle. Linear answer networks degrade mildly; the MLP again
roughly halves the error. The 1-step |err| column is dominated by irreducible sensor noise.

| variant (depth 3, history 2) | linear λ=1 | linear λ=.5 | linear depth 4 | MLP-64 + Adam |
|---|---|---|---|---|
| deterministic | .331 | | .350 | .249 (depth 4) |
| slip 0.1 (+0.02) | .364 | .389 | .335 | **.194** |
| noisy sensor | .399 | .435 | .378 | **.259** |
| slip + noisy sensor | .335 | .350 | .356 | **.146** |

Figures: `figs/big_maps_1bit.png`, `figs/big_stochastic.png`.

## 10. Growing and pruning the question network

`tdnet/grow.py`, script `exp_grow.py`. The network starts from the depth-1 nodes only. Every
round (20k–40k steps) it uses only statistics the learner keeps anyway, none of them oracle-based:

* **grow** under a leaf that is *answered* (λ-return MSE < .005, so its children can bootstrap from
  it) or *stuck* (error improved < 5 % over the round: the current state is insufficient), at
  most 4 parents per round, breadth-first, never beyond `max_depth`;
* **prune** a leaf that is answered and either *constant given its context* (the input y_{t-1}^i
  is a deterministic function of (a_{t-1}, o_t), so it is folded exactly into that weight set's
  bias) or *redundant* (E[(y_i − y_j)^2] < 1e-4 with an older answered node; its input weights are
  added onto the twin's), or that is still unlearnable after 6 rounds although its parent is
  answered. Pruned (parent, action) pairs are tomb-stoned.

Three versions of the pruning rule failed before this one, and the failures are instructive
(all on the empty room, where a depth-4 tree is known to suffice):
correlation-based redundancy (> .98) merged questions that differ in rare states; judging
unlearned nodes let the early, all-correlated predictions prune half the network; and
"constant given the current context" was conceptually wrong, because a node whose *prediction*
is fixed by (a_{t-1}, o_t) is used one step later, where it carries o_t forward as a history
feature. Conditioning on the *input* and making every prune weight-preserving fixed it.

| world | fixed trees (nodes → next-obs RMSE) | grown + pruned | grown, no pruning |
|---|---|---|---|
| 8-ring | depth 4 (30) .0005; depth 6 (126) .0003; paper's sparse (16) .0005 | **8 nodes**, .0007 | 64 nodes (capacity), .0004 |
| empty 4x4 room | depth 2 (6) .22; depth 3 (14) .12; depth 4 (30) .0012 | **11 nodes**, .0019 | |
| bit-to-bit (104), α=.1 | depth 4 (30) .234; depth 5 (62) .280; depth 6 (126) .278; depth 7 (254) .320 | 159 nodes, depth 7, .266 | *(still running when this was written; see `exp_grow.json`)* |

The grown 8-ring network is exactly the minimal model: L, LL, LLL, LLLL and R, RR, RRR (LLLL =
RRRR), eight questions where the ICML paper hand-designed sixteen. On the room it keeps 11 of the
33 questions it tried (F, R, FF, RF, RR, RRR, RRF, RRFF, RFF, RRRF, RRRR) at the same accuracy as
the 30-node tree.

On the bit-to-bit world the grower does **not** crack the problem. It fills its 160-node capacity
in ~1.8M steps with a network that is a full tree to depth 4 (30 nodes) and then sparse: 29/32
nodes at depth 5, 53/64 at depth 6, 47/128 at depth 7. Almost all growth there is triggered by
"stuck", not "answered": with insufficient state nothing deep becomes answerable, so the grower
has no signal for *which* deep questions to add and ends up approximating the full tree. Its
next-observation RMSE (.266) is no better than the 30-node fixed tree (.234). Over the whole
experiment (ring, room, gridworld) 79 prunes were redundancy merges, 39 were constant-in-context
folds and 2 were unlearnable; on the gridworld pruning is rare, because few deep nodes ever
become answered. Figures: `figs/grow_ring8.png`, `figs/grow_room4.png`, `figs/grow_grid.png`.

## 11. What I would do next

1. **Run the full-rank network on a GPU.** The depth-8 tree (510 nodes, rank 104) is the first
   fixed network that can represent the bit-to-bit world; at 18 ms/step on this CPU it needs five
   hours per million steps, on a GPU it is minutes. This is the single cleanest test of whether
   the learning algorithm or the question network is the bottleneck. The code needs no changes.
2. **A better growth signal.** "Stuck" says *that* state is missing, not *which* question supplies
   it. A direct usefulness test is cheap in the forward view: ablate a candidate node's input
   column and measure the change in the λ-return loss on the ring buffer, or score candidates by
   how much they reduce the error of the depth-1 nodes. Grow the few with the largest effect.
3. **MLP answer network inside the grower**, with Adam; it was the best learner everywhere, and
   the grower currently only uses the linear one.
4. **Factor the gating.** A weight set per (a,o) history grows as (|A||O|)^k. A multiplicative
   (bilinear) layer action × previous-predictions would give the same gating with |A| factors, and
   leave observations and history as additive features.
5. **Sort out λ at intermediate values with many streams.** Summed multi-stream updates act as a
   B× step size; λ=0.5 solves the cycle world single-stream but not with 16 streams. Either scale
   α by 1/B and accept slower per-stream progress, or use Adam for the linear model too.
6. **Curriculum from rich to poor sensors**: the 3-bit sensor is nearly solved at depth 2; the
   predictions it learns are exactly the state the 1-bit agent lacks.
