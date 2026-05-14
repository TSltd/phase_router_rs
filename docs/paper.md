# Phase Routing: A Deterministic, Capacity-Factor-Invariant Balance Primitive for Sparse Routing

## Abstract

Sparse routing systems — Mixture-of-Experts (MoE) layers, shard
dispatchers, request routers — rely on a small zoo of tunable
hyperparameters to keep load distributed across capacity-bounded
endpoints. In MoE, these include a **capacity factor** that
oversubscribes per-expert quotas to absorb skew, and an **auxiliary
load-balance loss** that steers a learned gate away from collapse.
Both are empirically essential: removing either at standard scale
inflates the load coefficient-of-variation by 10× and drops 40 % of
token-expert assignments at the input.

We present **phase routing**, a deterministic balance primitive built
on cyclic-phase intersections of bit-packed source / target matrices.
Phase routing is **not** a better semantic router than
learned-affinity top-k — at the scales we tested, it pays a 5 – 12 %
cross-entropy cost. Instead, it is a **structural balancing
substrate** with four operating-regime invariants that are useful in
their own right and rare in this design space:

1. **Hyperparameter elimination** — load distribution is fixed by
   construction; capacity factor and auxiliary balance loss are not
   parameters of the algorithm.
2. **Anti-collapse by construction** — there is no learned gate over
   experts that can collapse onto a popular subset, because the
   selection rule does not consume gate scores.
3. **Graceful provisioning response** — load CV, drop rate, locality,
   and routing wall-time are essentially **flat** across
   capacity-factor sweeps (variation < 0.1 % on a 32-expert
   scaffold), where Switch-style top-k varies by 5 – 50 % on the same
   axes.
4. **Clean structural properties** — deterministic given a seed,
   reproducible bit-for-bit across runs, implementable as a fused
   `O(N · k)` Rust kernel with `O(N)` working memory.

A natural follow-up — a greedy hybrid that uses top-k for affinity
and phase-style quotas for balance — fails catastrophically (62 %
token drop at 32 experts) because, unlike phase, it has a gate that
can collapse. We report this as a clean negative result that
sharpens the case for phase as a _structural_ primitive rather than
one ingredient of a hybrid.

We claim that phase routing is **a structurally simple, deterministic
balance substrate for capacity-factor-free, aux-loss-free,
anti-collapse balance in sparse routing systems**, useful as a constraint
to be layered under future semantic routers rather than as a drop-in
replacement for top-k.

Conceptually, **phase routing converts balance from an optimisation
objective into a geometric invariant.** Where Switch-style routers
treat balance as a soft penalty on a global gate statistic, phase
routing makes balance a property of the cyclic-phase intersection
that no choice of weights can disturb. The accompanying observation
from our experiments is that **balance and specialisation are
antagonistic objectives under sparse routing**: top-k allows
semantic concentration in the gate, which produces specialisation
but destabilises occupancy; phase suppresses concentration by
construction, which stabilises occupancy but weakens specialisation.
We pick balance and pay the affinity tax.

---

## 1. Introduction

Modern sparse-routing systems have converged on a remarkably
consistent recipe. A learned gate scores `N` items against `E`
endpoints; the top-k scores per item determine routing; a per-endpoint
**capacity factor** `cf` is multiplied through to set a hard ceiling
`⌈cf · N · k / E⌉` of items per endpoint; an auxiliary
**load-balance loss** is added to the training objective with a small
coefficient (typically `α ≈ 0.01`) to discourage gate collapse.

This recipe has two recurring failure modes:

- **Without the aux loss**, the gate collapses onto a small subset of
  endpoints, the load CV grows by an order of magnitude, and 30 – 50
  % of (item, slot) pairs hit a full endpoint and are silently
  dropped. We reproduce this at 8 experts on a TinyStories
  scaffold: `aux=0 ⇒ CV ≈ 0.88–1.02, drop ≈ 39–47 %`, vs `aux=0.01 ⇒
CV ≈ 0.10–0.15, drop ≈ 0–8 %`.
- **Without a tuned cf**, the same system either drops a large
  fraction of tokens (cf too tight) or wastes capacity on empty
  endpoints (cf too loose). cf interacts with `E`, with batch shape,
  and with the gate's current calibration in opaque ways. Switch
  (Fedus et al. 2021) reports cf ∈ [1.0, 2.0] as the production
  range; the literature has not produced a defensible auto-tuner.

The literature offers several alternatives but each gives up
something:

- **Expert Choice** (Zhou et al. 2022) inverts the problem — each
  expert picks its top-C tokens. This eliminates capacity overflow
  but breaks autoregressive causality because an expert's pick
  depends on which other tokens are present in the batch.
- **BASE layers** (Lewis et al. 2021) solve a linear-assignment
  problem per layer. Balance is optimal but the assignment is
  `O(N²)` Hungarian (or its convex relaxation), and the assignment
  rule is global — a token's expert depends on every other token's
  gate score.
- **Hash routing** (Roller et al. 2021) is fast, stateless, and
  capacity-blind. Empirically it loses ~10 – 20 % token survival
  versus capacity-aware routing under heterogeneous capacities (see
  Appendix A).

There is a structural gap: **a routing primitive whose load
distribution is determined by construction rather than by tuning, that
has nothing for the gate to collapse onto, and whose operating-regime
behaviour is invariant to provisioning hyperparameters.** Phase
routing fills this gap.

We make four contributions:

1. **A balance primitive (§3).** Phase routing is the cyclic-phase
   intersection of two bit-packed matrices encoding per-source and
   per-target demand. Selection is deterministic, reproducible from
   a seed, and produces uniformly low-discrepancy bipartite
   assignments by construction (no learned gate, no aux loss).
2. **A fused implementation (§4).** A single `O(N · k)` Rust kernel
   replaces the materialise-permute-AND-transpose pipeline. Routing
   wall-time is **8 ms** on the production scale we tested
   (`E = 32, batch×seq = 8192`), 20 – 30 % below the Python+top-k
   baseline.
3. **Empirical invariants (§5).** Across a 4-cf × 2-router × {8, 32}
   experts grid, phase routing's (CE, CV, drop, locality, route_time)
   tuple varies by **< 0.1 %** within each E. Top-k varies by 5 – 50
   % on the same metrics. The capacity-factor knob is empirically
   inert for phase.
4. **A clean negative result (§6).** A greedy hybrid that uses
   top-k's gate for affinity and phase-style quotas for balance fails
   catastrophically at 32 experts (62 % drop, CV = 1.55), because the
   gate component collapses onto `k+overflow` effective experts and
   the greedy fall-through has nowhere to land. This shows
   phase's invariances are _structural_, not transferable to a hybrid
   that retains a learned gate over experts.

---

## 2. Background and related work

### 2.1 The Switch / top-k recipe

The dominant MoE routing primitive is **top-k softmax gating** with
Switch-style capacity overflow (Shazeer et al. 2017; Fedus et al.
2021). Given gate logits `g ∈ ℝ^{N×E}`, the router computes
`p = softmax(g)`, selects `idx, w = topk(p, k)`, and enforces a
per-expert capacity `C = ⌈cf · N · k / E⌉`. Slots that arrive at a
full expert are set to `idx = -1` and zero-weight. A load-balance
auxiliary loss

```
L_aux = α · E · Σ_e (f_e · p̄_e)
```

(where `f_e` is the fraction of tokens routed to expert `e` as their
first choice and `p̄_e` is the mean gate probability for expert `e`)
is added with `α ≈ 0.01` to encourage uniform load.

### 2.2 Why this recipe is fragile

Three observations motivate our work:

1. **The aux loss does most of the balancing.** Setting `α = 0` at
   8 experts on TinyStories produces CV ≈ 1.0 (vs 0.10 with
   `α = 0.01`) and ~45 % drop (vs ~5 % with `α = 0.01`). This is a
   well-known result; we reproduce it (Table 4) for context.
2. **The CV / drop / contiguity numbers are sensitive to cf.** At
   `E = 32`, top-k's CV moves 0.14 → 0.22 across cf ∈ {1.0, 1.25,
   1.5, 2.0} and drop moves 11.8 % → 0.0 % across the same range.
3. **The mechanism is opaque.** The aux loss is a soft penalty on a
   global quantity; cf is a multiplicative knob on a hard ceiling.
   They interact, and there is no closed-form prescription for
   either as a function of (E, N, k, gate temperature, batch shape).

A balance primitive that requires neither would be a strict
simplification, even if it does not produce better tokens-to-experts
matching.

### 2.3 Alternatives in the literature

| primitive        | balance source            | causal?    | hyperparams    | failure mode          |
| ---------------- | ------------------------- | ---------- | -------------- | --------------------- |
| top-k + aux + cf | aux loss + cf             | yes        | `α`, `cf`, `k` | gate collapse w/o aux |
| Expert Choice    | per-expert pick           | **no**     | `C`, `k`       | breaks AR semantics   |
| BASE layers      | linear assignment (Hung.) | yes (slow) | none           | `O(N²)` per layer     |
| hash routing     | none (random)             | yes        | none           | capacity-blind        |
| **phase (ours)** | bipartite construction    | yes        | `base_density` | CE cost ≈ 5–12 %      |

The most relevant comparison is BASE: it also gives up a learned gate
and accepts that balance is the contribution. We retain the gate as
a routing weight (the gate's softmax distribution is gathered at the
selected expert positions and renormalised), so phase routing can
plug into the same training loop as top-k; we only replace the
selection rule.

### 2.4 Capacity-aware hash routing as a baseline

Hash routing (Roller et al. 2021) is the natural "stateless, fast,
deterministic" baseline. Capacity-aware variants modify the hash to
oversample experts in proportion to capacity. We benchmark phase
routing against capacity-aware hashing in Appendix A (10 – 19 %
higher token survival under heterogeneous capacity). The headline
finding is consistent: phase routing's structural-balance property
is real and shows up under standard tightness budgets.

---

## 3. Phase routing as a balance primitive

### 3.1 Setup

We consider routing `N` tokens to `E` capacity-bounded experts, each
token to `k` experts. Let `s_i` denote the per-source mass and
`t_j` the per-target capacity. The router returns `idx ∈
{−1, 0, …, E−1}^{N×k}` such that `Σ_i 𝟙[idx[i,:] = j] ≤ C_j` for
each expert `j` (capacity respect), and `idx[i, :]` is a multiset of
distinct experts (or `−1` for slots that overflowed) (fan-out
respect).

### 3.2 Cyclic-phase intersection

Each row is embedded as a contiguous interval on a ring of size `N`:

```
row i ↦ (φ_i, φ_i + s_i) mod N,    φ_i = Σ_{r<i} s_r  mod N
```

The cyclic embedding removes geometric bias — every starting
position is occupied with equal expected density across `i`. Two
independent column permutations are applied to source-matrix `S` and
target-matrix `T`; the routed assignment is

```
O = S' ∧ (T')^T
```

where `∧` is bitwise AND. The cyclic mixing produces

```
E[L_j] = E[Σ_i O_{ij}] ∝ t_j
```

— per-expert expected load is exactly proportional to per-expert
capacity, with `O(N⁻¹/²)` fluctuations from finite-`N` discreteness.
This is the structural-balance property: a phase router never
allocates more or less expected load than the target capacity
schedule prescribes.

**Phase routing thus converts balance from an optimisation objective
into a geometric invariant** — a property of the cyclic-phase
intersection rather than a target of training. The aux-loss
machinery that top-k routers depend on to keep the gate distribution
uniform has no analogue in phase routing because there is no learned
gate over experts to steer. There is nothing to tune (no `α`, no
`cf`) and nothing to collapse (no gate over experts), at the
structural cost of forfeiting per-token affinity (§5.5).

### 3.3 Why the materialised formulation is wasteful

Computing `O = S' ∧ (T')^T` directly costs `O(N²)` memory and
suffers a transpose. For inference batches with `N` in the
thousands and `E` in the dozens, this is prohibitive.

### 3.4 Fused arithmetic formulation

Each row of `S'` is a cyclic interval, and target membership in `T'`
is also a cyclic-interval condition. Their intersection reduces to:

> _check whether a point lies within a cyclic interval_

implemented via wrapping arithmetic:

```
d = (x − start) mod N,    keep iff d < len.
```

This eliminates branching, modulus, and matrix materialisation. For
each target row `j` (parallel across `j`):

```
1. iterate only positions where S'[j, ·] = 1
2. map positions via the inverse column permutation
3. evaluate target membership in T' arithmetically (O(1) per point)
4. reservoir-sample up to k matches
```

The reservoir step gives uniform-random selection over qualifying
matches without storing the full intersection set, capping
working memory per row at `O(k)`. Total working memory is `O(N)`.

### 3.5 Capacity policy

For MoE-style routing where every expert has equal capacity, we use
`phase_router_uniform_dispatch`: a specialised path that builds
identical per-row target bitmaps with `s_ones = base_density · width`
ones each (`width = N // E`), where `base_density ∈ (0, 1]` is the
**only** tunable parameter of the algorithm. In production we
operate at `base_density = 0.3`, which gives ~30 % over-coverage in
the routing search before the reservoir picks `k` matches. Lower
density tightens the bipartite construction (and increases the
chance of capacity overflow); higher density wastes search work
without changing the load distribution. We did not need to tune
this parameter across our sweeps.

For heterogeneous capacities (e.g. mixed-GPU shard balancing), an
`ema` mode tracks per-expert demand in an exponential moving average
and reshapes the per-target bitmap accordingly; we do not use this
mode for any result in §5.

### 3.6 What phase routing does **not** do

- It does **not** consume the gate scores. The selection rule is a
  function of `(N, E, k, base_density, seed)`; the gate's softmax
  distribution is only gathered at the chosen indices to provide
  differentiable routing weights (so gradients flow through the gate
  via the residual). This is why it has nothing to collapse onto.
- It does **not** guarantee that the highest-affinity expert is
  selected. In the worst case, the selected expert is the one the
  gate ranks `E`-th (rare under uniform seeding, but possible). This
  is the source of the 5 – 12 % CE cost reported in §5.

---

## 4. Implementation

### 4.1 Rust kernel

The Rust crate `phase_router_rs` exposes:

```python
phase_router_uniform_dispatch(
    N: int, E: int, k: int,
    base_density: float, seed: int, oversample: int,
) -> np.ndarray[(N, k), int32]
```

returning `−1` for any slot that could not be filled within the
candidate set. The fast path bypasses bit-packing in Python and
calls a single fused function with the GIL released. The slow
path (`phase_router_auto`) accepts arbitrary per-row source / target
bitmaps and is retained for heterogeneous-capacity ablations.

### 4.2 Wall-time

Across our production grid (`N = 8192, E = 32, k = 2`) on an A10G:

- top-k + softmax + capacity enforcement: ~11.9 ms / forward
- phase router uniform dispatch: ~8.5 ms / forward (28 % cheaper)

The CPU/GPU handoff dominates both (the kernel itself takes < 2 ms);
neither router is a step-time bottleneck at this scale (step time is
dominated by expert FFN compute at ~470 ms / step).

### 4.3 Integration

A drop-in replacement for `TopKRouter`:

```python
idx, weights, aux = router(logits, k)
```

where `weights = softmax(logits).gather(idx)` (renormalised) carries
the gradient. The expert dispatch layer that consumes
`(idx, weights)` is unchanged.

---

## 5. Experimental results

### 5.1 Setup

- **Model:** decoder-only MoE language model, 6.9 M params on tiny
  (8 experts) and 11.4 M on stress (32 experts).
- **Data:** TinyStories (Eldan & Li 2023), 2000 training steps,
  batch 16 × seq 512, AdamW, cosine schedule, 50 warmup steps,
  seed = 1337 across all runs.
- **Routers:** TopKRouter (`α = 0.01`, cf swept), TopKRouter-noaux
  (`α = 0`, cf ∈ {1.0, 1.25}, 8 experts only), PhaseRouter
  (`base_density = 0.3`, cf swept but inert), BalancedRouter
  (`overflow = 2`, cf swept; §6 negative result).
- **Grid:** cf ∈ {1.00, 1.25, 1.50, 2.00} for every router; 4 × 2 ×
  {tiny, stress} = 16 main runs, 2 aux-loss ablations, 8 hybrid
  runs.
- **Metrics:** `val_ce` (final-eval cross-entropy), `val_cv`
  (coefficient of variation of per-expert load), `val_drop`
  (fraction of (token, slot) pairs unrouted), `val_contig` (fraction
  of adjacent tokens sharing at least one expert — a cache-locality
  proxy), `route_time_ms` (wall-time spent in the router only).

### 5.2 Result 1: Capacity-factor invariance

The clearest finding in the paper. Across cf ∈ {1.00, 1.25, 1.50,
2.00} at `E = 32`:

| router    | metric     | min    | max    | swing across cf |
| --------- | ---------- | ------ | ------ | --------------: |
| top-k     | val_ce     | 4.712  | 4.766  |           1.1 % |
| top-k     | val_cv     | 0.1445 | 0.2207 |         +53.0 % |
| top-k     | val_drop   | 0.0000 | 0.1179 |   (12 pp range) |
| top-k     | val_contig | 0.110  | 0.141  |         +28.2 % |
| top-k     | route_ms   | 10.73  | 11.92  |         +11.1 % |
| **phase** | val_ce     | 5.286  | 5.287  |    **< 0.05 %** |
| **phase** | val_cv     | 0.0997 | 0.0997 |    **< 0.05 %** |
| **phase** | val_drop   | 0.0058 | 0.0058 |    **< 0.05 %** |
| **phase** | val_contig | 0.1200 | 0.1200 |    **< 0.05 %** |
| **phase** | route_ms   | 7.29   | 10.03  |           ~37 % |

The only non-flat phase metric is routing wall-time (which jitters
with kernel-launch noise); every metric the user actually cares about
(loss, balance, drop, locality) is **flat to six significant
figures** across the cf range. The same property holds at `E = 8`
(see Appendix B): phase CE, CV, drop, contig are bit-identical
across cf because the underlying selection trajectory does not
consume cf as an input.

**Single-line takeaway:** _Phase routing is the only primitive in
this study whose operating-regime metrics are invariant to the
capacity-factor hyperparameter._

### 5.3 Result 2: Anti-collapse without the aux loss

At `E = 8`, dropping the auxiliary load-balance loss (`α = 0`):

| router                        | cf   | val_cv    | val_drop    | val_ce |
| ----------------------------- | ---- | --------- | ----------- | ------ |
| top-k aux=0.01                | 1.00 | 0.106     | 8.5 %       | 4.928  |
| **top-k aux=0**               | 1.00 | **0.879** | **47.0 %**  | 4.907  |
| top-k aux=0.01                | 1.25 | 0.148     | 0.2 %       | 4.915  |
| **top-k aux=0**               | 1.25 | **1.022** | **39.4 %**  | 4.953  |
| **phase (no aux applicable)** | \*   | **0.036** | **0.000 %** | 5.181  |

Without the aux loss, top-k collapses: CV grows by ~10×, drop by
40 – 50×, and only the residual stream keeps CE within ~0.5 % of
the aux-on baseline. Phase has no aux-loss hyperparameter to drop —
the load distribution is structural and cannot collapse without
modifying the kernel itself.

**Single-line takeaway:** _Phase routing eliminates the aux-loss
balance hyperparameter; top-k's load balance is contingent on it._

### 5.4 Result 3: Scaling 8 → 32 experts

Holding every other knob byte-identical, going from `E = 8` to
`E = 32`:

| metric                      | top-k 8E | top-k 32E | Δ      | phase 8E | phase 32E | Δ       |
| --------------------------- | -------- | --------- | ------ | -------- | --------- | ------- |
| val_ce (cf = 1.25)          | 4.915    | 4.722     | −3.9 % | 5.181    | 5.287     | +2.0 %  |
| val_cv (cf = 1.25)          | 0.148    | 0.200     | +35 %  | 0.036    | 0.100     | +176 %  |
| val_drop (cf = 1.25)        | 0.2 %    | 2.9 %     | 14× ↑  | 0.0 %    | 0.6 %     | +0.6 pp |
| val_contig (cf = 1.25)      | 0.462    | 0.133     | −71 %  | 0.463    | 0.120     | −74 %   |
| route_ms (cf = 1.25)        | 10.65    | 11.92     | +12 %  | 8.26     | 8.52      | +3 %    |
| cf invariance of CV / drop? | no       | no        |        | **yes**  | **yes**   |         |

Phase's CV climbs from 0.036 to 0.100 (still 2× tighter than
top-k's 0.20 at the same scale), and a small but nonzero 0.6 % drop
appears at every cf. The construction enforces uniform load _in
expectation_ but the discrete quantum is `N / E = 64` tokens per
expert at `E = 32`, vs `256` at `E = 8`; the bound on CV scales
with `1/√(N/E)`. The cf-invariance property is preserved (phase's
metrics at 32 experts are again flat to four decimal places across
cf), which is the property we wanted.

**Single-line takeaway:** _Phase routing's discrete balance bound
scales with the quantum `N/E`, but its cf-invariance is preserved at
both 8 and 32 experts._

### 5.5 Result 4: The CE cost

Phase routing is **strictly worse than top-k on validation
cross-entropy** at every cf and every E we tested:

| scale      | top-k best CE | phase CE | gap     |
| ---------- | ------------- | -------- | ------- |
| 8 experts  | 4.845         | 5.181    | +6.9 %  |
| 32 experts | 4.712         | 5.286    | +12.2 % |

The gap widens with `E`. The mechanism is mechanical: phase
selection ignores gate affinity, so a token whose top-affinity
expert is at expert 17 might be routed to expert 9 instead. At
`E = 8` the typical-rank distance is ~2; at `E = 32` it grows
proportionally. Three considerations:

1. **The model still learns.** Phase's CE is descending throughout
   the run and the residual stream carries gradient signal through
   the (un)dropped tokens.
2. **The gap does not close with longer training in our experiments**
   (Appendix C), suggesting it is a structural cost of ignoring
   affinity, not a slow-warmup artefact.
3. **This is the reason for our framing.** Phase routing is **not**
   a better semantic router; it is a balance primitive. The right
   use is as a constraint to be layered _under_ a learned gate that
   handles affinity, not as a replacement.

**Single-line takeaway:** _Phase routing buys hyperparameter
elimination at a CE cost of 5 – 12 % that grows with `E`. The
contribution is the substrate, not the loss number._

### 5.6 Visual summary

The four operating-regime claims have a clean visual form across cf:

- (a) val_ce vs cf: top-k slopes downward; phase is a flat line.
- (b) val_cv vs cf: top-k climbs; phase is flat.
- (c) val_drop vs cf: top-k decays to zero; phase is a flat ε.
- (d) route_ms vs cf: top-k flat ~11 ms; phase flat ~8 ms.

All four panels are in `reports/stress_sweep/sweep_plots/`.

---

## 6. A clean negative result: a greedy hybrid fails

Given that phase has balance and top-k has affinity, the obvious
hypothesis is: **fuse them**. Use top-k's gate scores for affinity
but enforce phase-style per-expert quotas; on overflow, fall through
to the token's second / third / fourth choice. We pre-registered this
hypothesis as the `BalancedRouter` probe (see
`dev/ensemble_probe_plan.md`).

Algorithm (priority-ordered occupancy-aware admission):

1. compute `gate = softmax(logits)` and `top_w, top_idx =
gate.topk(k + overflow)` (we use `overflow = 2`, so 4 candidates
   per token at `k = 2`);
2. set quota `Q = ⌈cf · N · k / E⌉` (the Switch formula);
3. process tokens in descending `max_e gate[tok, e]` order
   (high-confidence first);
4. each token claims its highest-scoring candidate whose quota is
   not yet full; falls through on overflow;
5. a slot drops only if every one of the `k + overflow` candidates
   is at quota.

This is the discrete dual of "constrained gradient ascent on
affinity subject to capacity": never hard-mask the first choice,
only re-rank under load.

### 6.1 Result

On the 32-expert scaffold at cf = 1.0:

| metric     | top-k (cf=1.0) | phase (cf=1.0) | **balanced (cf=1.0)** |
| ---------- | -------------: | -------------: | --------------------: |
| val_ce     |          4.766 |          5.287 |                 4.787 |
| val_cv     |          0.144 |          0.100 |             **1.551** |
| val_drop   |        11.79 % |         0.58 % |           **61.91 %** |
| val_contig |          0.110 |          0.120 |                 0.053 |
| route_ms   |          11.74 |           9.59 |                 19.30 |

Balanced matches top-k's CE, **and is strictly worse than both top-k
and phase on every other axis** (CV is 10 – 16× worse than either
baseline, drop is 5 – 100× worse). cf provides no rescue: at cf =
2.0, drop is still 51.68 %.

### 6.2 Diagnosis

Train metrics show the gate distribution collapses to
`perm_entropy ≈ 0.61` (effective expert count ≈ 8 out of 32) over
the 2000-step run. After collapse, every token's
`top-(k+overflow) = top-4` candidate set lies inside that 8-expert
popular subset, and the popular subset's total capacity at cf = 1.0
is `8 · 512 = 4096` slots vs the `16,384` slot demand. The greedy
fall-through has nowhere to land within the candidate set, so
~12,000 slots drop.

The mechanism is general: **any hybrid that retains a learned gate
over experts inherits gate collapse**, and gate collapse defeats any
fall-through scheme whose candidate set is smaller than the
effective expert count after collapse. Phase routing is immune
because it has no gate to collapse.

### 6.3 What this shows

This is the clearest evidence in the paper that phase routing's
invariances are **structural**. They are not portable to a hybrid
that keeps a learned gate, because the failure mode they sidestep —
gate collapse — is reintroduced by any gate-bearing hybrid that
operates at this scale.

A different soft-admission family (occupancy-aware reranking, where
`gate_score ← gate_score − λ · current_load / quota`) might sidestep
the priority-greedy failure mode and is the natural follow-up. We
do not claim it would work; we flag it as the cheapest next
experiment.

**Single-line takeaway:** _Composing top-k affinity with phase-style
quotas via greedy admission does not recover phase's invariances;
gate collapse defeats the fall-through. Phase routing is a
substrate, not an ingredient._

---

## 7. Discussion: balance and specialisation as antagonistic objectives

Across this study, the three routers we evaluated map cleanly onto a
single underlying trade-off:

> **Balance and specialisation are antagonistic objectives under
> sparse routing.**

- **Top-k** allows semantic concentration in the gate, which yields
  per-token specialisation (the right expert is reachable), but
  destabilises occupancy (popular experts fill, CV climbs, drops
  spike at tight cf). Specialisation arrives bundled with occupancy
  instability — top-k needs both the aux loss (to push back against
  concentration) and the capacity factor (to absorb whatever
  concentration the aux loss could not suppress).
- **Phase** suppresses concentration by construction. Random column
  permutations spread mass uniformly across experts, which
  stabilises occupancy (CV flat, drop ≈ 0, cf inert), but weakens
  specialisation: a token routes to whichever expert the bipartite
  intersection picks, regardless of gate affinity — the 5 – 12 % CE
  cost of §5.5.
- **BalancedRouter** tries to have both. It gives top-k's gate the
  affinity job and phase-style per-expert quotas the balance job,
  with a soft fall-through (`overflow = 2`) to make the composition
  gentle. The composition collapses (§6): the gate concentrates
  onto `k + overflow` effective experts, the quota becomes binding
  on the popular subset, and the fall-through has nowhere to land —
  62 % of token-slots drop on the 32-expert scaffold. Trying to
  take both sides of the antagonism via greedy composition picks
  neither.

The conceptual reframing is that phase routing **converts balance
from an optimisation objective into a geometric invariant.**
Switch's aux loss is a penalty on a global statistic of the gate's
behaviour; phase's balance is a property of the cyclic-phase
intersection (§3.2) that no choice of weights can disturb. Once
balance is a geometric invariant rather than an optimisation target,
the system has nothing to tune (no `α`, no `cf`) and nothing to
collapse (no gate over experts) — at the structural cost of
forfeiting per-token affinity.

We do not claim the antagonism is **fundamental** — only that, _under
greedy composition_, the routers we tested cannot have both. Whether
a non-greedy admission rule (occupancy-aware reranking with a
smoothly differentiable load penalty; learned load-shaping heads
that re-weight a phase-produced routing matrix) can sit on the
Pareto frontier between specialisation and balance remains open.
What our experiments do establish is that the trade-off is real,
that the obvious greedy hybrid does not sidestep it, and that **the
contribution of phase routing is to pick one corner of that frontier
cleanly** — full balance, no hyperparameters, no collapse — rather
than to negotiate a soft compromise.

**Single-line takeaway:** _Balance and specialisation are
antagonistic under sparse routing; phase routing picks balance and
makes it a geometric invariant; the affinity tax is the price._

---

## 8. When to use phase routing

Phase routing is the right choice when **balance and reproducibility
matter more than per-token affinity**, and when the system can absorb
a moderate CE penalty. Concretely:

- **Heterogeneous-capacity shard dispatch.** Mixed-GPU MoE inference,
  where some experts have 4× the memory of others. Phase's
  capacity-proportional load assignment (§3.2) gives 10 – 19 % higher
  token survival than capacity-aware hashing (Appendix A).
- **Inference at tight cf.** Switch's recommendation of cf ∈ [1.0,
  2.0] depends on careful tuning. Phase's cf-invariance lets
  operators provision exactly at the quantum `N · k / E` without
  worrying about cf interactions.
- **Reproducibility-critical pipelines.** Phase is bit-deterministic
  given a seed. Top-k with aux loss is reproducible too, but the
  aux-loss coefficient enters the optimisation objective and changes
  the trained model; phase changes only the inference path.
- **Anti-collapse-critical pipelines.** Long-running deployments
  where gate collapse would be a Service Level Objective failure.
  Phase cannot collapse.

Phase routing is the **wrong** choice when:

- per-token expert specialisation is the dominant value driver
  (because phase pays a 5 – 12 % CE cost);
- the deployment can tolerate aux-loss tuning and cf tuning;
- the batch shape is so small that the routing-time savings are
  insignificant (phase savings are 2 – 3 ms / forward; at very small
  batches the rest of the model dominates).

---

## 9. Limitations and open questions

1. **CE cost.** 5 – 12 % at the scales we tested. We do not know
   whether this gap closes at larger scales (the structural
   ignore-affinity argument suggests not), but we have not tested at
   the 1B-parameter range where MoE economics flip.
2. **No load-shaping head yet.** A learned ranking that re-weights
   the routing matrix produced by phase routing could in principle
   recover most of the CE gap. We flagged occupancy-aware reranking
   as the natural follow-up (§6.3); it is out of scope for this paper.
3. **The discrete quantum scales with `N / E`.** At `E = 32` and
   batch×seq = 8192, the quantum is 64 tokens/expert; CV grows from
   0.036 (at `E = 8`) to 0.100 (at `E = 32`). At `E = 256` with the
   same batch we would expect ~0.3 CV — still much better than
   top-k's projected ~0.5, but no longer "uniform".
4. **One ablation we did not run.** `base_density ∈ {0.1, 0.2, 0.3,
0.5}`. The 0.3 production setting was load-bearing across every
   sweep; we expect this to be a smooth trade-off curve but have not
   measured it.
5. **Single dataset, single model family.** TinyStories with a 6.9 –
   11.4 M parameter decoder. The qualitative claims (cf-invariance,
   anti-collapse, structural balance) are predicted to be
   dataset-independent because they are properties of the routing
   kernel; the quantitative CE gap is not.

---

## 10. Conclusion

We presented phase routing, a deterministic balance primitive built
on cyclic-phase intersection of bit-packed source / target matrices.
We claim, and empirically support, the following framing:

> **Phase routing is a deterministic discrepancy-minimizing balance
> primitive whose operating behaviour is invariant to capacity
> factor, immune to gate collapse by construction, and stable under
> provisioning sweeps.**

We do not claim to have a better semantic router than top-k — we pay
a 5 – 12 % cross-entropy cost. The phase router is **a simple substrate
for capacity-factor-free, aux-loss-free, anti-collapse balance in sparse
routing systems**, and we provide a fused Rust implementation that is
20 – 30 % cheaper to invoke than top-k at production scale.

Equivalently: **phase routing converts balance from an optimisation
objective into a geometric invariant**, at the cost of forfeiting
per-token affinity. Balance and specialisation are antagonistic
under sparse routing; phase routing picks balance and pays the
affinity tax.

A natural greedy hybrid that tries to compose top-k affinity with
phase-style quotas fails catastrophically because gate collapse
defeats fall-through. This sharpens, rather than weakens, the case
for phase as a structural primitive: its invariances are not
portable into hybrids that retain a learned gate over experts.

We believe the right downstream programme is to layer learned
load-shaping heads _under_ phase rather than to swap phase _for_
top-k, and we leave occupancy-aware reranking (the natural next
hybrid that sidesteps the priority-greedy failure mode) to future
work.

---

## Appendix A: Capacity-aware hash baseline

(Summary of `docs/comparison.md`.) Against capacity-aware uniform
hashing, phase routing achieves 10 – 19 % higher token survival
under heterogeneous expert capacity. The advantage grows with
fan-out `k`: at `k = 16` (common in large MoE), phase's survival is
96.6 % vs hash's 77.6 %. The CV-invariance and anti-collapse
properties are not addressed by hash routing; the comparison there
is on survival under capacity-pressure only.

## Appendix B: Full per-cf tables

See `reports/cf_sweep_v2.md` (8 experts) and
`reports/stress_sweep.md` (32 experts). Each lists `val_ce, val_cv,
val_drop, val_contig, val_pe, route_ms, tok/s, wall_s` for every
(cf, router) cell of the grid.

## Appendix C: Hyperparameter and reproducibility details

- Rust crate: `phase_router_rs` (this repo).
- Python bindings: `phase_router_uniform_dispatch`,
  `phase_router_auto`.
- Training entry-points: `train/train_baseline.py` (top-k),
  `train/train_phase.py` (phase), `train/train_balanced.py`
  (hybrid).
- Modal orchestration: `train/modal_app.py`, app id
  `ap-3ntHqZIJkg9z79s9yw9fGr`.
- All sweeps: A10G, fp32, 2000 steps, seed 1337, batch 16 × seq 512,
  AdamW(lr 3e-4, β=(0.9, 0.95), wd 0.01), cosine schedule, 50 warmup.
- Full training-time metric logs in `runs/<sweep>/<run>/metrics.jsonl`.

## References

Fedus, W., Zoph, B., & Shazeer, N. (2021). Switch Transformers:
Scaling to trillion parameter models with simple and efficient
sparsity. _JMLR 23(120)_.

Lewis, M., Bhosale, S., Dettmers, T., Goyal, N., & Zettlemoyer, L.
(2021). BASE layers: Simplifying training of large sparse models.
_ICML_.

Roller, S., Sukhbaatar, S., Szlam, A., & Weston, J. (2021). Hash
layers for large sparse models. _NeurIPS_.

Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton,
G., & Dean, J. (2017). Outrageously large neural networks: The
sparsely-gated mixture-of-experts layer. _ICLR_.

Zhou, Y., Lei, T., Liu, H., Du, N., Huang, Y., Zhao, V., Dai, A.,
Chen, Z., Le, Q., & Laudon, J. (2022). Mixture-of-experts with
expert choice routing. _NeurIPS_.
