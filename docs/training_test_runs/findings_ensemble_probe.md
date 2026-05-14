# Ensemble probe — findings (BalancedRouter on tiny + stress)

**Date:** 2026-05-14
**Sweeps:** `runs/ensemble_stress_sweep/`, `runs/ensemble_tiny_sweep/` (4 cf each, ~$3 of credit)
**Reports:** `reports/ensemble_stress_sweep.md`, `reports/ensemble_tiny_sweep.md`
**Plan:** `dev/ensemble_probe_plan.md`
**Comparison baselines:** `runs/cf_sweep_v2/` (8 experts) and `runs/stress_sweep/` (32 experts) on byte-identical scaffolds.

## TL;DR

**The pre-registered "wash" outcome landed — except worse.** The
`BalancedRouter` (priority-ordered occupancy-aware admission with
overflow = 2) matches top-k's CE on the 32-expert scaffold, but it
is **strictly worse than both top-k and phase** on every other axis
(load CV, drop rate, locality, routing wall-time). The plan's
hypothesis grid had explicit guidance for this outcome:

> _Wash — numbers somewhere between phase and top-k on every
> dimension. Greedy priority-driven assignment isn't expressive enough
> — need Sinkhorn or learned admission. Drop balanced; pitch paper on
> the cf-elimination claim from stress_sweep alone._

That is what we are doing. The contribution of this sweep is a clean
negative result with a clean diagnosis (router collapse → quota
binding → no fall-through room), and it sharpens the paper's framing
around phase as a structural balance primitive rather than as one
ingredient of a hybrid.

Headline numbers, all `val_final` row:

| dimension              | top-k (cf=1.0) | phase (cf=1.0) | **balanced (cf=1.0)** | balanced direction |
| ---------------------- | -------------: | -------------: | --------------------: | ------------------ |
| **Stress, 32 experts** |                |                |                       |                    |
| val_ce ↓               |          4.766 |          5.287 |             **4.787** | ≈ top-k ✓          |
| val_cv ↓               |          0.144 |          0.100 |             **1.551** | 10× **worse**      |
| val_drop ↓             |        11.79 % |         0.58 % |           **61.91 %** | 5× **worse**       |
| val_contig ↑           |          0.110 |          0.120 |             **0.053** | 2× **worse**       |
| val_rt_ms ↓            |          11.74 |           9.59 |              **19.3** | 2× **worse**       |
| **Tiny, 8 experts**    |                |                |                       |                    |
| val_ce ↓               |          4.928 |          5.181 |             **4.889** | beats top-k ✓      |
| val_cv ↓               |         0.1057 |         0.0360 |             **0.580** | 5–16× **worse**    |
| val_drop ↓             |         8.54 % |         0.00 % |           **27.91 %** | 3× **worse**       |
| val_contig ↑           |          0.404 |          0.463 |             **0.291** | 1.4× **worse**     |
| val_rt_ms ↓            |          10.44 |           7.87 |              **15.5** | 1.5–2× **worse**   |

The cf knob does help — at cf = 2.0 on tiny, drop falls to 0.2 % and
CE is 4.853 (essentially top-k). But at cf = 2.0 on stress, drop is
still **51.68 %** (down only ~10 pp from cf = 1.0), and CV is
**1.803** (worse than at cf = 1.0). The mechanism that makes balanced
fail at cf = 1.0 also makes it fail at cf = 2.0 on stress.

## Headline plot

(Verbal summary; PNGs in `reports/ensemble_stress_sweep/sweep_plots/`
and `reports/ensemble_tiny_sweep/sweep_plots/`.)

Drop curves (stress, val_final):

- top-k: 11.79 % → 2.92 % → 0.39 % → 0.00 % (across cf 1.0–2.0)
- phase: 0.58 % flat
- **balanced: 61.91 % → 61.88 % → 57.16 % → 51.68 %** — barely
  responsive to cf.

CV curves (stress, val_final):

- top-k: 0.14 → 0.20 → 0.21 → 0.22 (rising with cf)
- phase: 0.10 flat
- **balanced: 1.55 → 1.90 → 1.80 → 1.80** — 10× worse than either,
  flat-ish across cf.

The cf-elimination property that phase enjoys (CE / CV / drop /
contig all change by < 0.1 % across cf) is **not** inherited by
balanced. Balanced has top-k-shaped cf sensitivity for CE and drop,
but with offsets that put it strictly worse than both baselines on
balance.

## What changed during training (router collapse)

Train metrics tell a clean story of router collapse. Comparing
step = 1 to step = 2000:

| run                      | step 1            | step 2000              |
| ------------------------ | ----------------- | ---------------------- |
| stress balanced cf = 1.0 | cv 0.08, drop 4 % | cv 1.54, **drop 62 %** |
| stress balanced cf = 2.0 | cv 0.25, drop 0 % | cv 1.80, **drop 52 %** |
| tiny balanced cf = 1.0   | cv 0.05, drop 2 % | cv 0.58, **drop 28 %** |
| tiny balanced cf = 2.0   | cv 0.15, drop 0 % | cv 0.75, drop 0.2 %    |
| stress balanced **pe ↓** | 0.99 → 0.65       | 0.99 → 0.61            |
| tiny balanced **pe ↓**   | 0.999 → 0.835     | 0.994 → 0.788          |

Across all four runs, `perm_entropy` (normalised to log(E)) collapses
from ~1.0 at init to 0.6–0.8 at convergence. With E = 32, that
corresponds to an effective expert count of `exp(pe · log E) ≈ 8.3`
— i.e. the gate distribution concentrates onto roughly 8 popular
experts out of 32. Top-k can absorb this collapse because it has no
quota; phase can absorb it because its bipartite construction
enforces uniform load by ignoring the gate. **Balanced cannot
absorb it** because:

1. The model learns to concentrate gate mass on a small subset of
   experts.
2. For most tokens, the entire top-(k + overflow) = top-4 candidate
   set is inside that subset.
3. The per-expert quota fills within a few hundred tokens.
4. The remaining tokens find _every_ candidate in their top-4 at
   quota → both slots drop.

Hence drops climb monotonically with training rather than annealing
out, and CV climbs in lock-step.

## Diagnosis — why greedy priority-driven assignment fails

The priority-greedy assignment with overflow = 2 has an implicit
assumption: that even when the most-popular experts are at quota, a
typical token's top 3rd or 4th choice will be a _less_ popular expert
that still has room. This assumption holds when the gate
distribution is nearly uniform; it fails catastrophically when the
gate distribution concentrates onto k+overflow ≈ effective-E
experts.

At stress (E = 32, k = 2, batch×seq = 8192):

- Quota at cf = 1.0: `ceil(1.0 · 8192 · 2 / 32) = 512` tokens/expert.
- Total capacity at cf = 1.0: `512 · 32 = 16,384 = N·k` (exactly
  tight).
- Effective experts after collapse: ~8 (from perm_entropy 0.65).
- Popular subset capacity at cf = 1.0: `8 · 512 = 4,096` token-slots,
  vs effective demand on those 8 experts ≈ `16,384` token-slots.
- ⇒ Approximately `12,288 / 16,384 = 75 %` of the demand cannot fit
  inside the popular subset. The top-(k + overflow) = top-4 list
  for those tokens almost certainly lives entirely inside the popular
  subset, so the spillover is drops, not fall-through. Observed
  drop: 62 %. (The shortfall vs the back-of-envelope 75 % is closed
  by the few low-confidence tokens that genuinely _do_ rank an
  unpopular expert in their top 4.)

At cf = 2.0 the popular subset still saturates at `8 · 1024 = 8192`
token-slots vs 16384 demanded → 50 % spillover → observed drop
51.68 %. The cf knob just isn't expressive enough to absorb a
gate-distribution that collapses onto k+overflow effective experts.

This is exactly the failure mode the **plan's "wash" hypothesis row**
called out:

> _Greedy priority-driven assignment isn't expressive enough — need
> Sinkhorn or learned admission._

## Why CE is misleadingly good

A reasonable reaction is: "62 % of token-slot pairs dropped, but CE
matches top-k? Something's wrong." It isn't — the explanation is
that:

1. **The surviving 38 % of token-slot pairs are the highest-confidence
   ones** (priority ordering picks them). They contribute most of
   the routing-quality signal that CE rewards.
2. **Dropped tokens still propagate via the residual stream** (the
   MoE block is an additive sub-layer; if both slots drop, the
   residual passes through unchanged). The model loses the expert
   FFN's compute for those tokens but doesn't lose their gradient
   signal entirely.
3. **CE is a token-averaged loss, not a token-conditional one** — at
   62 % drops the per-token sample is heavily weighted toward
   high-confidence tokens, which is exactly where any router does
   well.

So balanced's CE on stress (4.78–4.72) is real but reflects the
biased subsample of confidently-routed tokens, not the router's
overall quality. The honest router-quality metric is `CE × (1 −
drop)` adjusted for the residual path, but we didn't pre-register
this and it conflates a few effects we can't separate without a
controlled ablation. The takeaway is: **CE alone hides router
pathology when drop rates are this high**.

This is itself a useful methodological observation for the paper:
single-metric router comparisons are misleading once the constraint
becomes binding. The `cf_sweep_v2` and `stress_sweep` tables that
report (CE, CV, drop, contig, rt_ms) together are the right reporting
contract.

## Tiny-vs-stress diff

The same algorithm has very different failure pressure at E = 8 vs
E = 32:

| metric                   | tiny (E = 8)           | stress (E = 32)         |
| ------------------------ | ---------------------- | ----------------------- |
| balanced drop @ cf = 1.0 | 28 %                   | 62 %                    |
| balanced drop @ cf = 2.0 | 0.2 %                  | 52 %                    |
| cf knob effective?       | yes (28 % → 0.2 %)     | **no** (62 % → 52 %)    |
| balanced cv @ cf = 1.0   | 0.58                   | 1.55                    |
| balanced cv @ cf = 2.0   | 0.75                   | 1.80                    |
| ce vs top-k @ cf = 1.0   | balanced wins by 0.039 | balanced loses by 0.021 |
| effective experts (pe)   | ~5.6 (out of 8)        | ~8.3 (out of 32)        |

At E = 8, the (k + overflow) = 4 candidate list per token covers half
the experts, so even after collapse there is usually somewhere to
fall through to. cf = 2.0 is enough to drive drops to ~0 because the
quota becomes non-binding. At E = 32, the candidate list covers 1/8
of the experts and overlaps almost entirely with the popular subset
after collapse — there is nowhere to fall through to within
the candidate set, so drops stay catastrophic _regardless of cf_.

This is the standard **"overflow doesn't scale with E"** failure
mode. Fixing it requires either (a) overflow ∝ effective-E (which
balloons to a Sinkhorn-style soft assignment over all experts), or
(b) admission penalty embedded in the gate score rather than as a
hard quota (the "occupancy-aware reranking" family from the plan).

## Routing wall-time and throughput

- Balanced route_time_ms ≈ 15–19 ms; top-k ≈ 10–12 ms; phase ≈ 8–10
  ms. The python priority-greedy loop adds 5–10 ms over the fused
  top-k path. On stress this is a ~50 % overhead on routing alone.
- Balanced tok/s ≈ 8,700 on stress vs top-k's 4,500. Balanced
  appears 2× faster — but this is misleading: balanced is dropping
  62 % of token-slots, so the expert FFN does ~38 % of the work that
  top-k does. Adjusted for retained work, balanced's effective
  throughput is `8,700 · 0.38 ≈ 3,300 tok/s`, _below_ top-k. The
  raw tok/s metric is just measuring how many tokens flow past the
  MoE block, not how many are actually getting expert compute.

So balanced does not even win on the throughput axis once we adjust
for dropped work.

## Pre-registered hypotheses — which arm landed?

The plan listed four pre-registered outcomes. Mapping observations
onto them:

| outcome         | criterion                                                       | landed?                                                                                                           |
| --------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Sweet spot      | CE within 1 % of top-k AND CV ≤ 0.10 AND drop ≤ 1 % at every cf | **no** — CE met on stress, but CV (10× phase) and drop (60×) catastrophically miss.                               |
| CE win, CV loss | CE ≤ top-k, CV between phase and top-k                          | **partially** — CE matched on stress; but CV is _worse than_ top-k, not between.                                  |
| CV win, CE loss | CV ≈ phase, CE between phase and top-k                          | **no** — CV is much worse than top-k, not better.                                                                 |
| Wash            | numbers somewhere between phase and top-k on every dimension    | **closest** — but balanced is _outside_ the [phase, top-k] interval on CV and drop, so this is "wash, but worse". |

The success criteria explicitly required 3 of 5 thresholds:

| threshold          | target | observed (stress cf = 1.0) | met? |
| ------------------ | ------ | -------------------------- | ---- |
| val_ce ≤ 4.80      | yes    | 4.787                      | ✓    |
| val_cv ≤ 0.105     | yes    | 1.551                      | ✗    |
| val_drop ≤ 0.010   | yes    | 0.619                      | ✗    |
| val_contig ≥ 0.110 | yes    | 0.053                      | ✗    |
| val_rt_ms ≤ 12     | yes    | 19.30                      | ✗    |

1 of 5 met. The paper does **not** rewrite around balanced. We fall
back to the cf-elimination story already validated in
`dev/findings_stress_sweep.md`.

## Implications for the paper

1. **The hybrid framing is dead.** The plan's "phase as constraint
   atop a learned head" hypothesis isn't operationalised by a
   greedy priority-driven assignment. The paper's contribution
   should not include a hybrid router as a primary claim.

2. **The cf-elimination property is sharpened.** The stress sweep
   already showed that phase's metrics are flat in cf; this sweep
   shows that injecting top-k-style affinity scoring into a
   capacity-constrained admission process _re-introduces_ cf
   sensitivity, and the wrong shape of cf sensitivity (drop responds
   to cf, but CV and contig do not). Phase's cf-invariance is a
   _structural_ property of the bipartite construction, not just an
   empirical coincidence.

3. **The single-metric trap.** This sweep makes a clean
   methodological point that the paper should make explicitly:
   reporting CE alone is enough to mislead an unwary reader into
   believing balanced "works" on the 32-expert scaffold. The
   right comparison is the `(CE, CV, drop, contig, rt_ms)` tuple —
   which we have been reporting since `cf_sweep_v2` and which
   _immediately_ shows balanced's pathology.

4. **What about future hybrids?** A different soft-admission family
   might still work — the plan explicitly listed three:
   - (a) **quotas + greedy fall-through** (this sweep — failed),
   - (b) **occupancy-aware reranking** (penalty `−λ · load/quota` in
     the gate score, then standard top-k — not tested),
   - (c) **Sinkhorn-style soft assignment** (not tested).

   Family (b) is the cheapest follow-up because the load can be a
   running EMA buffered across the batch, no global sort or priority
   loop required. But this is now follow-up work, not in-scope for
   the current paper.

## Recommended next step

**Stop iterating on hybrids; finalise the cf-elimination paper.**
The result here, combined with `dev/findings_stress_sweep.md`, gives
a strong negative-result-supports-the-positive-result pairing:

- _Positive:_ Phase routing is the only primitive in this experiment
  whose behaviour is independent of the capacity-factor
  hyperparameter, at a CE cost that scales with `n_experts`.
- _Negative:_ A natural hybrid that tries to inherit both top-k's CE
  and phase's balance via greedy capacity-constrained admission fails
  catastrophically once the gate distribution collapses onto
  `k + overflow` effective experts, because the soft fall-through has
  nowhere to fall through to.

The paper now has one clean contribution (cf-elimination) and one
honest negative result (greedy admission doesn't compose). That is
a tighter pitch than "we have a hybrid that works", and it survives
adversarial review better.

If anyone wants to revisit the hybrid story later, the natural next
attempt is **occupancy-aware reranking** — `gate_score ← gate_score
− λ · current_load / quota`, then standard top-k — because it
sidesteps the priority-greedy failure mode entirely: a token's
gate-rank reorders smoothly as loads accumulate, rather than
discretely flipping from "popular and full → not admitted" to "less
popular → admitted". Cheap to implement, no priority sort needed,
and the gradient signal still flows through the gate. But it is a
follow-up, not a paper-blocker.

## Reproducibility

- Configs: `train/configs/stress.yaml` (32 experts) and
  `train/configs/tiny.yaml` (8 experts) — unchanged from prior
  sweeps.
- Sweep entrypoints:
  ```
  modal run --detach train/modal_app.py::ensemble_sweep --config stress
  modal run --detach train/modal_app.py::ensemble_sweep --config tiny --sweep-name ensemble_tiny_sweep
  ```
- Router: `train/routers.py::BalancedRouter` (capacity_factor swept;
  overflow = 2; aux_loss_alpha = 0.0).
- Aggregator: `python3 train/compare_sweep.py runs/ensemble_stress_sweep -o reports/ensemble_stress_sweep.md`
  (regex in `compare_sweep.py::SUBDIR_RE` already recognises
  `balanced_cf*`; style entry in `SERIES_STYLE` ditto).
- 2000 steps × A10G × fp32, ~$3 of credit total.
- Modal sweep runs: `ensemble_stress_sweep` (4 runs, ~$1.50) and
  `ensemble_tiny_sweep` (4 runs, ~$1.50).
