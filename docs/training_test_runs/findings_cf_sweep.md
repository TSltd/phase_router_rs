# Capacity-factor sweep — findings

`reports/cf_sweep.md` aggregates 10 Modal A10G runs of `tiny.yaml`
(2000 steps each, ~$1.50 total):

- 4 cf × 2 routers main grid = 8 runs
- 2 top-k aux-loss ablations (cf ∈ {1.0, 1.25}, `aux_loss_alpha=0`)

This document records the three concrete findings and what they mean
for framing the paper.

---

## 1. Phase router is _invariant_ to capacity factor

| cf   | phase val_ce | phase val_cv | phase val_drop |
| ---- | ------------ | ------------ | -------------- |
| 1.00 | 5.181        | 0.0360       | 0.0000         |
| 1.25 | 5.181        | 0.0360       | 0.0000         |
| 1.50 | 5.181        | 0.0360       | 0.0000         |
| 2.00 | 5.181        | 0.0360       | 0.0000         |

The phase router's capacity comes from `base_density × width` (constant
across runs), not from the `capacity_factor` arg, so all four runs are
literally bit-for-bit identical training trajectories. This is **a
feature, not a bug** — phase routing has no `cf` knob to tune, by
design. The whole concept of "capacity factor" doesn't apply.

This is publishable as: _"phase routing requires zero capacity tuning;
the per-expert load distribution is determined structurally"._

## 2. Top-k's aux-loss is doing 95% of the balancing work

| cf   | aux=0.01 val_cv | aux=0 val_cv | aux=0.01 val_drop | aux=0 val_drop |
| ---- | --------------- | ------------ | ----------------- | -------------- |
| 1.00 | 0.106           | **0.879**    | 0.085             | **0.470**      |
| 1.25 | 0.148           | **1.022**    | 0.002             | **0.394**      |

Without the auxiliary load-balance loss, top-k:

- CV jumps from ~0.10 to **~0.9-1.0** (close to maximally imbalanced
  for 8 experts, where CV ≈ √(N-1) = 2.6 if one expert eats everything).
- Drop fraction jumps to ~40-47%: nearly half the (token, slot) pairs
  are silently zeroed out by capacity overflow.

But CE is **barely affected** (4.928 → 4.907 at cf=1.0 — _better_
without aux loss!). This is the most surprising result: at this scale,
the model trains fine even when half its tokens are routing-dropped,
because (a) the residual stream still passes signal through MHSA, and
(b) tokens that _do_ reach an expert get processed.

This means phase's "free balance" advantage is real but **doesn't
translate into a CE advantage at this scale**.

## 3. The actual head-to-head story is "phase = deterministic balancer"

Per capacity factor, at this 2000-step `tiny` scale on A10G:

| cf   | top-k CE | phase CE | top-k CV | phase CV | top-k drop | phase drop | top-k tok/s | phase tok/s |
| ---- | -------- | -------- | -------- | -------- | ---------- | ---------- | ----------- | ----------- |
| 1.00 | 4.928    | 5.181    | 0.106    | 0.036    | 8.5%       | 0%         | 17,917      | 13,882      |
| 1.25 | 4.915    | 5.181    | 0.148    | 0.036    | 0.2%       | 0%         | 17,601      | 13,971      |
| 1.50 | 4.906    | 5.181    | 0.156    | 0.036    | 0.0%       | 0%         | 17,904      | 14,040      |
| 2.00 | 4.845    | 5.181    | 0.146    | 0.036    | 0.0%       | 0%         | 17,718      | 14,035      |

- **Phase always wins on CV (3× tighter) and drop rate (always 0%).**
- **Top-k always wins on CE (5%-7% lower) and throughput (~25% faster).**
- The CE gap _grows_ with cf because top-k uses the extra capacity
  to indulge gate preference, while phase ignores it.

`contig_frac` (locality proxy) is essentially the same for both
(~0.46), confirming that the original "phase has better routing
locality" hypothesis does not hold at this scale.

## 4. Where the throughput gap actually comes from

`route_time_ms` per forward (sum across 4 MoE blocks):

- top-k: **10.5 ms** (capacity enforcement is the per-slot Python loop)
- phase: **34.5 ms** (bit-packing in `_build_t_bits` + `_select`'s
  per-row uniqueness loop, both in Python)
- aux=0 top-k: 9.6 ms (no aux-loss bookkeeping)

Of phase's 34.5 ms, at most ~12 ms is the actual Rust kernel call
(estimated from local CPU runs). The remaining ~22 ms is **Python-side
glue** that could go straight to Rust:

- `_build_t_bits` / `_build_s_bits_uniform` (~10 ms)
- per-row uniqueness loop converting kernel columns → unique experts (~12 ms)

Moving those into the crate would close most of the throughput gap.
Locally on CPU we measured ~14 ms/forward for phase, suggesting most
of the 34 ms on A10G is CPU→GPU transfer + Python overhead, not the
kernel itself.

---

## What we have for the paper

A defensible single-config result:

> Phase routing produces **structurally balanced, drop-free expert
> assignments at every capacity factor, with no auxiliary loss and no
> capacity-factor tuning**, at a 5% cross-entropy cost and a 22%
> throughput cost (most of the latter in unoptimised Python glue).
> Top-k's load balance is _almost entirely_ provided by its auxiliary
> loss term (CV explodes from 0.1 to 1.0 without it), exposing how
> much of "Switch-style routing" is actually "Switch-style aux-loss".

That's a real, narrow contribution that doesn't oversell. It needs:

1. **Engineering polish** — move bit-packing into Rust to close the
   ~22ms Python overhead and report apples-to-apples kernel cost.
   ~1 day work.
2. **A "stress regime" run** where balance is known to matter — i.e.
   higher `n_experts` and tighter `cf`. At `tiny` we have 8 experts;
   moving to 32 or 64 experts on `small.yaml` would push the regime
   where top-k's aux loss starts to _fail_ and the CE gap may close
   or reverse.
3. **Throughput parity probe** — once Rust glue is in, redo cf=1.0
   to see if phase's ~14 ms route_time still constitutes a meaningful
   step-time fraction.

---

## Recommended next step

**Move `_build_t_bits` / `_build_s_bits_uniform` and the per-row
uniqueness pick into Rust** as a single function
`phase_router_uniform_dispatch(n, n_experts, k, base_density, seed) →
(N, k) int32 unique-per-row`. Estimated effort: 200-300 LOC of Rust,
~1 day. Local CPU bench expected to drop phase route_time from ~14 ms
to ~3-5 ms; on A10G it should similarly drop the 22ms Python overhead.

After that, re-run the same sweep and we'll have a single defensible
comparison + a 32-expert run on `small.yaml` to test the stress
hypothesis. Total cost: ~$5 of Modal credit.

If the throughput gap closes and CE gap stays the same (~5%), the
paper claim crystallises as **"phase routing = deterministic balance
without aux-loss, at parity throughput and small CE cost"**.

If 32 experts widens the CE gap further, the paper becomes **"phase
routing is a balance primitive, not a CE replacement; combine with
top-k for best of both worlds"** — which is also publishable.
