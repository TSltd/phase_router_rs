# Capacity-factor sweep v2 — findings (post fused dispatch)

**Date:** 2026-05-14
**Sweep:** `runs/cf_sweep_v2/` (10 A10G runs, ~$1.50)
**Report:** `reports/cf_sweep_v2.md`
**Prior sweep:** `runs/cf_sweep/` (v1, see `dev/findings_cf_sweep.md`)
**Code change:** `phase_router_uniform_dispatch` fused into Rust
(see `dev/fused_dispatch_notes.md`)

This sweep re-runs the v1 grid against an identical training pipeline,
with **only** `PhaseRouter._select` swapped from
NumPy-glue-around-`phase_router_auto` to a single fused Rust call
(`phase_router_uniform_dispatch`). All other code paths, configs,
seeds, and hyperparameters are byte-identical.

## TL;DR

| dimension                      | v1 (Python glue) | v2 (fused Rust) |         Δ |
| ------------------------------ | ---------------: | --------------: | --------: |
| phase `route_time_ms` (cf=1.0) |         ~34.5 ms |      **7.9 ms** | **−77 %** |
| phase `tok/s` (cf=1.0)         |           13,882 |      **17,814** | **+28 %** |
| top-k `route_time_ms` (cf=1.0) |          10.5 ms |         10.4 ms |       ±0% |
| top-k `tok/s` (cf=1.0)         |           17,917 |          17,705 |       ±0% |
| phase `val_ce` (cf=1.0)        |            5.181 |           5.181 |    bit-eq |
| phase `val_cv` (cf=1.0)        |           0.0360 |          0.0360 |    bit-eq |
| phase `val_drop` (cf=1.0)      |           0.0000 |          0.0000 |    bit-eq |

The change was a **pure plumbing optimisation**: phase routing's val
metrics are bit-identical across v1 and v2 (same seed, same kernel,
same uniqueness selection — just moved across the FFI boundary).

## What changed

1. **Phase route_time collapsed from 34.5 ms → ~8 ms** at every cf,
   slightly _beating_ top-k's ~10.5 ms.
2. **Throughput parity** between routers: phase 17.4–17.8k vs top-k
   17.3–17.7k tok/s. Phase actually wins throughput at 3 of 4 cf
   points (1.00, 1.25, 2.00).
3. Every other metric (CE, CV, drop, contig, perm_entropy) is
   unchanged for both routers, as expected for a pure plumbing change.

## What did NOT change

- **CE gap is still ~5%.** Phase 5.18, top-k 4.85–4.93. Top-k
  remains strictly better on validation cross-entropy at this scale.
  This wasn't going to change — selection is deterministic and we
  only sped up the path between Python and the kernel.
- **Phase locality (`contig_frac`) advantage is still modest.** ~0.46
  for phase vs ~0.40–0.46 for top-k, almost a wash. The original
  "phase has better routing locality" hypothesis does not survive at
  this scale.
- **Aux-loss ablation result stands.** Top-k without aux loss has
  CV ≈ 0.88–1.02 (vs 0.10–0.15 with aux) and drops 39–47% of tokens.
  Phase's load balance does not depend on any loss term, by
  construction.

## v2 numbers in full

Per capacity factor (cf), at 2000-step `tiny.yaml` on A10G:

| cf   | top-k CE | phase CE | top-k CV | phase CV | top-k drop | phase drop | top-k rt_ms | phase rt_ms | top-k tok/s | phase tok/s |
| ---- | -------- | -------- | -------- | -------- | ---------- | ---------- | ----------- | ----------- | ----------- | ----------- |
| 1.00 | 4.928    | 5.181    | 0.106    | 0.036    | 8.5 %      | 0 %        | 10.44       | **7.87**    | 17,705      | **17,814**  |
| 1.25 | 4.915    | 5.181    | 0.148    | 0.036    | 0.2 %      | 0 %        | 10.65       | **8.26**    | 17,326      | **17,509**  |
| 1.50 | 4.906    | 5.181    | 0.156    | 0.036    | 0.0 %      | 0 %        | 10.56       | **8.97**    | 17,652      | 17,441      |
| 2.00 | 4.845    | 5.181    | 0.146    | 0.036    | 0.0 %      | 0 %        | 10.63       | **8.09**    | 17,266      | **17,830**  |

Aux-loss ablation (no algorithmic change vs v1, but reproduced for
the record):

| cf   | aux=0.01 CV | aux=0 CV  | aux=0.01 CE | aux=0 CE | aux=0 drop |
| ---- | ----------- | --------- | ----------- | -------- | ---------- |
| 1.00 | 0.106       | **0.879** | 4.928       | 4.907    | 47.0 %     |
| 1.25 | 0.148       | **1.022** | 4.915       | 4.953    | 39.4 %     |

## Where the throughput stopped mattering

In v1 the headline story was "phase is 22 % slower because of Python
glue". In v2 that's gone. The new shape of the trade-off is:

- **Selection quality:** top-k is 5 % better on val CE.
- **Selection cost:** phase is **20–30 % cheaper** in routing
  wall-time, but it doesn't matter at this scale — `step_time_ms`
  is dominated by the MoE block's expert FFN compute (~230 ms/step
  for both), so the ~2.5 ms routing difference per forward is
  swallowed.
- **Robustness:** phase has **zero capacity-factor knob, zero aux loss,
  zero dropped tokens at every cf**, and CV is consistently 3× tighter.
  Top-k needs `aux_loss_alpha=0.01` to keep CV under 0.15; remove
  that and CV explodes to ~1.0.

So the v2 story for the paper is **not** "phase is faster". It's
**"phase is cheaper to invoke, structurally drop-free, and
aux-loss-free — at a 5 % CE cost on this 8-expert tiny config"**.

## Paper-grade single-line takeaway

> _On A10G with 8 experts and tiny.yaml, phase routing matches top-k's
> throughput while producing structurally balanced (CV ≈ 0.036),
> drop-free expert assignments without an auxiliary loss term and
> without capacity-factor tuning. Top-k, in contrast, requires both an
> aux loss (without which CV ≈ 1) and a capacity factor (without which
> ~9 % of slots are dropped at cf = 1.0). Phase pays for this with a
> ~5 % CE cost that does not close at higher cf._

## What's still open

1. **Stress regime.** All numbers above are for 8 experts on
   `tiny.yaml`. The interesting question for the paper is whether the
   CE gap closes (or reverses) when the regime stresses balance more:
   - `n_experts = 32` or `64` on `small.yaml`
   - longer training (`steps ≥ 10k`) — does phase catch up as top-k
     saturates the under-used experts?
   - tighter cf (cf = 1.0 with k = 2) — top-k loses 8.5 % of slots
     here; phase loses 0.
2. **Top-1 vs top-2.** Current sweep uses k = 2. Switch papers run
   k = 1 with a fallback. Worth probing whether phase's structural
   balance becomes a bigger win at k = 1 where top-k can't recover
   via the second-choice expert.
3. **Optional fairness-knob.** Phase currently has _one_ unused-on-
   purpose lever (the `base_density` × `width` product that sets per-
   expert capacity). Worth at least one sweep over `base_density` to
   show the trade-off curve, even if the production setting (0.3)
   already wins on balance.

## Recommended next step

Run the 32-expert `small.yaml` stress sweep:

- 4 cf × 2 routers = 8 runs (drop the noaux ablations — already
  characterised), ~$8–10 of Modal credit.
- If phase keeps CV ≈ 0.04 at 32 experts while top-k's CV ≈ 0.5+,
  the structural-balance argument becomes much stronger.
- If phase's CE catches up or reverses, that's the cleanest narrative
  for the paper.

After that, the paper can be drafted around a single, defensible
contribution: **"phase routing is a structural balance primitive,
realised as a deterministic Rust kernel, that matches Switch
throughput without aux-loss or capacity-factor tuning"**, with the
caveat that at scales where balance isn't load-bound, you give up
~5 % CE for it.
