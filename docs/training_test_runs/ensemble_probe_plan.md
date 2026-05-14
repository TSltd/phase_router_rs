# Ensemble probe — phase ⊕ top-k as constrained optimisation

**Date:** 2026-05-14
**Sequel to:** `dev/findings_stress_sweep.md`
**Status:** plan + implementation; smoke pending.
**Budget target:** ~$3 of Modal credit (8 runs).

## Motivation

`dev/findings_stress_sweep.md` established that at 32 experts:

- **Top-k solves** _maximise local affinity_ — and its load balance
  deteriorates with E (CV 0.14 → 0.22 across cf).
- **Phase solves** _minimise global discrepancy_ — and its CE cost
  grows with E (gap widens from 5 % to 12 % vs v2).

The natural mathematical object that absorbs both is

> **maximise affinity subject to occupancy constraints.**

Top-k is the affinity term with no constraint, phase is the constraint
with no affinity term, and the ensemble is the constrained programme.
This document specifies the ensemble, the experimental grid that tests
it, and the success criteria that decide whether it becomes the
paper's main contribution.

## Design constraints (from review)

1. **Do not hard-mask top-k's first choice.** Hard masking recreates
   dead-expert / hard-discontinuity / brittle-routing failure modes
   that the literature documents repeatedly.
2. **Use a soft admission process.** Three families considered:
   - expert quotas (hard ceiling, but fall-through to lower-ranked
     experts on overflow);
   - occupancy-aware re-ranking (penalty `−λ · load/quota` added to
     the gate score, then standard top-k);
   - balanced top-k relaxation (Sinkhorn-style soft assignment).
3. **Preserve cf as the one tunable knob.** Phase already removed the
   cf-dependence of `(CE, CV, drop)`; the ensemble should _at worst_
   preserve that and ideally _stay closer to top-k's CE_ than pure
   phase across the cf range.

We pick (1) **priority-ordered occupancy-aware admission** — a hybrid
of expert quotas + occupancy-aware reranking. It is the closest
discrete analogue of constrained gradient ascent on affinity subject
to capacity, and it has a single deterministic algorithm rather than a
solver loop.

## The `BalancedRouter` algorithm

Pseudo-code (see `train/routers.py` for the live implementation):

```
Input
    logits   :  (N, E)     # gate logits
    k        :  int        # experts per token
    cf       :  float      # capacity factor; quota = ceil(cf · N · k / E)
    overflow :  int = 2    # candidates considered per token  (k + overflow)

1. gate = softmax(logits, dim=-1)                          # (N, E)
2. weights_top, idx_top = gate.topk(k + overflow, dim=-1)  # candidate set
3. quota = ceil(cf · N · k / E)                            # per-expert ceiling
4. loads = zeros(E)
5. priority = argsort(-gate.max(dim=-1))                   # max-affinity first
6. out_idx     = -1  of (N, k)
   out_weights =  0  of (N, k)
7. for tok in priority:                                    # serial, O(N · (k + overflow))
       slot = 0
       for c, w in zip(idx_top[tok], weights_top[tok]):
           if loads[c] < quota:
               out_idx[tok, slot]     = c
               out_weights[tok, slot] = w
               loads[c]              += 1
               slot                  += 1
               if slot == k: break
       # if slot < k after candidate list exhausted → that slot stays -1
       # (drop), exactly as Switch does on capacity overflow.
8. weights ← weights / weights.sum(-1).clamp_min(eps)
```

### Why this is what the framing suggests

- **Top-1 choice is never hard-masked.** A token's top expert is only
  skipped if _every_ token that ranked it more highly already filled
  the quota — which is the discrete dual of "soft constraint, satisfied
  almost-everywhere".
- **Lower-affinity fall-through is the soft part.** A token whose top
  choice is full is _demoted_ to its 2nd / 3rd / … choice rather than
  silently dropped. Drops only happen when even the `(k + overflow)`-th
  choice is full, which under any reasonable cf and overflow is
  vanishingly rare.
- **Affinity ordering is preserved by priority.** Processing tokens in
  descending `max_e gate[tok, e]` order means the most-confident tokens
  get their first choice; less-confident tokens accept fall-through.
  Mutual information between token identity and expert identity is
  preserved at the high-affinity tail and only relaxed at the low-
  affinity tail, which is where it should be relaxed.
- **No aux loss.** The constraint is in the admission rule, not in the
  objective.
- **One knob, same shape as Switch.** `cf` directly sets the quota
  exactly as top-k uses it. At cf = 1.0 the constraint is tight; at
  cf = 2.0 it is essentially inactive. So `cf` becomes a continuous
  interpolation knob _between phase-like (cf=1.0) and top-k-like
  (cf → ∞)_ behaviour, which is the cleanest possible sweep shape.

### Relation to prior work

- **Expert Choice routing** (Zhou et al. 2022) implements the dual:
  each expert picks its top-C tokens. That eliminates the "tokens
  miss their first choice" failure mode but breaks autoregressive
  causality (an expert's pick depends on future tokens). Our router
  is _token-driven_ and stays causally compatible with autoregressive
  language modelling.
- **BASE layers** (Lewis et al. 2021) solve a linear assignment
  problem per layer. That is the convex relaxation of what we do
  here; we use a greedy priority-driven assignment instead, which is
  O(N k) vs BASE's O(N²) Hungarian.
- **Switch / top-k with capacity-factor** is the cf → ∞ limit of our
  router (every token gets its first choice, plus drops).
- **Phase uniform** is approximately the cf = 1.0 limit of our router
  with a randomised priority and no fall-through. The ensemble
  should strictly dominate phase by exposing affinity, and should
  strictly dominate top-k by enforcing balance.

## Experimental grid

| sweep                  | regime        | router   | cf grid                  | runs |
| ---------------------- | ------------- | -------- | ------------------------ | ---- |
| **A — 32 experts**     | `stress.yaml` | balanced | {1.00, 1.25, 1.50, 2.00} | 4    |
| **B — 8 experts diff** | `tiny.yaml`   | balanced | {1.00, 1.25, 1.50, 2.00} | 4    |

Total: **8 runs ≈ 4 GPU-hours ≈ $3**.

Sweep B exists so we can diff balanced against both `cf_sweep_v2`
(8 experts) and `stress_sweep` (32 experts) using identical scaffolds.
With both regimes in hand we can plot the CE / CV / drop curves vs
`n_experts` for three routers (top-k, phase, balanced) — the exact
appendix table the paper needs.

`overflow` is held at **2** (i.e. each token sees its top `k + 2 = 4`
candidates at k = 2) for the first pass. A sensitivity scan on
overflow ∈ {0, 1, 2, 4} is a cheap follow-up if the headline numbers
look encouraging.

## Pre-registered hypotheses

| outcome                                                                   | interpretation                                                                                       | next step                                                                                     |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **Sweet spot.** CE within 1 % of top-k, CV ≤ 0.10, drop ≤ 1 % at every cf | Constrained-optimisation framing works. This is the paper's headline.                                | Write paper around `balanced` as the proposed router; phase becomes the constraint primitive. |
| **CE win, CV loss.** CE ≤ top-k, CV between phase and top-k               | The fall-through is too generous — quotas aren't actually binding. Tighten overflow / increase λ.    | Re-run with `overflow ∈ {0, 1}` to make quota more binding.                                   |
| **CV win, CE loss.** CV ≈ phase, CE between phase and top-k               | The constraint is binding but priority ordering is destroying affinity. Probably overflow too small. | Re-run with `overflow ∈ {3, 4}` and/or learnable affinity-temperature.                        |
| **Wash.** Numbers somewhere between phase and top-k on every dimension    | Greedy priority-driven assignment isn't expressive enough — need Sinkhorn or learned admission.      | Drop balanced; pitch paper on the cf-elimination claim from stress_sweep alone.               |

The bar for the headline claim is the "sweet spot" outcome. Anything
else is a partial result that we report as an honest negative in the
appendix.

## Success criteria

A run is a **win** for balanced if, at cf = 1.0 on the 32-expert sweep:

- `val_ce` ≤ 4.80 (within 1 % of top-k's 4.766; phase scored 5.287),
- `val_cv` ≤ 0.105 (matches phase's 0.0997),
- `val_drop` ≤ 0.010 (matches phase; vs top-k's 0.118),
- `val_contig` ≥ 0.110 (no worse than phase),
- `val_rt_ms` ≤ 12 (overhead within 30 % of pure top-k).

If three of five thresholds are met, the paper's contribution rewrites
around balanced. If fewer than three, we fall back to the cf-elimination
story already validated in `dev/findings_stress_sweep.md`.

## What about a Rust kernel?

Not yet. The python implementation is O(N · (k + overflow)) with N ≤
2048 at this scale, so it'll cost ~0.5 ms per forward — comparable to
top-k's 11 ms (dominated by softmax & sort) and to phase's 8 ms (rust
kernel). If the ensemble wins the science argument we'll harden the
admission loop into a fused rust kernel — the algorithm fits naturally
into the existing `phase_router_rs` crate as a new
`phase_router_balanced_dispatch` entry. Out of scope for the probe.

## Implementation surface

Live in this commit:

- **`train/routers.py`** — new `BalancedRouter` class implementing the
  algorithm above. Selection is on CPU (small Ns); gradient flows
  through `weights = softmax(logits).gather(idx)` exactly like the
  phase router.
- **`train/train_balanced.py`** — thin entry point (`run("balanced")`).
- **`train/_loop.py`** — `make_router_factory` recognises
  `name == "balanced"`.
- **`train/modal_app.py`** — `train_one(router=...)` dispatches to
  `train_balanced.py` for `router == "balanced"`; new
  `orchestrate_ensemble_sweep` + `ensemble_sweep` entrypoint runs the
  4-cf grid at one config.
- **`train/configs/stress.yaml`** — unchanged; balanced consumes the
  same scaffold.

## Launch checklist

1. **Smoke** (200 steps, foreground, ~5 min):

   ```bash
   modal run train/modal_app.py::smoke_balanced --config stress --max-steps 200
   ```

   Verify: no crash, `metrics.jsonl` written, `step_time_ms` close to
   top-k's, `load_cv` near 0.10, `dropped_frac` < 0.02.

2. **Stress sweep** (4 cf × balanced, 2000 steps, detached, ~$1.50):

   ```bash
   modal run --detach train/modal_app.py::ensemble_sweep --config stress
   ```

3. **Tiny sweep** (4 cf × balanced, 2000 steps, detached, ~$1.50):

   ```bash
   modal run --detach train/modal_app.py::ensemble_sweep --config tiny --sweep-name ensemble_tiny_sweep
   ```

4. **Pull:**

   ```bash
   SKIP_MODEL=1 SWEEP=ensemble_stress_sweep bash scripts/pull_sweep.sh
   SKIP_MODEL=1 SWEEP=ensemble_tiny_sweep   bash scripts/pull_sweep.sh
   ```

5. **Aggregate:**

   ```bash
   python3 train/compare_sweep.py runs/ensemble_stress_sweep -o reports/ensemble_stress_sweep.md
   python3 train/compare_sweep.py runs/ensemble_tiny_sweep   -o reports/ensemble_tiny_sweep.md
   ```

   The aggregator's regex (`^(?P<router>topk|phase)_cf...`) needs a
   one-character tweak to recognise `balanced_cf*` directories — see
   "Open work" below.

6. **Write findings** to `dev/findings_ensemble_probe.md`, diffing
   against both `cf_sweep_v2` (8 experts) and `stress_sweep` (32
   experts).

## Open work before launch

- [ ] Extend `train/compare_sweep.py::SUBDIR_RE` to match `balanced`
      runs and add a style entry to `SERIES_STYLE`.
- [ ] Smoke the balanced router locally on CPU first (`tiny_cpu.yaml`,
      100 steps) before paying Modal for a 200-step smoke. Cost: 0.
- [ ] Confirm `BalancedRouter._select` returns int64 (the existing
      `_loop.py` path expects this).

## Why now is the right time to spend this $3

The stress sweep already gives us a defensible paper around
cf-elimination + structural balance. The ensemble probe is the test
that decides whether the paper has _one_ contribution (balance
primitive) or _two_ (balance primitive + constrained-optimisation
router). The downside risk is bounded ($3, two hours) and the upside —
a router that pareto-dominates Switch on balance _and_ matches it on
CE without an aux loss — is the most paper-worthy result available
inside this budget.
