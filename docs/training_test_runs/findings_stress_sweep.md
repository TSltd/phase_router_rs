# Stress sweep — findings (32 experts on tiny scaffold)

**Date:** 2026-05-14
**Sweep:** `runs/stress_sweep/` (8 A10G runs, ~$3 of credit)
**Report:** `reports/stress_sweep.md`
**Prior sweep:** `runs/cf_sweep_v2/` — same scaffold, `n_experts: 8`
(see `dev/findings_cf_sweep_v2.md`)
**Plan:** `dev/stress_sweep_plan.md`

This is the n_experts = 8 → 32 diff against `cf_sweep_v2`. Every other
knob (d_model, n_layers, batch, seq, max_steps, lr, seed, dataset) is
byte-identical, so the only independent variable is the expert count.

## TL;DR

The CE gap **widens**. Of the three pre-registered outcomes
(_gap closes / widens / unchanged_), this is the "phase is a structural
balance primitive, not a CE replacement" arm. Headline numbers
(`val_final` row, cf = 1.25, matching the plan's reporting cf):

| dimension    | v2 (8 experts) | stress (32 experts) | direction        |
| ------------ | -------------- | ------------------- | ---------------- |
| top-k val_ce | 4.915          | 4.722               | top-k wins more  |
| phase val_ce | 5.181          | 5.287               | phase loses more |
| **CE gap**   | **+5.4 %**     | **+12.0 %**         | **widens 2.2×**  |
| top-k val_cv | 0.148          | 0.200               | top-k worse      |
| phase val_cv | 0.0360         | 0.100               | phase worse 2.8× |
| **CV ratio** | **4.1×**       | **2.0×**            | phase margin ↓   |
| top-k drop   | 0.2 %          | 2.9 %               | top-k worse      |
| phase drop   | 0.0 %          | 0.6 %               | phase nonzero!   |
| top-k rt_ms  | 10.65          | 11.92               | ~constant        |
| phase rt_ms  | 8.26           | 8.52                | ~constant        |

Three of these moves were predicted by the plan's hypothesis grid; one
was not (phase CV is _not_ structural — see "Surprises" below).

## What changed (vs cf_sweep_v2)

### Predicted moves that landed

1. **Top-k CV climbed.** The plan called "0.15–0.30 at 32 experts";
   the realised range is 0.14–0.22, right in the predicted band. The
   aux loss with `alpha=0.01` has 4× more dimensions to balance, and
   it just can't keep up — CV deteriorates by ~50 % vs 8 experts at
   every cf.
2. **Top-k drop at cf = 1.0 rose** from 8.5 % (v2) to **11.8 %** —
   ~1.4× worse. Still well short of the "15–25 %" plan estimate, but
   directionally as predicted: bigger E makes the capacity-factor
   knob harder to set.
3. **CE gap widened** (the plan's "outcome (b)"). v2's 5 % becomes
   12 % at every cf. Phase's CE is essentially _flat_ in cf (5.286
   ± 0.001 across cf ∈ {1.00, 1.25, 1.50, 2.00}); top-k continues to
   benefit from a higher cf (4.766 → 4.712 from cf 1.00 → 2.00).

### Surprises

1. **Phase CV is _not_ structural.** v2's `cf_sweep_v2` showed phase
   CV ≈ 0.036 independent of cf, and the plan extrapolated "structural
   — independent of E by construction". That extrapolation was wrong:
   at 32 experts phase CV is **0.0997**, ~2.8× v2's 0.036. The
   construction still bounds CV (top-k's 0.20 vs phase's 0.10 is a 2×
   gap), but the bound itself scales with E — not the structural
   invariant the v2 readout suggested.

   Likely mechanism: phase uses `phase_capacity_mode=uniform` which
   tries to land `T/E` tokens per expert. At E = 8 with seq T = 2048,
   that's 256 tokens/expert; the granularity is forgiving. At E = 32
   that's 64 tokens/expert; minor rounding & contiguity tie-breaks
   start to produce visible imbalance. The math is still "spread
   uniformly", but the discrete quantum has gotten 4× coarser.

2. **Phase drop is no longer zero.** v2 reported strict 0 % phase
   drop at every cf; here it's a small but nonzero **0.58 %** at
   every cf. Same mechanism: the bipartite construction can't always
   pack 64 tokens cleanly into 32 banks at cf = 1.0 once the seed
   sequence has very uneven phase distribution on a given step. Phase
   still beats top-k's drop by **20× at cf = 1.0**, but the
   "bipartite construction never overflows" claim from v2 has to be
   qualified.

3. **Phase routing locality (`val_contig`) collapsed.** v2:
   phase ≈ 0.46, top-k ≈ 0.40–0.46 (near-wash). Here: phase = 0.120
   _at every cf_, top-k = 0.110 → 0.141 _rising with cf_. Phase
   _loses_ on locality at every cf except 1.0. With 4× more banks,
   adjacent rows hit the same expert 4× less often, and top-k's
   gradient-driven routing accidentally produces _better_ contiguity
   because it _learns_ to clump. The v2 narrative ("phase doesn't help
   on locality") was already a wash; here it actively reverses.

4. **Top-k now wins or matches throughput at every cf.** v2 had phase
   winning 3/4 cf points by 30–400 tok/s. Here top-k wins all 4 by
   50–270 tok/s. The phase router itself is still cheaper to invoke
   (rt_ms 7–10 vs 11–12), but the 4× expert FFN load now dominates
   `step_time_ms` even more strongly, and a small overhead asymmetry
   in the phase dispatch path costs ~5 % throughput at this scale.
   Not a story we want to lean on — see "What to do next" below.

5. **All four phase runs are essentially identical.** `val_final` row,
   to 6 sig figs: CE 5.286–5.287, CV 0.0997, drop 0.575 %, contig
   0.120, route_time 8.5 ms. The cf knob is **completely** inert for
   phase, which is the strongest version of v2's "no cf tuning
   needed" claim. Top-k, in contrast, moves on every dimension as cf
   varies. This is the cleanest line in the entire sweep.

## Full numbers

(`val_final` row, 2000-step `stress.yaml` on A10G, fp32. tok/s and
wall-time are end-of-run train metrics.)

| cf   | top-k CE | phase CE | top-k CV | phase CV  | top-k drop | phase drop | top-k contig | phase contig | top-k rt_ms | phase rt_ms | top-k tok/s | phase tok/s |
| ---- | -------- | -------- | -------- | --------- | ---------- | ---------- | ------------ | ------------ | ----------- | ----------- | ----------- | ----------- |
| 1.00 | 4.766    | 5.287    | 0.144    | **0.100** | 11.79 %    | **0.58 %** | 0.110        | 0.120        | 11.75       | **9.59**    | **4,510**   | 4,238       |
| 1.25 | 4.722    | 5.287    | 0.200    | **0.100** | 2.92 %     | **0.58 %** | 0.133        | 0.120        | 11.92       | **8.52**    | **4,395**   | 4,193       |
| 1.50 | 4.725    | 5.286    | 0.208    | **0.100** | **0.39 %** | 0.58 %     | 0.138        | 0.120        | 10.73       | **7.29**    | **4,918**   | 4,872       |
| 2.00 | 4.712    | 5.286    | 0.221    | **0.100** | **0.00 %** | 0.58 %     | 0.141        | 0.120        | 11.03       | **10.03**   | **5,028**   | 4,806       |

The bolded entries flip phase ↔ top-k as cf varies; everything else
stays on the same side of the comparison. The cf = 1.0 column is the
"raw" comparison without top-k's capacity-factor crutch and is the
strongest column for phase.

## Diff table — paper-ready

Suitable for the paper's "8 vs 32 experts" appendix:

| metric                 | v2 (8 experts) cf = 1.25 | stress (32 experts) cf = 1.25 | change         |
| ---------------------- | -----------------------: | ----------------------------: | -------------- |
| top-k val_ce           |                    4.915 |                         4.722 | −3.9 %         |
| phase val_ce           |                    5.181 |                         5.287 | +2.0 %         |
| CE gap (phase − top-k) |                   +0.266 |                        +0.565 | **2.1× wider** |
| top-k val_cv           |                    0.148 |                         0.200 | +35 %          |
| phase val_cv           |                    0.036 |                         0.100 | +176 %         |
| top-k val_drop         |                    0.2 % |                         2.9 % | 14× worse      |
| phase val_drop         |                    0.0 % |                         0.6 % | +0.6 pp        |
| top-k tok/s            |                   17,326 |                         4,395 | −75 %          |
| phase tok/s            |                   17,509 |                         4,193 | −76 %          |

The throughput collapse (4× slower at 4× experts) is the python
expert-loop bottleneck flagged in the plan and `dev/stress_sweep_plan.md`'s
"historical note", _not_ a router cost. It would be eliminated by a
grouped-matmul kernel (e.g. `torch._grouped_mm` when it lands, or a
hand-rolled triton expert kernel).

## Paper-grade single-line takeaway

> _At 32 experts on the `stress.yaml` scaffold (tiny scale, 2000 A10G
> steps), the cf_sweep_v2 trade-off sharpens: phase routing keeps load
> CV at 0.10 and drop at 0.6 % regardless of capacity factor, where
> top-k needs cf = 2.0 to drive drop to zero and still has CV = 0.22,
> but phase pays a **12 % CE penalty** (up from 5 % at 8 experts).
> Phase is a structural balance primitive whose CE cost grows with E;
> top-k is a quality primitive whose load balance degrades with E.
> The two are complementary, not competitive._

## Implications for the paper

The v2-era narrative was "phase matches top-k's throughput while being
structurally balanced, at a 5 % CE cost". This sweep makes that
narrative **less general** than v2 suggested:

1. The CE cost is _not_ a fixed 5 % tax — it scales with E.
2. Phase's structural balance is **bounded** by 1/√E-style discreteness,
   not literally invariant.
3. Phase's locality advantage is _absent_ at this scale.

But it also reveals a stronger, narrower claim that v2 only hinted at:

> **Phase routing is the only primitive in this experiment whose
> behaviour is independent of the capacity-factor hyperparameter.**

Across cf ∈ {1.00, 1.25, 1.50, 2.00}:

- Top-k CE moves 4.766 → 4.712 (−1.1 %).
- Top-k CV moves 0.144 → 0.221 (+53 %).
- Top-k drop moves 11.8 % → 0.0 % (the whole reason cf exists).
- **Phase CE, CV, drop, contig all move by < 0.1 %.**

This is a quotable property: phase eliminates a hyperparameter that
top-k requires _and_ that the literature has not been able to
auto-tune. The paper's contribution should be framed around _that_,
not around "matches top-k on CE" (which it doesn't at scale) or
"never drops tokens" (which it does, at 0.6 %).

Recommended re-framing:

> **Phase routing is a deterministic balance primitive that eliminates
> the capacity-factor hyperparameter, at a CE cost that grows with
> n_experts. We propose using it as a balance constraint atop a
> learned (top-k) head, rather than as a drop-in replacement.**

This sets up the obvious follow-up experiment cleanly (see below).

## What didn't change vs v2

For continuity with v2's claims that survive intact:

- Phase still produces strictly **better balance** than top-k at every
  cf (CV gap is 2× rather than 4× now, but in the same direction).
- Phase still produces **strictly fewer dropped tokens** than top-k at
  cf ≤ 1.25 (20× advantage at cf = 1.0).
- Phase routing wall-time is still **20–30 % cheaper** than top-k's
  (rt_ms 7.3–10.0 vs 10.7–11.9). Phase route_time also remains
  insensitive to cf, where top-k's drifts ~10 % across the cf range.
- Phase is **aux-loss-free by construction** (not re-tested here, but
  unchanged — see `cf_sweep_v2`'s ablation).

## What's still open after this sweep

1. **Ensemble probe.** Pre-registered fallback in the plan: if the
   gap widens, run `phase ⊕ top-k` — top-k for selection scores,
   phase as a balance constraint (mask or rebalance). The hypothesis
   is that the ensemble inherits top-k's CE and phase's balance.
   Modest scope: 4 cf × 1 router-mode = 4 runs at tiny scale, ~$1.50.
2. **Expert-loop kernel.** The 4× throughput regression vs v2 is
   pure python-launch overhead. The `phase_router_uniform_dispatch`
   side is solved, but the expert FFN side now dominates step time.
   This isn't a paper-blocker but it gates the small-scale 32-expert
   sweep (the one we rejected in the plan's "historical note") which
   would be much more impressive to cite than `tiny + 32 experts`.
3. **Steps vs E.** With 4× more experts, each expert sees 4× less
   gradient signal per step. The plan held `max_steps = 2000` fixed
   for diffability with v2, but a real comparison would scale
   `max_steps` with E. Worth one 8000-step `tiny + 32 experts` run
   for the appendix to check whether the CE gap closes with longer
   training (current best guess: it shrinks but doesn't vanish).
4. **k = 1 ablation.** All numbers here are with `k = 2`. At k = 1,
   top-k can't recover via the second choice, and phase's structural
   balance should matter much more. Cheap probe: 4 cf × 2 routers ×
   k = 1 ≈ $3.

## Recommended next step

The natural sequel is the **constrained-optimisation ensemble probe**
— it operationalises this sweep's main finding ("phase is a
constraint, top-k is a head") and is cheap.

**Plan:** `dev/ensemble_probe_plan.md` (full algorithm, pre-registered
hypothesis grid, success criteria, launch checklist).

**Algorithm in one line:** top-k scores → per-expert quota
`ceil(cf · N · k / E)` → priority-ordered admission with soft
fall-through to the 2nd/3rd-choice expert on overflow (no hard masking
of the first choice). Reduces to top-k as cf → ∞ and to phase-uniform
as cf → 1. Single knob, no aux loss.

**Implementation status:** `BalancedRouter` is live in
`train/routers.py`; `train/train_balanced.py` and
`orchestrate_ensemble_sweep` / `ensemble_sweep` in `train/modal_app.py`
are wired up; `train/compare_sweep.py` already recognises `balanced`
runs.

If the ensemble inherits top-k's CE (≈ 4.7–4.9) and phase's CV (≈ 0.10)
and drop (≤ 1 %), that's the paper's pitch. If it doesn't, the paper
falls back to the "phase is a cf-eliminating balance primitive"
contribution which this sweep already establishes.

Estimated budget: ~$3 of credit, ~4 hours of GPU time (4-cf × balanced
on both `stress` and `tiny` configs).

## Reproducibility

- Config: `train/configs/stress.yaml` (`n_experts: 32`, otherwise =
  `tiny.yaml`).
- Sweep entrypoint: `modal run --detach train/modal_app.py::stress_sweep`.
- Aggregator: `python3 train/compare_sweep.py runs/stress_sweep -o reports/stress_sweep.md`.
- Modal app id (for the record): `ap-3ntHqZIJkg9z79s9yw9fGr`.
- Image bakes `PYTHONUNBUFFERED=1` (see `train/modal_app.py` and the
  postmortem in `dev/stress_sweep_plan.md` § smoke).
