# Phase 3 plan — controlled-trajectory experiments

Phase 2 is complete. Its conceptual scaffolding — the four-workload
taxonomy, the `Q(R, E ; φ)` decomposition, the support-ensemble
ontology, and the closed-form analysis of the anchor-mix family `E_t`
— is frozen in three artefacts that this plan will not revise:

- `dev/findings_phase2.md` — empirical record of the four-point sweep.
- `dev/support_ensembles.md` — definition of `E`, `Q(R, E ; φ)`, and
  the §5 catalogue of candidate controlling coordinates.
- `dev/controlled_overlap.md` — analytic specification of the
  trajectory family `E_t`, closed forms for six §5 coordinates, and
  the response-curve methodology.

Phase 3 begins from these documents and adds, on top of them, exactly
one experimental result and the operator-level rewrite it unblocks.

The over-riding principle is that this is no longer the exploratory
phase. Phase 3 is a small number of high-information experiments,
not a large number of exploratory ones. Experiment design now matters
more than experiment quantity.

---

## 1. Phase 3 objective

> **Identify which ensemble coordinate(s) control router diagnostics
> via controlled trajectory experiments.**

Operationally: for at least one diagnostic `φ` and at least one router
`R`, decide which entry of the §5 candidate list (overlap `O_J`,
per-sample column discrepancy `D_col^sample`, GF(2) rank, overlap
variance `σ²_J`, mutual information) best predicts the response curve
`ρ_φ(R, t) := Q(R, E_t ; φ)` on a controlled trajectory, and eliminate
at least one candidate as inconsistent with the observed shape.

Phase 3 is the phase that turns the §5 list from "candidates we cannot
distinguish" into "one or two surviving candidates and a list of
ruled-out explanations".

---

## 2. Primary experiment — controlled-overlap trajectory

The experiment is the one specified in `dev/controlled_overlap.md`.
This plan does not duplicate that specification; it points at it and
records the operational shape.

- **Trajectory.** The anchor-mix family `E_t` parameterised by
  `t ∈ [0, 1]`, recovering `dense_diverse` at `t = 0` and
  `dense_uniform`-up-to-row-symmetry at `t = 1`.
- **Sweep grid.** `m_t` values of `t` in `[0, 1]` (likely
  `m_t ∈ {11, 21}` linearly spaced, possibly augmented with denser
  spacing near `t = 0` and `t = 1` if early data shows interesting
  behaviour at the boundaries), `m_seed ≥ 8` seeds per `t`, all five
  routers (hash, phase_router, rs_add, rs_aff, rs_hyb).
- **Per-sample observables.** Both classes from
  `controlled_overlap.md` §3.0:
  - **Expectation-level coordinates:** `O_J(E_t)`,
    `E[H_col(E_t)] = log n` (constant, sanity check), `I(S_i ; S_j)`.
  - **Sample-level coordinates:** per-sample column discrepancy
    `D_col^sample`, GF(2) rank, per-sample column-coverage profile.
  - **Pairwise variance:** `σ²_J(E_t)` estimated from the row-pair
    sample.
- **Per-(t, seed, router) diagnostics.** The full Phase 2 suite:
  survival, CV, lf%, support validity, autocorrelation preservation.

The output is a long-format CSV with one row per
`(t, seed, router, φ)` cell, plus a sibling CSV with one row per
`(t, seed)` cell carrying the ensemble coordinates (since those are
router-independent).

Cost is dominated by router invocation. At `n = 1024`, `m_t = 11`,
`m_seed = 8`, `5` routers, total router calls ≈ 440. Cheap.

---

## 3. Diagnostic suite

Phase 3 reuses the Phase 2 diagnostic suite without modification.
Adding new diagnostics in this phase would dilute the experiment's
identifying power; the existing suite is sufficient to decide which
candidate `q_φ` controls each existing `φ`.

| diagnostic                     | what it probes               |
| ------------------------------ | ---------------------------- |
| `survival`                     | overall capacity sufficiency |
| `CV`                           | load-balance dispersion      |
| `lf%`                          | low-frequency load energy    |
| `validity`                     | support compliance           |
| `autocorrelation_preservation` | locality preservation        |

Per-(router, diagnostic) response curves `ρ_φ(R, t)` are the primary
output. The discriminating predictions per kernel are recorded in
`controlled_overlap.md` §6 and not repeated here.

---

## 4. Candidate coordinates under test

The candidates being tested as controlling quantities for the response
curves come from `support_ensembles.md` §5, with the closed forms on
`E_t` recorded in `controlled_overlap.md` §3:

| coordinate                       | shape on `E_t`            | linearity in `t` |
| -------------------------------- | ------------------------- | ---------------- |
| size-normalised overlap `O_J`    | `t² + (1 - t²)·d/n`       | quadratic        |
| expected col entropy `H_col`     | `log n` (constant)        | constant         |
| per-sample col discrepancy `D`   | `≈ t · (1 - d/n)`         | linear           |
| GF(2) effective support rank     | `(1 - t)·n + O(1)`        | linear (decr.)   |
| pairwise overlap variance `σ²_J` | `≈ t²(1 - t²)·(1 - d/n)²` | peaked interior  |
| pairwise mutual info `I`         | `≈ t² · log C(n, d)`      | quadratic        |

The deliberately distinct functional shapes (linear / quadratic /
peaked / constant) are the engine of identifiability. The two
quadratic candidates (`O_J` and `I`) are explicitly **not** separable
on this trajectory; that is acknowledged in `controlled_overlap.md`
§7.2 and is a known limit of the experiment.

---

## 5. Decision criteria

For each `(R, φ)` cell, the response curve `ρ_φ(R, t)` is fit against
each candidate's functional form via least squares, with
`(amplitude, baseline)` as free parameters per candidate. Candidates
are then compared on three grounds, in this order:

1. **Functional shape.** Does the curve have the right qualitative
   shape — linear, quadratic, peaked, flat? A linear-in-`t` candidate
   cannot fit a curve with non-zero curvature even with optimal
   amplitude/baseline; a peaked candidate cannot fit a monotone curve.
   This eliminates whole families of candidates before any residual
   comparison.
2. **Residual comparison.** Among candidates that pass the shape
   test, the one with smallest sum-of-squared-residuals after
   fitting is preferred. Differences are interpreted with AIC or
   leave-one-out cross-validation across `t` so that "the more
   flexible functional form" doesn't trivially win.
3. **Coordinate-pull plots.** As a sanity check on the fit, plot
   `ρ_φ(R, t)` directly against `q(E_t)` for each candidate `q`,
   not against `t`. The candidate against which the response curve
   collapses to a single one-dimensional relation is the one
   carrying the predictive structure. This is invariant to the
   parameterisation of the trajectory and is the cleanest visual
   discriminator.

The methodology principle stated in `controlled_overlap.md` §5.3 is
load-bearing here: **functional shape is more identifying than
monotonicity**. A monotone fit is necessary but not sufficient; the
test is whether the shape matches.

---

## 6. Exit conditions

Phase 3 is complete when, and only when, all five of the following
hold:

1. **At least one controlled trajectory implemented.** A working
   `controlled_overlap(n, d, t, seed) -> Workload` (or equivalent)
   exists in tree, exercised by an example binary, and produces
   samples whose ensemble coordinates match the closed forms of
   `controlled_overlap.md` §3 within statistical error.
2. **At least one candidate coordinate eliminated.** The fit
   procedure identifies at least one §5 candidate as inconsistent
   with the observed response curves (failed shape test or
   significantly higher residuals than competitors).
3. **At least one response curve robustly fit.** For at least one
   `(R, φ)` cell, a single candidate is identified as the controlling
   coordinate, with the fit stable under bootstrap resampling of
   seeds.
4. **Pushforward formalism written against surviving coordinate(s).**
   `dev/pushforward.md` exists and defines `R_*` on the coordinate(s)
   that survived (3), not on a guess.
5. **Paper framing updated accordingly.** `docs/paper.md`'s framing
   reflects the surviving coordinate(s) as the project's
   controlling-resource axis. This is the rewrite that
   `findings_phase2.md` deferred and `support_ensembles.md` flagged
   as unblocked only by an experimental result.

These exit conditions are deliberately minimal. Phase 3 is not "test
all five candidates against all five diagnostics on three trajectories
and write a comprehensive picture"; it is "produce one falsified
hypothesis and one surviving hypothesis, then update the formalism
to match".

---

## 7. Anti-goals

Phase 3 is at risk of expanding into "build many ensembles" — the
sparse-routing benchmark zoo — under the cover of "we should test on
more trajectories to be thorough". This expansion is to be resisted.

- **Do not introduce additional trajectory families** within Phase 3
  unless `E_t` _alone_ has produced an ambiguous result that another
  trajectory provably resolves. The follow-up trajectories listed in
  `controlled_overlap.md` §7.3 (degree-heterogeneity, locality-anchor,
  Markov-row, second-moment-only) are explicitly _post-Phase-3_ work.
- **Do not introduce new diagnostics** within Phase 3. The existing
  suite is sufficient.
- **Do not introduce new routers** within Phase 3. The five existing
  kernels are the test population.
- **Protect the trajectory-vs-workload distinction.** A trajectory is
  a one-parameter family designed to discriminate hypotheses, not a
  set of named profiles dressed up with a continuous label. Adding a
  trajectory only to enrich the benchmark zoo is a false economy: it
  costs experimental time without buying identifying power.
- **Do not generalise the operator framework before the experiment.**
  `pushforward.md` follows the experimental result, not preceding it.
  Writing pushforward language against a hypothetical surviving
  coordinate would re-import the entropy-shaped vocabulary that
  `support_ensembles.md` §4 deliberately rejected.

The shape of Phase 3 is: **one trajectory, one experiment, one fit,
one operator note, one paper rewrite.** Anything more is
post-Phase-3.

---

## 8. Operational sequence

The work order is fixed:

1. **`dev/phase3_plan.md`** — this document. _(complete on commit.)_
2. **Implement `controlled_overlap`.** Add the constructor in
   `src/workloads.rs` (or a new module if it grows beyond a single
   function), with tests confirming the closed forms of
   `controlled_overlap.md` §3 hold to within statistical error on
   sampled instances.
3. **Implement ensemble-coordinate estimators.** Add per-sample
   estimators for `O_J`, `D_col^sample`, `rank_GF2`, `H_col_sample`,
   and `σ²_J` in `src/metrics.rs` (or a sibling module). One pass per
   sample; cost is dominated by router calls, not estimation.
4. **Run the sweep.** A driver
   `examples/controlled_overlap_probe.rs` produces both CSVs
   described in §2.
5. **Fit response curves.** A small Python script
   `scripts/fit_response.py` consumes the long-format CSV, fits each
   `ρ_φ(R, t)` against each candidate's functional form, and emits
   per-`(R, φ)` residual tables and shape-match flags. Generates the
   coordinate-pull plots from §5.
6. **Write `dev/controlled_overlap_results.md`** — short result memo,
   in the style of `findings_phase2.md` but tighter: identify the
   surviving coordinate(s), the eliminated candidate(s), and any
   secondary findings worth recording. This document is the empirical
   counterpart to `controlled_overlap.md` and the prerequisite for
   the operator note.
7. **Write `dev/pushforward.md`** — defines `R_*` against the
   surviving coordinate(s) from (6). If multiple coordinates survive
   for different diagnostics, the operator is parameterised by `φ`
   rather than the project committing to a single closure axis
   prematurely.
8. **Rewrite paper framing in `docs/paper.md`.** This is the final
   Phase 3 deliverable.

Each step gates the next. In particular, (6) gates (7) and (8): the
operator and paper framings are committed to a coordinate only after
the experiment identifies which coordinate to commit to.

---

## 9. Phase 3 deliverables

Mapped to exit conditions:

| deliverable                            | source artefact                        | exit condition |
| -------------------------------------- | -------------------------------------- | -------------- |
| `controlled_overlap` constructor       | `src/workloads.rs` (or sibling)        | (1)            |
| coordinate estimators                  | `src/metrics.rs` (or sibling)          | (1)            |
| sweep driver                           | `examples/controlled_overlap_probe.rs` | (1)            |
| `controlled_overlap.csv` (long)        | run output                             | (1)            |
| `coordinates.csv` (one row per t,seed) | run output                             | (1)            |
| fit script                             | `scripts/fit_response.py`              | (2), (3)       |
| result memo                            | `dev/controlled_overlap_results.md`    | (2), (3)       |
| operator note                          | `dev/pushforward.md`                   | (4)            |
| paper rewrite                          | `docs/paper.md`                        | (5)            |

The result memo (`controlled_overlap_results.md`) is the artefact most
likely to be over-engineered. It should be short — comparable in
length to `findings_phase2.md`'s headline section — and structured as
"this is what we falsified, this is what survived, here are the fits,
here are the open questions". It should not duplicate the analytic
content of `controlled_overlap.md`; that document is the
specification, the result memo is the outcome.

---

## 10. State at start of Phase 3 — what is and isn't frozen

**Frozen** (will not be revised within Phase 3 unless the experiment
forces a revision, in which case the revision is itself a Phase 3
deliverable):

- `dev/findings_phase2.md` (empirical record).
- `dev/support_ensembles.md` (ontology and §5 catalogue).
- `dev/controlled_overlap.md` (trajectory specification, closed forms,
  per-kernel predictions).
- `dev/rank_select_progress.md` (engineering log; continues to receive
  build-level updates).
- The four-workload taxonomy in `src/workloads.rs` and the diagnostic
  suite in `src/metrics.rs`.

**Not yet frozen**, and produced by Phase 3:

- The implementation of `controlled_overlap(t)` and the coordinate
  estimators.
- The empirical result in `dev/controlled_overlap_results.md`.
- The operator framing in `dev/pushforward.md`.
- The paper framing in `docs/paper.md`.

**Out of Phase 3 scope** (referenced for record only; future phases):

- Additional trajectory families
  (`controlled_overlap.md` §7.3 list).
- Non-exchangeable ensembles.
- New routers, new diagnostics, new workloads.
- Performance work on the RS kernels (they are quality-parity-or-better
  on diverse supports; throughput is acknowledged in
  `findings_phase2.md` and not a Phase 3 concern).

---

## Status

This document is the contract for Phase 3. It commits the project to:

- one trajectory experiment, executed once, on the `E_t` family;
- one fit procedure with shape and residual tests;
- one result memo;
- one operator note;
- one paper rewrite.

The next concrete step is implementing
`controlled_overlap(n, d, t, seed)` and running the sweep. The
analytic content needed to interpret that sweep is already in
`controlled_overlap.md`. From here on, the bottleneck is evidence,
not vocabulary.
