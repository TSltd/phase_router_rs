# Phase 2 — Empirical findings

This document is the consolidated empirical writeup of the Phase 2 work
(`dev/rank_select_progress.md` is the running engineering log; this file
is the _interpretation_). It is written for the paper, not for the build.

## Summary in one sentence

> `phase_router`'s empirical advantage on the original benchmark is **not**
> a universal property of sparse-routing transport — it is a specific
> response to support degeneracy, and the four-workload taxonomy now lets
> us isolate exactly which mechanisms survive under which support
> geometries.

The contribution is therefore evolving from _"a novel sparse routing
primitive"_ into _"an experimental framework for decomposing sparse-routing
mechanisms"_. The router family is the vehicle for the decomposition, not
the headline.

---

## The four workloads as regime separators

Phase 2 introduced a deliberate workload taxonomy. After running the full
diagnostic sweep, the four profiles act as _regime separators_ — each
isolates one structural assumption that some kernel relies on:

| workload         | isolates                          | key property                                               |
| ---------------- | --------------------------------- | ---------------------------------------------------------- |
| `dense_uniform`  | pure global-scrambling effects    | every row's support is identical (`[0, d)`)                |
| `dense_diverse`  | support diversity _alone_         | constant degree `d`, uniformly random size-`d` subsets     |
| `sparse_diverse` | realistic heterogeneous occupancy | heavy-tailed (Pareto α=1.5) degrees, uniform subsets       |
| `block_local`    | per-row locality preservation     | constant degree, contiguous support `[o_i, o_i + d) mod n` |

This structure proved more useful than expected: each workload makes a
_different_ mechanism the load-bearing one, and the kernels split cleanly
along those mechanisms.

---

## The headline empirical result

### Survival rate at headroom 1.0× (N=1024, k=2, strong-hetero capacity)

| workload         | hash  | phase_router | rs_add | rs_aff    | rs_hyb    |
| ---------------- | ----- | ------------ | ------ | --------- | --------- |
| `dense_uniform`  | 0.786 | **0.897**    | 0.439  | 0.437     | 0.437     |
| `dense_diverse`  | 0.786 | 0.897        | 0.888  | **0.903** | 0.892     |
| `sparse_diverse` | 0.786 | 0.898        | 0.893  | **0.910** | 0.899     |
| `block_local`    | 0.786 | 0.897        | 0.881  | 0.884     | **0.908** |

`dense_uniform` is the _only_ regime where `phase_router` strongly
dominates the rank/select kernels. On every other workload all three RS
kernels match it within 1–3 percentage points, and on every workload other
than `dense_uniform` an RS variant is best.

### CV (load std/mean) on the same configuration

| workload         | hash | phase_router | rs_add | rs_aff | rs_hyb |
| ---------------- | ---- | ------------ | ------ | ------ | ------ |
| `dense_uniform`  | 0.50 | **1.05**     | 2.30   | 2.28   | 2.34   |
| `dense_diverse`  | 0.50 | 1.05         | 1.07   | 1.07   | 1.05   |
| `sparse_diverse` | 0.50 | 1.06         | 1.04   | 1.05   | 1.05   |
| `block_local`    | 0.50 | 1.05         | 1.01   | 1.01   | 1.07   |

Same picture: a 2× CV gap on `dense_uniform`, parity everywhere else.

### Capacity-load Pearson correlation

| workload         | hash   | phase_router | rs_add | rs_aff | rs_hyb    |
| ---------------- | ------ | ------------ | ------ | ------ | --------- |
| `dense_uniform`  | -0.010 | **0.884**    | 0.440  | 0.434  | 0.445     |
| `dense_diverse`  | -0.010 | 0.884        | 0.876  | 0.880  | 0.874     |
| `sparse_diverse` | -0.010 | 0.884        | 0.882  | 0.873  | **0.894** |
| `block_local`    | -0.010 | 0.884        | 0.852  | 0.872  | 0.883     |

Capacity alignment fails for the RS kernels on `dense_uniform` and recovers
to parity everywhere else. The mechanism is the same one that controls CV
(see next section).

---

## Why the hybrid collapses on `dense_uniform` — analytic identity

The hybrid was designed to combine the cumulative-degree phase from
`offsets_s` with intrinsic `select1` decode. On `dense_uniform` the hybrid
collapses to the additive variant by an analytic identity:

```
dense_uniform: every row has the same dense support, so popcount(s_bits[j]) = d
              for all j. Therefore offsets_s[j] = cumulative_ones(j) = j · d.

Hybrid phase: phi_j = offsets_s[j] mod d = (j · d) mod d = 0     for every j.

Therefore on dense_uniform the hybrid's per-row phase is identically zero,
the rank-space walk reduces to r = 0, 1, ..., d-1 for every row, and the
emitted candidate set per row is exactly {select1(row, 0), ..., select1(row, d-1)}
= the row's support. With identical supports across rows, this is the
same set for every row.
```

This is the _mechanistic_ completion of the structural-impossibility
argument logged in `dev/rank_select_progress.md`: not only is no
rank-space transport (additive, affine, …) able to break the symmetry on
identical-support workloads, the cumulative-degree phase that was the
hybrid's distinguishing feature _also_ trivialises in this regime.

The collapse is therefore not a consequence of insufficient mixing; it is
a consequence of the workload's algebraic structure.

---

## What each kernel actually preserves

Cross-referencing the results with the four-kernel ablation, a clean
mechanism / preservation table emerges:

| mechanism                        | preserved by                | breaks under                              |
| -------------------------------- | --------------------------- | ----------------------------------------- |
| global col-space decorrelation   | `phase_router`              | nothing in scope (it's column-space)      |
| support validity                 | RS kernels (all three)      | `phase_router` (val ≈ 0.30 on diverse)    |
| locality preservation (autocorr) | RS kernels, esp. hybrid     | `phase_router` (it scrambles globally)    |
| low-frequency mode suppression   | `phase_router`              | RS kernels on `dense_uniform`             |
| identical-support robustness     | `phase_router`              | all RS kernels (structural-impossibility) |
| support-sensitive routing        | RS kernels                  | `phase_router` (support-agnostic)         |
| permutation-free decode          | RS kernels                  | `phase_router` (needs `inv_perm_s`)       |
| capacity-aligned load            | `phase_router` + RS-diverse | RS on `dense_uniform`                     |

Two families fall out cleanly:

```
phase_router   = support-agnostic balancing via global col-space scrambling
RS kernels     = support-preserving balancing via intrinsic rank-space walks
```

That distinction is the headline contribution.

---

## Spectral diagnostics: a `low-freq energy share` signature

The diagnostic framework in `src/metrics.rs` includes a naive O(n²) DFT
of the per-target load. The bottom 1/16 of the non-DC band ("lf%")
yields a clean separator:

| workload         | hash  | phase_router | rs_add    | rs_aff    | rs_hyb    |
| ---------------- | ----- | ------------ | --------- | --------- | --------- |
| `dense_uniform`  | 0.131 | 0.123        | **0.526** | **0.534** | **0.506** |
| `dense_diverse`  | 0.131 | 0.123        | 0.127     | 0.124     | 0.130     |
| `sparse_diverse` | 0.131 | 0.136        | 0.121     | 0.129     | 0.150     |
| `block_local`    | 0.131 | 0.123        | 0.126     | 0.119     | 0.130     |

The RS kernels' low-frequency energy share **quadruples** on
`dense_uniform`, while `phase_router`'s remains at hash-baseline values
across all four workloads. This is precisely what the _low-discrepancy
distributed occupancy allocator_ interpretation predicts:

- `phase_router` walks `[offsets_s[j], + d) mod n` in _physical_ space
  with row-distinct (cumulative-degree) starting points and decodes
  through the support-independent `inv_perm_s`. Coherent large-scale
  clustering of loads is suppressed: the lf% stays at noise level.
- The RS kernels on `dense_uniform` have no such low-discrepancy
  scrambler — every row samples the same dense block — and large-scale
  load modes survive. lf% rises sharply.

This is the cleanest _direct measurement_ in the project of the
discrepancy-control mechanism. The low-frequency energy share is now a
diagnostic worth keeping.

---

## A subtler observation: support entropy substitutes for routing entropy

Comparing `dense_uniform` (where RS kernels collapse) to `dense_diverse`
(where they recover), the _only_ thing that changes is whether the source
supports are identical or independently random. The RS kernels gain back
their balancing properties simply because supports differ across rows.

This suggests a deeper claim:

> Once supports differ sufficiently across rows, explicit global
> scrambling becomes much less important. Support diversity itself acts
> as a discrepancy suppressor.
>
> ⇒ _support entropy substitutes for routing entropy._

In the language of the discrepancy-control framing: the
`inv_perm_s × offsets_s` machinery in `phase_router` is one way to
inject a deterministic, low-discrepancy "shake" across rows; an
ensemble of diverse supports is another way to achieve the same effect
_for free_ from data structure rather than algorithm structure.

This is a useful theoretical handle for future work and re-frames
"sparse routing" as discrepancy control over support ensembles rather
than randomised matching.

---

## What this means for the original `phase_router` paper framing

### Before Phase 2

> `phase_router` achieves superior balancing via phase mixing.

### After Phase 2

> `phase_router`'s dominant advantage appears specifically when:
>
> - source supports are degenerate / identical, and
> - routing must decorrelate rows independently of support structure.
>
> In diverse-support regimes, intrinsic rank/select transport performs
> comparably while preserving support validity. The original kernel is
> best understood as a _low-discrepancy distributed occupancy allocator_
> that exploits the support-independent `inv_perm_s` scrambler precisely
> in the regime where support cannot supply its own discrepancy.

The reframing is sharper, more falsifiable, and lines up with both the
analytic identity (`offsets_s[j] mod d = 0`) and the spectral signature
(`lf%` jump on `dense_uniform`).

---

## Caveats and scope of evidence

To the extent that the diagnostic suite is not exhaustive:

- **Only one capacity profile** is exercised so far
  (10% × 8.0 + 20% × 2.0 + 70% × 1.0). The regime separation should not
  depend strongly on the capacity distribution, but a sensitivity sweep
  is worth doing.
- **`sparse_diverse`** uses Pareto α=1.5; results may shift for very
  heavy tails (α close to 1) where the mean degree estimate breaks
  down. The current implementation clamps to `[1, n]` but a wider sweep
  on α should be checked before paper-grade claims.
- **Performance**. RS kernels are 5–10× slower than `phase_router` at
  large `N` (e.g. 261–581 ms vs 36–72 ms at N=4096, k=2, headroom=1.2),
  dominated by reservoir RNG and `select1` dispatch. This is a real
  cost. The paper needs to acknowledge it; the RS family's _quality_
  parity-or-better on diverse supports does not yet translate to
  _throughput_ parity.
- **Reservoir sampling order-independence** is the structural reason
  the RS kernels collapse on identical supports. A non-reservoir
  selection rule (e.g. priority-driven) might decouple them, but is
  outside scope.

---

## Mechanism table (compact form)

| mechanism                      | preserved by        |
| ------------------------------ | ------------------- |
| global col-space decorrelation | `phase_router`      |
| support validity               | RS kernels          |
| locality preservation          | RS kernels / hybrid |
| low-frequency suppression      | `phase_router`      |
| identical-support robustness   | `phase_router`      |
| support-sensitive routing      | RS kernels          |
| permutation-free decode        | RS kernels          |

---

## What this changes about future work

The natural next directions, in rough order of theoretical leverage:

1. **Discrepancy theory framing.** Treat `phase_router` as a _quasirandom
   occupancy walk_: characterise it via star-discrepancy bounds on the
   per-target load distribution. The lf% measurement is the empirical
   shadow of this; the analytic version would tie the kernel to the
   established quasirandom-sequences literature.
2. **Support-ensemble characterisation.** Define the support entropy
   under which a kernel becomes redundant for discrepancy control, and
   verify experimentally where that threshold sits. The four-workload
   sweep is a discretisation of one axis of this question.
3. **Hybrid kernel beyond identical supports.** The hybrid is currently
   only "the additive kernel with phase = `offsets_s[j] mod d`". A
   richer hybrid that uses _full_ `offsets_s` rotation in rank-space
   (i.e. `r = (offsets_s[j] + i) mod degree` _with degree varying per
   row_) is more interesting on `sparse_diverse` and is the obvious
   next experiment.
4. **Spectral load shaping.** The lf% measurement suggests a direct
   training signal: an MoE router trained to minimise low-frequency
   load energy approximates a discrepancy-controlled allocator. This
   is speculative but a clean future direction.
5. **A non-reservoir selection rule.** To break the structural
   impossibility on identical-support workloads, replace reservoir
   sampling with a priority-driven choice that _can_ depend on row
   identity. This would let intrinsic-decode kernels recover on
   `dense_uniform` and would close the last regime in which they lose
   to `phase_router`.

The paper rewrite (`docs/paper.md`) is intentionally _not_ on this list
yet. The framing is still settling; another conceptual iteration is
likely cheaper before locking language into the manuscript.

---

## Reproducing the numbers in this document

```bash
cargo test --features rank-select
cargo run --release --features rank-select --example spectral_probe   # → spectral_probe.csv
cargo run --release --features rank-select --example quality_probe
cargo run --release --features rank-select --example moe_bench_rs     # → moe_results_rs.csv
```

`spectral_probe.csv` carries the discrepancy / spectrum / Jaccard /
correlation columns; `moe_results_rs.csv` carries the survival sweeps
across (workload, experiment, parameter, method).
