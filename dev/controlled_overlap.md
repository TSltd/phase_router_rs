# Controlled-overlap ensemble — analytic specification

This note is the analytic prerequisite to the first experiment in the
project that can _eliminate_ candidate ensemble coordinates rather than
generate further observations. It depends on `dev/support_ensembles.md`
for vocabulary (`E`, `Q(R, E ; φ)`, the §5 candidate list) and is
intended as the design document for one specific class of ensemble
trajectories: those that vary pairwise support overlap while leaving
expected column marginals invariant.

It deliberately does **not** declare which ensemble coordinate is the
controlling one. The construction is shaped to _let the experiment
identify it_, not to encode the answer in the design.

---

## 1. Why a trajectory rather than a point

Phase 2 sampled four points in ensemble space (the four named
profiles). Phase 2's findings can be re-read as:
"these four points lie in different regimes". But four points in a
high-dimensional space cannot, on their own, identify which axis of
that space the regimes are arrayed along. Multiple candidates from
`support_ensembles.md` §5 simultaneously change between
`dense_uniform` and `dense_diverse` (mean overlap, column coverage
entropy, support rank, mutual information all move together), so any
of them is a possible explanation and none can be falsified.

The remedy is to construct a **trajectory** through ensemble space:

```
t ∈ [0, 1]   ⟼   E_t ∈ ensemble space
```

along which **one** §5 candidate moves smoothly while the others are
held fixed (or at least decoupled). Then for each diagnostic `φ` the
function

```
ρ_φ(R, t) := Q(R, E_t ; φ)        ("response curve")
```

is observable, and the experimental question becomes: which §5
quantity does `ρ_φ(R, t)` track? That is the question the rest of
this note prepares the ground for.

The conceptual upgrade is from **point sampling** of named workloads
to **trajectory geometry** on ensemble space. Trajectories carry
derivatives, sensitivities, low-perturbation slopes, curvature,
interior extrema, and threshold structure — the primitives for
theory-building. Workloads carry none of those. From this point on
the project's primary experimental object is the trajectory family,
not the workload list.

I want to flag that even within this note, the _trajectory_ is the
primary object. `μ_J` happens to be the candidate I think the path
tests most directly, but the methodology — controlled paths,
response curves, "moves first" — is meant to survive any rejection
of `μ_J` as the controlling quantity. If the experiment falsifies
`μ_J`, the reply is "construct a path that decouples the next
candidate", not "the methodology was wrong".

---

## 2. Construction: the anchor-mix family `E_t`

The construction has two random ingredients: an _anchor permutation_
`σ ∈ S_n` and per-row _coupling coins_ `C_i ∈ {0, 1}`. A sample
`S ~ E_t` is produced as follows:

1. Sample `σ ~ Uniform(S_n)`.
2. For each row `i ∈ [n]` independently, sample `C_i ~ Bernoulli(t)`:
   - If `C_i = 1`: set `π_i := σ`. The row's support is the
     **anchor support** `S^σ := σ⁻¹([0, d))`.
   - If `C_i = 0`: sample `π_i ~ Uniform(S_n)` independently. The
     row's support is the **independent support**
     `S_i^0 := π_i⁻¹([0, d))`.
3. Stack rows: `S_i = π_i⁻¹([0, d))` for each `i`.

Equivalent description, slightly cleaner: every row's support is a
uniformly random size-`d` subset of `[n]`, but with probability `t` the
row "agrees with the anchor" (uses the same permutation `σ`).

### 2.1 Boundary cases

- `t = 0`: all rows are independent uniform size-`d` subsets.
  Reproduces `dense_diverse` exactly.
- `t = 1`: all rows equal `S^σ`, a single uniform random size-`d`
  subset shared across the matrix. Reproduces `dense_uniform`
  _up to a uniform random permutation of column space_ — which is
  exactly what `dense_uniform` is when its arbitrary "support is
  `[0, d)`" choice is replaced with the symmetry-respecting "support
  is a uniform random size-`d` set".

The boundary recovery is the first sanity check on the construction.

### 2.2 Why this construction is good for the analysis

Three properties make it cleaner than other interpolations:

1. **Per-row independence given the anchor.** Conditional on `σ`,
   the rows are independent of each other, which makes pair-statistics
   factorise.
2. **The anchor `σ` is itself uniform.** Marginalising over `σ`
   restores all column-space symmetries, which keeps expected column
   marginals constant in `t`. (See §3.2.)
3. **Closed-form pair-overlap.** The mixture has only three pair
   states (both anchor, one anchor, neither anchor), each with a
   tractable conditional expectation. (See §3.1.)

---

## 3. Closed forms for ensemble coordinates

The §5 candidate list from `support_ensembles.md` becomes, on this
family, a set of analytic functions of `t`. I derive them in turn.

### 3.0 Two classes of ensemble coordinates

Before the derivations, one structural distinction is worth flagging.
Some ensemble coordinates are **expectation-level**: they are
properties of the measure `E_t` itself, defined by averaging over the
randomness in `S ~ E_t`. Examples on this family:

- `O_J(E_t) = E[|S_i ∩ S_j|]/d` (an expectation over row pairs),
- `E[c(b)] = d/n` (the expected column-coverage profile),
- `E[H_col(E_t)] = log n`,
- `I(S_i ; S_j)` (a divergence between joint and product
  of marginals, defined on the measure).

Other coordinates are **sample-level**: they are functionals of a
single realised matrix `S`, and the trajectory acts on them through
their distribution rather than through their mean.

- `D_col^sample(S)` (per-sample column-coverage discrepancy),
- `rank_{GF(2)}(S)` (per-sample matrix rank),
- the per-sample column-coverage profile `c(·)` itself
  (a step function whose level-spacing depends on the sample, not on `E_t`).

Crucially, the two classes can disagree on `t`-dependence. §3.2 will
show `E[c(b)] = d/n` for all `t` (flat at the expectation level),
while §3.3 shows per-sample `D_col` rises linearly in `t`
(non-flat at the sample level). This is not a contradiction: a
distribution can have constant mean and `t`-dependent fluctuations.

The structural relevance of the distinction is that **a router's
diagnostic might depend on the sample, not on the expectation** — in
which case its response curve will track sample-level coordinates
even on a trajectory along which the expectation-level coordinates
are constant. The experiment in §5 measures both classes per sample
so the response-curve fit can decide which class each diagnostic
keys off.

Naming this distinction more precisely is deliberately deferred. The
present note keeps "expectation-level" and "sample-level" as
descriptive labels and avoids importing terminology from adjacent
fields (statistical mechanics, learning theory) until the experiment
shows the distinction is operationally load-bearing.

### 3.1 Pairwise overlap

For any pair of rows `i ≠ j`, condition on the joint state of
`(C_i, C_j)`:

| state           | probability | conditional `E[\|S_i ∩ S_j\|]` |
| --------------- | ----------- | ------------------------------ |
| `(1, 1)`        | `t²`        | `d` (identical supports)       |
| `(1, 0), (0,1)` | `2t(1-t)`   | `d²/n` (anchor vs. iid)        |
| `(0, 0)`        | `(1-t)²`    | `d²/n` (iid vs. iid)           |

Both `(1,0)` and `(0,0)` cases reduce to "a fixed size-`d` subset
intersected with an independent uniform size-`d` subset", whose
expected intersection is `d²/n` exactly. Combining:

```
E_{S~E_t}[ |S_i ∩ S_j| ] = t² · d + (1 - t²) · d²/n.
```

Define the **size-normalised pairwise overlap coefficient**

```
O_J(E) := E|S_i ∩ S_j| / d.
```

Then on the family,

```
O_J(E_t) = t² + (1 - t²) · (d/n).             (★)
```

This is the headline closed form. It is exact (no large-`n`
approximation), monotonic in `t`, and recovers the boundaries:
`O_J(E_0) = d/n`, `O_J(E_1) = 1`.

For comparison with `dev/support_ensembles.md`, the (linearised)
mean Jaccard is

```
μ̃_J(E_t) := E|S_i ∩ S_j| / E|S_i ∪ S_j|
          = [ t²·d + (1-t²)·d²/n ] / [ 2d - t²·d - (1-t²)·d²/n ]
```

which is monotone in `O_J` and so equally good for ranking ensembles.
I prefer `(★)` for analytic work because it is pure quadratic in `t`.

### 3.2 Column-coverage marginals

Define the column-coverage profile `c(b) := |{i : b ∈ S_i}| / n`.
The _ensemble-expected_ coverage is

```
E_{S~E_t}[ c(b) ]
   = t · P(b ∈ S^σ)  +  (1-t) · P(b ∈ S_i^0)
   = t · (d/n)        +  (1-t) · (d/n)
   = d / n.
```

The first term equals `d/n` because `σ` is uniform on `S_n`, so
`σ(b) ∈ [0, d)` with probability `d/n` regardless of `b`. The second
term is the trivial uniform-subset marginal.

So **expected column marginals are uniform `d/n` for all `t`**, and
therefore the expected column-coverage entropy is

```
E[ H_col(E_t) ] = log n     for all  t ∈ [0, 1].             (♦)
```

This is the design feature: the family decouples `μ_J` from `H_col`
in expectation, which is exactly what the §5 catalogue asked for.

### 3.3 Per-sample (vs ensemble-expected) coverage

A subtle point that the experiment must respect: while
`E[c(b)] = d/n` for all `t`, _per-sample_ column-coverage
discrepancy is **not** constant in `t`.

Conditional on a fixed `σ` and on `n_a := #{i : C_i = 1}`
(the number of anchor-coupled rows), the empirical column-coverage on
a single sample is

- on `b ∈ S^σ` (a `d`-subset): `c(b) ≈ n_a/n + (1 - n_a/n)·(d/n)`,
- on `b ∉ S^σ`: `c(b) ≈ (1 - n_a/n)·(d/n)`.

Since `n_a / n → t` as `n → ∞`, the per-sample column-coverage is a
two-level step function with levels approximately `t + (1-t)d/n` and
`(1-t)d/n`. Per-sample discrepancy is therefore

```
D_col^sample(E_t) ≈ t · (1 - d/n)             (large n, leading order).
```

So `D_col^sample` rises **linearly** in `t`, while `E[c(b)]` is flat.
This is significant because:

> Two ensemble statistics that agree on every average disagree
> per-sample. Diagnostics that read per-sample structure
> (e.g. `lf%`) will respond differently from diagnostics that read
> ensemble averages.

The experiment should therefore measure both. Section 5.2 returns
to this.

### 3.4 GF(2) effective support rank

The number of anchor-coupled rows is `n_a ~ Binomial(n, t)`; with
high probability, `n_a = tn + O(√n)`. Anchor-coupled rows all carry
the same support `S^σ`, contributing rank `1` jointly. The remaining
`n - n_a` rows are i.i.d. uniform size-`d` subsets, which (for
`d` not too small) span a generic `(n - n_a)`-dimensional subspace
with high probability. Therefore

```
E[ rank_{GF(2)}(S(E_t)) ] = (1 - t)·n + 1 + o(n).             (♥)
```

So rank decays linearly in `t`. This contrasts with `μ_J` and column
discrepancy, both of which are quadratic or linear in `t`. If the
experiment finds that response curves track rank rather than `μ_J`,
the difference is detectable: rank is concave-down in `t` (linear),
`μ_J` is convex (quadratic). The shape of the response curve, not
just its monotonicity, is informative.

### 3.5 Pairwise overlap variance

The variance `σ²_J(E_t) := Var_{i ≠ j}[ J(S_i, S_j) ]` is harder.
The dominant source of variance is the joint state `(C_i, C_j)` which
takes three distinct values producing three distinct conditional
intersections; the variance is approximately

```
σ²_J(E_t) ≈ t²(1 - t²)·(1 - d/n)²        (leading order in n).
```

This is non-monotone in `t`: zero at both `t = 0` and `t = 1`, peaked
near `t = 1/√2`. Useful: if a diagnostic responds non-monotonically
to `t` with a peak in the interior, the controlling coordinate is
`σ²_J`, not `μ_J`. This is one of the cleanest hypothesis-distinguishers
the construction admits.

### 3.6 Mutual information between row pairs

Under `E_t`, the joint distribution of `(S_i, S_j)` decomposes as

```
P(S_i, S_j) = t² · P_anchor(S_i) δ(S_i = S_j)
            + (1 - t²) · P_iid(S_i) · P_iid(S_j).
```

The mutual information satisfies

```
I(S_i ; S_j) = H(S_i) - H(S_i | S_j)
```

with `H(S_i) = log C(n, d)` (entropy of a uniform size-`d` subset)
and `H(S_i | S_j)` reduced by the `t²` mass that pins `S_i = S_j`.
A clean closed form is awkward, but the leading behaviour is

```
I(S_i ; S_j) ≈ t² · log C(n, d).
```

So mutual information is _quadratic_ in `t`, like `μ_J`, and the two
are not separable on this single trajectory. Distinguishing them
requires a second trajectory (e.g. one that varies `μ_J` linearly
while keeping `I` constant); that is left for a follow-up note.

### 3.7 Summary table

| coordinate                 | symbol on `E_t`     | functional form        |
| -------------------------- | ------------------- | ---------------------- |
| size-normalised overlap    | `O_J(E_t)`          | `t² + (1 - t²)·d/n`    |
| expected column entropy    | `E[H_col(E_t)]`     | `log n` (constant)     |
| per-sample col discrepancy | `D_col^sample(E_t)` | `≈ t·(1 - d/n)`        |
| effective GF(2) rank       | `E[rank(E_t)]`      | `(1-t)·n + O(1)`       |
| pairwise overlap variance  | `σ²_J(E_t)`         | `≈ t²(1-t²)(1 - d/n)²` |
| pairwise mutual info       | `I(S_i ; S_j)`      | `≈ t² · log C(n, d)`   |

The functional shapes are all distinct except for the `μ_J` / `I` pair.
The shapes — quadratic, linear, peaked, etc. — are what the
experiment will fit response curves against.

---

## 4. Sample complexity

The estimators that the experiment will need:

- **`O_J(E_t)`:** Sample `m` random row pairs, compute mean
  intersection. Variance `O(d²/m)`. To resolve `O_J` to precision `ε`
  takes `m = O(d²/ε²)` pairs, which is cheap.
- **`H_col(E_t)`** (per-sample): one-pass histogram over `n`
  columns, `O(n·d)` time. The estimator from one sample has bias
  `O(1/n)`, well below the signal we are trying to read.
- **`D_col^sample(E_t)`:** same as the histogram; `D = max_B …` is
  computed by sorting the per-column coverage and bounding excursions.
  `O(n log n)` per sample.
- **`σ²_J(E_t)`:** needs `O(d⁴/ε²)` pairs for `ε`-precision because
  it is a fourth moment. Larger but still manageable at `n = 1024`.
- **`rank(S(E_t))`:** `O(n³)` Gaussian elimination over `GF(2)`,
  exact. Cheap at `n ≤ 4096`.

Across `m_t` values of `t` and `m_seed` per-`t` seeds, the total cost
of the experiment is dominated by router invocations, not ensemble
estimation. So we can afford to compute _all_ §5 candidates on every
sample and let the response-curve fit decide which coordinate matters.

---

## 5. Response curves

### 5.1 The basic object

For each kernel `R` and each diagnostic `φ`, the experiment yields a
response curve

```
ρ_φ(R, t) := Q(R, E_t ; φ)        (estimated by averaging φ over m_seed
                                   samples per t over m_t values of t)
```

The headline product of the experiment is the family of curves
`{ρ_φ(R, ·) : R ∈ {hash, phase_router, rs_add, rs_aff, rs_hyb},
              φ ∈ {survival, CV, lf%, validity, autocorr}}`.

### 5.2 Methodology: "which observable moves first?"

The phrase "moves first" in `dev/support_ensembles.md` is operational
on this trajectory. As `t` is dialled up from `0`, we ask, for each
kernel `R`:

- Which `ρ_φ(R, t)` departs first from its `t = 0` baseline?
- At what `t` does `ρ_φ(R, t)` cross half the gap between baseline
  and saturation?
- What is the slope `∂_t ρ_φ(R, t) |_{t=0}` (low-overlap sensitivity)?

The kernel for which `ρ_survival` _never_ departs from baseline is
the one that responds to no §5 coordinate this trajectory exposes —
i.e., one that is invariant under `E_t`. The `phase_router` is
predicted to be such a kernel (see §6). The kernel for which
`ρ_survival` is monotone in `t` and shapes like `O_J(E_t)` is one
that responds _primarily_ to overlap. RS kernels are predicted to
be such kernels.

This is a strictly stronger experimental tool than the four-point
sweep, because it extracts not only "which kernels depend on the
ensemble" but _which coordinate of the ensemble each kernel
responds to most strongly_, including a quantitative slope.

### 5.3 Distinguishing functional shapes

Concretely, a response curve `ρ_φ(R, t)` predicted to track:

- `O_J(E_t) = t² + (1-t²)·d/n` is a **convex parabola in `t`** with
  inflection at `t = 0`, slope at `t=0` proportional to `0` (the
  derivative is `2t(1 - d/n)`, zero at the origin),
- `D_col^sample(E_t) ≈ t·(1 - d/n)` is **linear**,
- `rank(E_t) ≈ (1-t)n + O(1)` is **linear, decreasing**,
- `σ²_J(E_t) ≈ t²(1-t²)(1-d/n)²` is **peaked in interior at
  `t = 1/√2`**.

Fitting `ρ_φ(R, t)` to each functional form (least-squares over the
`m_t` measured points) and comparing residuals identifies which
candidate predicts the data best. Multiple candidates will be close
on a single trajectory — that's expected; a second trajectory will
break the remaining ties.

The principle: **functional shape is more identifying than monotonicity**.

---

## 6. Predictions per kernel

These are the falsifiable predictions the experiment will test. Each
is stated in shape-of-curve language so it can be confirmed or
rejected by fit.

### 6.1 `phase_router` — predicted near-flat in `t`

Hypothesis: `phase_router`'s use of the support-independent
`inv_perm_s` scrambler makes it _insensitive to ensemble overlap
structure_. The cumulative-degree walk `[offsets_s[j], + d_j)` in
physical space is driven by `(ones_s, offsets_s)` only, both of which
on this family are **identically distributed for all `t`** (every row
has degree `d`, so `ones_s[j] = d` for all `j` and all `t`).

Predicted response curve: `ρ_survival(phase_router, t) ≈ const`,
within statistical noise, for all `t ∈ [0, 1]`. Same for `ρ_CV`,
`ρ_lf%`. The `ρ_validity` curve may move because validity is by
definition support-dependent.

This is the cleanest falsifiable claim the project has produced.

### 6.2 RS kernels — predicted to track `O_J(E_t)`

Hypothesis: intrinsic-decode kernels, on per-sample input, have
candidate sets entirely determined by row supports. Sample-wise
diagnostics (CV, lf%, survival) should therefore depend on `E_t`
through the _per-sample_ support overlap structure, of which the
ensemble-expected representative is `O_J(E_t)`.

Predicted response curve: `ρ_survival(R_RS, t)` decreases monotonically
in `t`, with shape

```
ρ_survival(R_RS, t) ≈ baseline − amplitude · O_J(E_t)
                    = baseline − amplitude · (t² + (1-t²)·d/n).
```

Convex parabola in `t`, falling. Same prediction across the three RS
variants up to constants; the additive / affine / hybrid distinction
on this family is small (the family is non-local, so the hybrid's
locality preservation is not load-bearing here — see `findings_phase2.md`
on `block_local`). The overlap-controlled prediction is the test of
the claim that `O_J` is the controlling coordinate for RS kernels.

### 6.3 RS kernels alternative — `D_col^sample(E_t)` instead

If `ρ_survival(R_RS, t)` fits _linearly_ rather than quadratically,
the controlling coordinate is more likely `D_col^sample(E_t)` (which
is linear in `t`) or rank (also linear). In that case the RS kernels
respond to per-sample column-coverage rather than to row-pair overlap
directly. This is a weaker claim than the §6.2 prediction but still
informative.

Distinguishing these on a single trajectory is hard because both are
monotone, but the curvature test (§5.3) is the cleanest available
distinguisher.

### 6.4 `lf%` — predicted to follow per-sample column discrepancy

Hypothesis: `lf%` is empirically a column-coverage discrepancy
diagnostic (cf. `dev/support_ensembles.md` §6.3). On `E_t`, per-sample
column discrepancy is linear in `t`. Predicted shape:

```
ρ_lf%(R_RS, t) ≈ baseline + slope · t
ρ_lf%(phase_router, t) ≈ const.
```

Linear (not parabolic) for the RS kernels, flat for `phase_router`.
This is a more delicate prediction than the survival one and
distinguishes column-coverage from overlap as controlling coordinates,
because column-coverage is linear in `t` and overlap is quadratic.

### 6.5 Hybrid — predicted to behave like additive on this family

Because `E_t` does not have block-local structure (anchor-coupled
rows share an _arbitrary_ `σ⁻¹([0, d))` rather than a contiguous
window), the hybrid kernel's locality-preservation feature has no
substrate to act on. Predicted response curves match the additive
RS variant within a few percent.

If this is **not** observed — if hybrid systematically wins on
`E_t` — that would suggest hybrid responds to a coordinate other
than locality, which would be a non-trivial discovery worth following
up.

### 6.6 Cross-prediction summary

| diagnostic       | phase_router  | rs_add / rs_aff        | rs_hybrid         |
| ---------------- | ------------- | ---------------------- | ----------------- |
| `survival`       | flat          | `∝ -O_J(E_t)` (parab.) | ≈ rs_add          |
| `CV`             | flat          | `∝ +O_J(E_t)` (parab.) | ≈ rs_add          |
| `lf%`            | flat          | `∝ +t` (linear)        | ≈ rs_add          |
| `validity`       | possibly down | identically `1.0`      | identically `1.0` |
| `autocorr_pres.` | low           | low (no locality)      | low (no locality) |

The two parabolic-vs-linear predictions are the discriminating ones.
Validity flatness across RS kernels is structural, not informative.

---

## 7. What this experiment can — and cannot — falsify

### 7.1 It can falsify

- **"`μ_J` is the controlling coordinate for RS-kernel survival."**
  If `ρ_survival(R_RS, t)` fits the linear or rank-shaped predictions
  better than the parabolic `O_J(E_t)` shape, `μ_J` is rejected as
  the primary controlling coordinate.
- **"`phase_router` is `O_J`-invariant."** If `ρ_φ(phase_router, t)`
  varies significantly with `t`, the support-agnostic story is wrong
  and `phase_router` responds to the ensemble through some channel
  not yet identified.
- **"`lf%` reads column-coverage discrepancy."** If `ρ_lf%(R_RS, t)`
  is parabolic rather than linear, `lf%` does not directly read
  column-coverage; it reads something pair-wise.
- **"Hybrid behaves like additive on non-local ensembles."**
  Distinguishing hybrid from additive on `E_t` would falsify
  the "hybrid wins iff locality" reading.

### 7.2 It cannot falsify

- The hypothesis that `μ_J` is _one of several_ controlling
  coordinates. A single trajectory cannot separate `μ_J` from
  another quantity that varies in lockstep with it on this path.
  In particular, `O_J(E_t)` and `I(S_i; S_j)` are both quadratic in
  `t` on this family (§3.6); they are not separable here.
- The hypothesis that `phase_router` is sensitive to _some_
  ensemble coordinate, just not one this trajectory exposes.
  A different trajectory (e.g. one that varies degree heterogeneity
  with overlap fixed) would be required.
- Anything about non-exchangeable ensembles. `E_t` is exchangeable
  by construction (rows are conditionally i.i.d. given `(σ, t)`).

### 7.3 What follow-up trajectories would be needed

Roughly:

| trajectory                   | what it isolates                                  |
| ---------------------------- | ------------------------------------------------- |
| `E_t` (this note)            | overlap vs. column-coverage vs. rank              |
| degree-heterogeneity sweep   | controls for the "diversity ≠ heterogeneity" axis |
| locality-anchor sweep        | tests hybrid's locality channel directly          |
| non-exchangeable Markov path | tests whether row order carries information       |
| second-moment-only sweep     | separates `μ_J` from `σ²_J`                       |

These are listed as future-work; this note specifies only the first.

---

## 8. Implementation outline (deliberately brief)

The actual implementation belongs in a separate artefact and is held
back deliberately until the closed forms above have been reviewed.
For completeness, the minimum surface looks like:

- A new ensemble constructor `controlled_overlap(n, d, t, seed) -> Workload`
  in `src/workloads.rs` (or a new module — to be decided when writing
  it). Internally: sample `σ`, sample row coins, fill `s_bits`.
- An estimator routine that reports
  `(O_J, H_col_sample, D_col_sample, rank_GF2)` for the produced
  matrix. These are independent of router invocation and are computed
  once per sample.
- A driver `examples/controlled_overlap_probe.rs` that sweeps
  `t ∈ {0.0, 0.1, ..., 1.0}` × `m_seed` seeds × `5` routers,
  emits a long-format CSV with one row per `(t, seed, router, φ)` cell.
- A fit routine (probably in Python — `scripts/fit_response.py`)
  that fits `ρ_φ(R, t)` against each candidate functional form
  from §3.7 and reports residuals.

The cost is dominated by router calls: at `n = 1024`, `m_t = 11`,
`m_seed = 8`, `5` routers, total router invocations ≈ `440`,
well within wall-clock budget.

I am leaving the code unwritten until the analytic specification is
confirmed.

---

## 9. Forward link: why pushforward language waits

The pushforward operator `R_*` (sketched in the prior plan-mode
discussion) acts naturally on whichever coordinates of ensemble space
turn out to be _closed_ under routing. We do not yet know which §5
candidate sits in that closed set. After the controlled-overlap
experiment, we will know that

```
ρ_φ(R, t) is well-fit by    f_φ(   q( E_t )   )
```

for some specific candidate `q ∈ §5`, and the pushforward operator
`R_*` can then be defined on the corresponding low-dimensional
coordinate space without arbitrariness. Writing pushforward language
_before_ identifying `q` would result in either a generic operator
on the full bit-matrix space (analytically intractable) or a guess
about the closure coordinate (premature).

So the pipeline is:

```
support_ensembles.md  →  defines E and Q(R,E;φ)
controlled_overlap.md →  identifies q via response-curve fit
pushforward.md        →  defines R_* on coordinate(s) q
docs/paper.md         →  rewrites the project around q and R_*
```

Each artefact unblocks the next. The current document is the second
of four.

---

## Status

This note is a specification, not a result. It contains:

- a parameterised ensemble family `E_t` interpolating from
  `dense_diverse` (`t = 0`) to `dense_uniform` (`t = 1`) along a
  controlled-overlap path,
- closed-form analytic expressions for six §5 candidate coordinates
  on this family, with distinct functional shapes,
- a response-curve methodology that converts diagnostic measurements
  into per-coordinate likelihoods,
- per-kernel predictions sufficient to falsify the leading
  hypothesis ("`μ_J` is the controlling coordinate for RS-kernel
  survival") if it is wrong,
- an explicit accounting of what the experiment cannot decide.

The next concrete step is the experiment itself: implement
`controlled_overlap(t)` in `src/workloads.rs`, run the sweep, fit
response curves, and report the result. That step is deferred to a
separate Act-mode session, on the understanding that the
specification above is the contract the experiment is testing.

The pushforward note `dev/pushforward.md` follows the experimental
result, not this specification.
