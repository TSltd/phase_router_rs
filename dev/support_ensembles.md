# Support ensembles — a conceptual design note

This document is the theoretical companion of `dev/findings_phase2.md`.
Findings is empirical: "what the diagnostic sweep showed". This file
attempts to identify the _objects_ the experiments are actually about.
It is deliberately a design note, not a specification: no API, no traits,
no code. The point is to settle the vocabulary before any of it hardens
into a module.

The motivation is explicit. Phase 2 produced four empirical regimes that
clearly separate router behaviour, but the language of "workloads" and
"named profiles" is inadequate to express _why_ they separate. Multiple
candidate quantities (degree distribution, support entropy, pairwise
overlap, locality) are correlated within the four chosen profiles, and
the diagnostic sweep cannot, on its own, decide which of them is the
controlling parameter. This note's purpose is to make those candidate
quantities first-class so that subsequent experiments can test between
them rather than around them.

---

## 1. Why named workloads are insufficient

The four profiles in `src/workloads.rs` —
`dense_uniform`, `dense_diverse`, `sparse_diverse`, `block_local` — were
introduced as _regime separators_. That goal succeeded: the kernels split
cleanly across them. But the categories are also a trap, because they
describe what we sampled, not what made the sample matter.

A "workload" is a single bit-matrix instance `S ∈ {0,1}^{n × n}`.
A "profile" is a _sampler_ that returns one such instance under a seed.
The diagnostic sweep ran the sampler once per profile and read the
single resulting `S` into the router. So when we say

> on `dense_uniform`, RS kernels collapse,

we are conflating two distinct claims:

1. **Sample claim.** "On the particular `S` that the seed produced
   from `dense_uniform`'s sampler, RS kernels collapse." —
   trivially measured.

2. **Distributional claim.** "On _every_ `S` drawn from a measure where
   all rows share the same support, RS kernels collapse." —
   the actual scientific statement, and the one the analytic identity
   `offsets_s[j] mod d = 0` actually establishes.

The named-workloads framework doesn't distinguish these. To talk
precisely about regime structure, we need to lift the object of
discussion from `S` to the _measure that produced `S`_. That measure is
a **support ensemble**.

This is the abstraction the rest of the document develops.

---

## 2. Formal decomposition

Let:

- `S ∈ {0,1}^{n×n}` be a source bit-matrix (one row per token,
  one bit per column position). Row `i` is a _support_ — the set of
  columns the router is allowed to emit for that token.
- `E` be a probability measure over the space of bit-matrices:
  a **support ensemble**. We write `S ~ E`.
- `R` be a router: a deterministic function `R(S, T, σ) → routes`
  parameterised by the target structure `T`, a seed `σ`, and the
  source `S`. (`T` is held fixed throughout this note; the variation
  of interest is on the source side.)
- `φ` be an observable / diagnostic functional on routes:
  CV, capacity-load correlation, low-frequency energy share `lf%`,
  survival, support validity, …

Then the basic quantity the experiments have been estimating is

```
Q(R, E ; φ)  :=  E_{S ~ E, σ ~ U} [ φ(R(S, T, σ)) ]
```

with `σ` drawn uniformly to average out router randomness.

Three facts about this decomposition deserve highlighting before any
other reasoning is attempted.

1. **It is the natural decomposition.** The router is a function;
   the ensemble is a measure; the diagnostic is a real-valued
   functional. Each of the three lives in a different mathematical
   category and the experimental design follows from that separation
   without forcing.

2. **`Q(R, E ; φ)` is the only thing we have ever measured.** Survival,
   CV, lf% — every diagnostic in `findings_phase2.md` is an estimate of
   `Q(R, E ; φ)` for some `(R, E, φ)` triple, with the ensemble
   estimated by a single sample. Phase 2's headline claims are
   single-sample estimates of three-argument functions. This is fine
   for separation but inadequate for theory.

3. **The interesting structure lives in `E`.** The router family is
   small (5 entries), the diagnostics are many but each is a known
   functional, and the ensemble side carries all the structural
   diversity that the rest of the project has been responding to.
   Yet the ensemble side is the side with the least vocabulary so far.
   That asymmetry is the gap this note tries to close.

---

## 3. Which ensemble statistics matter empirically?

The Phase 2 sweep already told us — implicitly — which features of `E`
the kernels respond to. Reading the findings backwards yields the
following catalogue. This is the substantive section: the rest of the
document is in service of identifying which entry on this list is the
_controlling_ parameter.

### 3.1 Marginal degree distribution

The distribution `D := popcount(S_i)` over rows. Constant for
`dense_uniform` / `dense_diverse` / `block_local`; heavy-tailed for
`sparse_diverse`.

Empirically: the RS kernels _do not_ split between the constant-degree
diverse profiles and the heavy-tailed diverse profile. Survival on
`dense_diverse` and `sparse_diverse` is within a percentage point. So
**marginal degree variation is not the controlling parameter**. It
modulates throughput (variable-degree rows mean variable inner loop
length) but not regime.

### 3.2 Pairwise overlap structure

The distribution of `|S_i ∩ S_j|` (or its normalised form, the Jaccard
similarity `J(S_i, S_j) = |S_i ∩ S_j| / |S_i ∪ S_j|`) over distinct row
pairs `i ≠ j`. We can summarise this by its mean and variance:

```
μ_J(E) := E_{i ≠ j}[ J(S_i, S_j) ]
σ²_J(E) := Var_{i ≠ j}[ J(S_i, S_j) ]
```

This is the candidate the empirical results most strongly point at.
For `dense_uniform`, `μ_J = 1` and `σ²_J = 0`: every row pair has
maximal overlap. For `dense_diverse`, `μ_J = d / (2n − d)`
(small for `d ≪ n`) and `σ²_J > 0`. The collapse / no-collapse
boundary tracks this quantity far more cleanly than it tracks any
entropy-shaped measure.

### 3.3 Support coherence (collision structure)

A finer-grained relative of pairwise overlap. Define the
column-coverage profile

```
c(b) := |{ i : b ∈ S_i }| / n      for each column b ∈ [0, n).
```

For `dense_uniform`, `c(b) = 1` on `b < d` and `0` elsewhere — perfectly
coherent. For `dense_diverse`, `c(b) ≈ d/n` uniformly — perfectly
incoherent. The contrast captured by `c` is exactly _whether supports
collide on the same columns across rows_, which is what the RS kernels
penalise. The Phase 2 spectral signature (`lf%` quadrupling on
`dense_uniform`) is plausibly best understood as a Fourier readout of
this coherence — `c(b)` is a step function for `dense_uniform` and
near-constant for `dense_diverse`, which is exactly the kind of contrast
the low-frequency energy share resolves.

### 3.4 Exchangeability / row symmetry

`E` is _exchangeable_ if the joint distribution of supports is invariant
under permutations of the row index `i`. All four current profiles are
exchangeable. This is convenient — it means single-sample estimates of
any per-row quantity are unbiased for the ensemble — but it also means
exchangeability has not yet been _isolated_. A non-exchangeable ensemble
(e.g. supports drawn from a Markov chain over `i`) would let us test
whether the kernels respond to per-row structural information beyond
"row index is a label".

### 3.5 Support locality

The block-local profile demonstrates that the kernels can in principle
exploit per-row contiguity (`block_local` is the only workload where
the hybrid kernel is best). Locality is a _geometric_ property of the
support set within `[0, n)` and is orthogonal to overlap structure —
two rows can be jointly local-and-overlapping (e.g. `S_i = [0, d)` for
all `i`) or local-but-disjoint (e.g. `S_i = [(i·d) mod n, …)`). The
Phase 2 results on `block_local` show that **kernels which decode
through the row support directly** (`select1`-based) preserve this
locality whereas the global-permutation kernel destroys it; the
controlling quantity here is the autocorrelation
`P[(b+1) ∈ S_i | b ∈ S_i]` averaged over rows.

### 3.6 Per-row entropy — the candidate that turned out not to matter

The textbook first guess for "support diversity" is the per-row
indicator entropy

```
H_row(E) := E_i [ -d_i/n · log(d_i/n) - (1 - d_i/n) · log(1 - d_i/n) ]
```

This quantity depends only on the marginal degree distribution and is
**unable to distinguish `dense_uniform` from `dense_diverse`**. Both
have constant degree `d`, so both give the same `H_row`.

This is empirically the most important observation in this section,
and is the reason the rest of the document resists "entropy" as the
default vocabulary: it's a vocabulary that points at the wrong object.
See §4.

### 3.7 The catalogue, ranked by apparent empirical leverage

| candidate                    | distinguishes the four regimes? | empirical leverage |
| ---------------------------- | ------------------------------- | ------------------ |
| pairwise overlap `μ_J, σ²_J` | yes                             | extremely high     |
| column coherence `c(b)`      | yes                             | high               |
| support locality             | partial (only `block_local`)    | high in its regime |
| exchangeability              | not yet exercised               | unknown            |
| degree distribution          | partial (separates sparse only) | moderate           |
| per-row entropy              | no                              | very low           |

The conclusion is that the controlling object is _relational structure
between supports_, not summary statistics of individual supports. This
is a qualitatively different ontology from the one a naive
"sparse-routing" framing would suggest, and it is the most important
thing this note is trying to record.

---

## 4. Why naive entropy fails — and what it warns us about

§3.6 is short but consequential. Per-row entropy is the most natural
first guess for "support diversity", it is mathematically standard, it
has tractable closed forms, and it is _wrong_ — `dense_uniform` and
`dense_diverse` are the two regimes the kernels respond to most
differently, and per-row entropy assigns them the same value.

The right way to read this failure is not "we should compute a different
entropy". It is:

> The relevant structure is not a property of any single row.
> It is a property of how rows relate to each other.

That is a categorical shift. It pushes us out of the language of
single-distribution information theory and into the language of:

- **exchangeable arrays** (de Finetti / Aldous–Hoover) — the natural
  setting for "random matrix where row labels carry no information",
- **discrepancy theory** — quantifying how much a deterministic or
  random structure deviates from the most-spread-out arrangement,
- **coding theory** — supports as codewords, Hamming/Jaccard distances
  between codewords as the controlling quantity for decoder behaviour,
- **derandomization / pseudorandomness** — the controlling question
  there is "how much randomness does a downstream procedure actually
  extract from its input?", which is structurally identical to "how
  much support diversity does the router actually exploit?".

I emphasise this because it suggests the eventual paper does _not_
sit in the standard MoE / sparse-attention literature. It sits closer
to the theory of pseudorandom matrices, which is a much more
interesting place to be.

A practical consequence: throughout the rest of this document,
"entropy" appears only when it is the literal Shannon quantity of a
specific distribution we have a reason to compute. Generic
"diversity-flavoured" claims should be phrased in terms of
overlap or discrepancy.

---

## 5. Candidate controlling quantities

Given §3 and the resistance counselled in §4, the candidate
_controlling quantities_ — single scalars that might predict
`Q(R, E ; φ)` for a given diagnostic — are:

1. **Mean pairwise Jaccard `μ_J(E)`.** Closed form for the canonical
   profiles, directly measurable, separates `dense_uniform = 1.0` from
   `dense_diverse ≈ d/(2n−d)`. The simplest quantity that respects the
   relational character identified in §4.

2. **Pairwise Jaccard variance `σ²_J(E)`.** Distinguishes ensembles
   with the same mean overlap but different overlap geometry. Likely
   matters when the controlling object is not "how much rows agree on
   average" but "how concentrated agreement is".

3. **Column-coverage entropy / dispersion.** `H_col(E) := -∑ c(b) log c(b)`
   after normalising `c` to a distribution. Equals `log d` for
   `dense_uniform` and ≈ `log n` for `dense_diverse`. This _is_ an
   entropy, but it is an entropy of a different distribution: column
   coverage rather than row content. It is well-defined, and it
   separates the regimes.

4. **Spectral flatness of the column-coverage signal.** Closely related
   to (3); ratio of geometric mean to arithmetic mean of `|F[c]|²` over
   non-DC bins. The Phase 2 `lf%` measurement is a coarsened version
   of this. A more principled statistic would replace `lf%`.

5. **Effective support rank.** Treating the bit-matrix as `n` vectors
   in `GF(2)ⁿ`, the rank is `1` for `dense_uniform` and approaches
   `min(n, log₂ |E|)` for high-diversity ensembles. Fast to estimate
   from samples; semantically clean. May be too coarse — e.g. a
   2-sample identical-support draw has rank 1 regardless of ensemble.

6. **Mutual information between row-pairs.** `I(S_i ; S_j)` for
   `i ≠ j`, under the joint distribution induced by `E`. Vanishes when
   supports are independent (`dense_diverse`), maximal when they are
   identical (`dense_uniform`). This is the cleanest information-theoretic
   quantity that respects the relational ontology, but is the hardest
   to estimate from samples.

7. **Discrepancy of column coverage.**
   `D(E) := sup_B |c̄(B) − |B|/n|` where `c̄(B)` is the row-averaged
   coverage of column subset `B` and the sup is over some test family.
   This is the closest entry on the list to a _theorem-shaped_ object;
   it places the project into the established discrepancy literature
   directly.

The hypothesis worth recording — falsifiable, and likely the central
prediction of the eventual paper — is:

> **For each diagnostic `φ`, there exists a single scalar `q_φ(E)` of
> ensemble structure such that `Q(R, E ; φ)` depends on `E` only
> through `q_φ(E)`, up to a router-family constant.**

The candidates above are the candidates for `q_φ`. Different
diagnostics may select different controlling quantities — `lf%` may
key off (4), survival may key off (1), locality preservation may key
off something not on the list at all.

We do not yet know which entry on this list is correct. The point of
making the list is to make it possible to be wrong about it.

---

## 6. Relationship to Phase 2 findings

This section translates the Phase 2 empirical results into
ensemble-statistic language.

### 6.1 The collapse on `dense_uniform`

In ensemble language: `dense_uniform` is the support ensemble
`E_unif_d` for which all mass is on the single matrix
`S` with `S_i = [0, d)` for every `i`. So:

- `μ_J(E_unif_d) = 1` (max).
- `σ²_J(E_unif_d) = 0` (no variation).
- `H_col(E_unif_d) = log d` (degenerate compared to `log n`).
- Effective support rank `= 1`.
- Mutual information between rows is the maximum possible.

Every candidate in §5 is in its degenerate / extremal value
simultaneously, which is exactly why `dense_uniform` is a "single
extreme point" rather than a regime: it is an atom of the ensemble
space, not an open subset of it. The structural-impossibility argument
in `dev/rank_select_progress.md` says, precisely, that intrinsic-decode
kernels `R` satisfy

```
S_i = S_j     ⇒     candidates(R, i) = candidates(R, j) as sets.
```

This identity has a direct ensemble interpretation: when the support
ensemble is supported on a single point of _zero pairwise diversity_,
no rank-space transport can manufacture row diversity. The lossy
information was never present in `S` to begin with. The router cannot
recover it because the ensemble didn't supply it.

### 6.2 The recovery on `dense_diverse`

`dense_diverse` corresponds to `E_iid_d` in which every row is an i.i.d.
uniform draw from `(  [n] choose d  )`. Then:

- `μ_J(E_iid_d) ≈ d / (2n − d) ≪ 1`.
- `σ²_J(E_iid_d) > 0`.
- `H_col(E_iid_d) ≈ log n`.
- Effective support rank approaches its maximum.
- Mutual information between rows vanishes.

All candidates are simultaneously at their _non-degenerate_ values, and
the RS kernels recover. The "support entropy substitutes for routing
entropy" framing from `findings_phase2.md` is, in this language,

> when ensemble pairwise overlap drops below some threshold, the
> ensemble has already supplied enough discrepancy that explicit
> router-induced discrepancy is redundant.

This phrasing is more careful than the entropy phrasing because it
identifies _which property of the ensemble_ is doing the work.

### 6.3 The spectral signature

The lf% quadrupling on `dense_uniform` for the RS kernels is a
direct consequence of column-coverage being a step function: the
loads inherit the coverage shape, so they have strong large-scale
modes, so `lf%` is high. `phase_router`'s `inv_perm_s` scrambler
breaks that coverage step-function in physical space, producing
near-uniform loads regardless of the underlying coverage shape. So:

> lf% is reading column-coverage discrepancy through the load.
>
> RS kernels: load mirrors `c(b)`. lf% high when `c(b)` is structured.
>
> phase_router: load is a permuted view of `c(b)`. lf% low always.

Reading `lf%` as a column-coverage discrepancy diagnostic, rather than
as a generic spectral diagnostic, is the right way to think about it.

### 6.4 Why the hybrid recovers on `block_local`

The hybrid wins on `block_local` because its rank-space walk is
`r = (i + offsets_s[j] mod d) mod d`, and the locality of
`block_local` means consecutive rank-space positions decode to
consecutive _physical_ columns. So the hybrid's load profile, which
on `dense_diverse` is a randomly permuted version of `c(b)`, becomes a
_locally smoothed_ version of `c(b)` on `block_local` — small shifts
in rank space stay close in physical space. That smoothing happens to
also be what the discrepancy diagnostic is asking for.

In ensemble terms: locality in the ensemble (rows are contiguous blocks)
permits the router to spend less effort breaking locality, because the
ensemble's locality structure does some of the discrepancy work
"for free", in the same sense as §6.2.

---

## 7. Implications

The implications structure into three groups, in increasing order of
reach.

### 7.1 Within this project

The reframing finalises the central conceptual claim of the paper:

> Routers compensate for deficiencies in support-ensemble discrepancy.
> Where the ensemble is already incoherent across rows, simpler routers
> suffice; where the ensemble is degenerate (supports collide), the
> router must inject the discrepancy itself.

This is the cleanest statement of the empirical decomposition and is
what `findings_phase2.md` was groping toward when it said "support
entropy substitutes for routing entropy". The corrected language is
"ensemble overlap structure substitutes for router-induced
discrepancy", and the corrected scalar is some entry on the list in §5
rather than a Shannon entropy.

### 7.2 For the experimental programme

This note suggests a different shape for the next round of experiments
than the natural extrapolation of Phase 2 would. Concretely:

- **Do not yet add a continuous-interpolation ensemble.** The
  controlling quantity is unidentified, so a one-axis interpolation
  experiment will produce visually beautiful but interpretively muddy
  curves. First identify the axis.

- **Sweep ensembles whose `μ_J` is varied while other candidates are
  held fixed.** Concretely: ensembles with constant degree, constant
  `H_col`, and tunable pairwise overlap (e.g. by partial coupling
  between rows). If `Q(R, E ; φ)` collapses onto `μ_J` across kernels,
  hypothesis confirmed; if not, the controlling quantity is something
  finer.

- **Sweep non-exchangeable ensembles** (e.g. row-Markov chains over
  supports). This isolates whether per-row order information matters
  at all, which is currently untested.

The shape of these experiments is "vary one candidate from §5 with
others held fixed and watch which diagnostic moves". That is a
classical control-variable experimental programme, and it requires
the §5 list to exist before it can be designed. Hence: §5.

### 7.3 Beyond this project

If the controlling quantity _is_ an overlap or discrepancy statistic
of the support ensemble, then the result generalises beyond the
specific OLBIO kernel. It becomes a statement of the form:

> Any sparse-routing primitive whose decode is intrinsic to row support
> requires the input ensemble to supply pairwise decorrelation;
> any sparse-routing primitive whose decode is row-independent
> (global permutation) supplies its own.

That generalisation is testable on any sparse-routing primitive in the
literature (top-k routers, hash routers, learned routers under
fixed-support workloads), which gives the eventual paper a much wider
reach than "we built another router". This is why §4 emphasised that
the natural literature neighbours are pseudorandomness, discrepancy,
and exchangeable arrays rather than MoE engineering.

The other, more speculative reach: support-ensemble structure and
router-induced structure may be partially-substitutable resources in
the same sense that ensemble randomness and algorithmic randomness are
substitutable in derandomization. If made precise, that is a
conservation-law-shaped statement of the form

> some functional `D` of (ensemble, router) is bounded below by the
> diagnostic, and the only way to drive the diagnostic down is to
> increase one or the other.

This is too speculative to commit to in print yet. But it is the
shape of the eventual theorem worth reaching for, and it is the
reason this note exists rather than a `src/ensembles.rs` module.

---

## Status and what comes next

This document defines:

- the object **`E`** (support ensemble),
- the basic decomposition **`Q(R, E ; φ)`**,
- a catalogue of candidate ensemble statistics in §3 and §5,
- a deliberate refusal to commit to "entropy" as the controlling word,
- a falsifiable hypothesis (existence of a controlling scalar `q_φ`
  per diagnostic),
- the reinterpretation of every Phase 2 finding in this language.

It does not define an API, a trait, a module, or a probe. Those should
follow only after a control-variable experiment of the kind sketched
in §7.2 confirms or rejects a specific candidate from §5, because the
right API is "the API that exposes whichever quantity turned out to
matter", and that is currently unknown.

The next concrete artefact is therefore not code. It is one of:

1. A second design note that picks one or two candidates from §5
   and works out closed forms / estimators / sample complexity for
   them on the canonical ensembles. This is the analytic complement
   of the present note and the prerequisite for the experiment.

2. A small ensemble that varies a single §5 candidate while holding
   the others fixed — e.g. a "controlled-overlap" ensemble that
   interpolates from `dense_uniform` to `dense_diverse` along the
   `μ_J` axis with degree and column-marginal held constant. The
   experimental run on such an ensemble is what would let us
   _identify_ the controlling quantity rather than just naming
   candidates for it.

Either step is small; both eventually want to happen. The order
between them is not yet obvious. The argument for (1) first is that
analytic closed forms tell us _which_ candidate to vary; the argument
for (2) first is that the empirical signal is strong enough that even
a poorly-controlled sweep would narrow the field. In practice we will
probably do both in parallel, with the analytic work setting the
parameters of the empirical work.

The paper rewrite (`docs/paper.md`) is still deferred. The framing now
has the right _shape_ — `Q(R, E ; φ)`, ensemble-as-distribution,
overlap-as-controlling-resource — but the controlling quantity is
unidentified, and committing the manuscript to a candidate before that
identification would be premature. The rewrite waits for one
control-variable result.
