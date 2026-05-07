# Rank/Select Integration — Progress Notes

This file is the running log of the rank/select integration outlined in
`dev/plan.md`. Status as of the first implementation pass:

## Summary

A second router kernel, `phase_router_rs`, lives alongside the original
`phase_router` and is gated behind the `rank-select` Cargo feature. It
replaces the **source-side** `inv_perm_s[p]` global indirection with an
intrinsic `select1` decode in occupancy space. The target-side cyclic
overlap test is preserved, so the two kernels share workload semantics and
can be compared head-to-head with no rewiring of benchmarks.

Two transport variants are available:

- **`phase_router_rs`** — additive shift `r' = (r + φ_j) mod d_j`
- **`phase_router_rs_affine`** — affine `r' = (a_j·r + b_j) mod d_j`,
  with `gcd(a_j, d_j) = 1` (eliminates the
  conceptual role of the global source
  permutation entirely).

Both derive `(φ_j, a_j, b_j)` deterministically from `(seed, j, col_perm_s[j])`
via SplitMix64, so callers who already supply a source permutation still see
distinct routes vs. callers who don't, while callers who pass `col_perm_s = identity`
get a clean "permutation-free" experiment.

## What was built (Milestones 0–5)

| Milestone                           | Status | Artefact                                                                                                        |
| ----------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------- |
| 0 Property tests for current kernel | ✓      | `tests/invariants.rs` (`phase_router_*` tests)                                                                  |
| 1 Rank/select primitives            | ✓      | `src/bitsupport.rs` (`rank1`, `select1`, `select_in_word`, `SupportView`, `mix64`, `gcd`, `coprime_multiplier`) |
| 2 Hybrid rank/select router         | ✓      | `src/router_rs.rs::phase_router_rs`                                                                             |
| 3 Quality probe                     | ✓      | `examples/quality_probe.rs`, `examples/moe_bench_rs.rs`                                                         |
| 4 Broadword `select1`               | ✓      | BMI2 `PDEP` fast path + portable `tzcnt` fallback in `bitsupport.rs`, runtime-detected and cached               |
| 5 Affine rank-space transport       | ✓      | `src/router_rs.rs::phase_router_rs_affine`                                                                      |

`select_in_word` automatically uses BMI2 `PDEP` on x86_64 hosts that
report it via CPUID, with the result cached in a relaxed atomic (one
load + branch on the hot path). Compile with
`RUSTFLAGS="-C target-cpu=native"` to additionally let the compiler use
BMI2 for surrounding scalar code.

## Build / run

```bash
# Default build (only the original kernel, public API unchanged)
cargo build --release
cargo test

# With the rank/select kernel enabled
cargo build --release --features rank-select
cargo test --features rank-select

# Head-to-head quality comparison
cargo run --release --features rank-select --example quality_probe

# MoE survival sweeps (4 methods × 3 experiments, writes moe_results_rs.csv)
cargo run --release --features rank-select --example moe_bench_rs
```

The Python bindings in `src/python.rs` are intentionally **not** wired to
the new kernel yet — Python users continue to call `phase_router_rs` (the
Python module) which still maps to the original Rust `phase_router`. We can
expose `phase_router_rs` to Python once the new kernel's quality numbers
are decided.

## What to look for in the quality_probe output

The original `phase_router` is the gold standard for these metrics:

| Metric                  | Goal                                                               |
| ----------------------- | ------------------------------------------------------------------ |
| `mean` per-target load  | should equal `n*k/n = k` for all methods                           |
| `CV` (std/mean)         | lower = better balance; hash ≈ 1.0, phase_router ≈ 0.3–0.4 typical |
| `max/μ` (peak hot-spot) | lower = no overloaded experts                                      |
| `corr(L, c)`            | capacity-aligned routers approach +1.0; hash ≈ 0                   |
| survival @ 1.2×         | the headline MoE number; phase_router ≈ 0.90                       |

Hypotheses for the rank/select kernels:

1. **Additive variant** should match `phase_router` on `corr(L, c)` because
   the target side is unchanged. It may have _worse_ CV than the global-perm
   kernel because each row has only an additive shift in its own rank space
   (less mixing) — but it should preserve row-local support topology, which
   the global kernel destroys.
2. **Affine variant** is the most interesting: per-row affine mixers replace
   the global source permutation entirely. Expected: comparable or better
   CV than additive, similar `corr(L, c)`, similar survival. If true, this
   is the key empirical evidence that _global permutations are not necessary_.
3. **Timing**: rank/select avoids the `inv_perm_s` lookup but pays for
   `select1` per occupancy index. On x86_64 with BMI2, `select1` is
   word-local + ~3 cycles per call. At small `n` (≤ 1024) phase_router likely
   wins on raw throughput; at larger `n` (≥ 8192) where `inv_perm_s` falls
   out of L1, rank/select should pull ahead.

If the data contradicts (1) or (2), the most likely cause is that the
per-row phase derivation `mix64(seed XOR j XOR col_perm_s[j])` is not mixing
strongly enough. A drop-in stronger mixer (e.g. `wyhash`-style) can be
swapped in `bitsupport::mix64`.

## Invariants enforced by tests

`cargo test --features rank-select` runs:

**Always (both kernels):**

- determinism (same input → same output)
- distinct seeds → distinct outputs
- fan-out bound: each row ≤ k assignments, columns in `[0, n)`
- dedup: no column appears twice in the same row
- capacity-load Pearson correlation > threshold

**Rank/select specific:**

- every emitted column is a 1-bit in the _original_ source row support
  (not just in some transformed proxy)
- additive ≠ affine on a non-trivial workload
- rank/select ≠ phase_router on a non-trivial workload (sanity check that
  the two kernels are actually doing different things)

**Bitsupport primitives:**

- `rank1(select1(x, k)) == k` for random multi-word bitvectors
- `select1` strictly increasing in `k`
- portable and BMI2 paths agree on every bit
- `coprime_multiplier(d, seed)` returns `a` with `gcd(a, d) = 1`,
  deterministic given `seed`

## Empirical findings — first pass

Running `examples/moe_bench_rs` (Phase 1 build) on the canonical MoE workload
(every source row has identical contiguous support `[0, d)`, target density
∝ relative capacity, strong heterogeneity) produced a clear collapse for the
intrinsic-decode kernels:

- `phase_router_rs` (additive) and `phase_router_rs_affine` produced
  _substantially worse_ survival, higher CV, lower `corr(L, c)`, and a
  flatter Fourier spectrum than the original `phase_router`.
- The two RS variants were nearly indistinguishable from each other —
  affine did not improve over additive, despite eliminating the global
  permutation entirely.

This was at first interpreted as a tuning failure (insufficient mixing in
the per-row phase). The mechanism turned out to be _structural_, not
parametric.

### Structural-impossibility argument

> On any workload where two source rows share the same support (i.e.
> `support_i == support_j` as bit-sets), every intrinsic-decode kernel
> visits the same per-row candidate set:
>
> ```
> support_i == support_j
>   ⇒  ∀r:  select1(support_i, r) == select1(support_j, r)
>   ⇒  candidates(i) == candidates(j) as sets.
> ```
>
> Reservoir sampling (Vitter's Algorithm R) has the property that its
> output distribution depends only on the input _set_ and the per-row RNG;
> it is _order-independent_. So the routes for rows `i` and `j` are
> distributed identically given the same per-row RNG seed, and
> approximately identically (modulo the small hash difference) under
> distinct seeds.
>
> Therefore _no choice of rank-space transport_ — additive, affine, or
> any other bijection on `[0, d)` — can break the symmetry on an
> identical-support workload. The collapse is not a bug or a tuning
> artefact; it is a property of the function class.

The original `moe_bench` workload happens to live in this regime
(`s_bits[i] = (1u64 << density) - 1` for every `i`).

### Reframing the original kernel

> _The original OLBIO kernel may fundamentally be a low-discrepancy
> distributed occupancy allocator rather than a sparse support transport
> operator._

What `phase_router` actually does on the identical-support workload is:
walk `p ∈ [offsets_s[j], offsets_s[j] + d_j) mod n` in _physical_ space,
look up `inv_perm_s[p]` (a row-indexed, support-independent col-space
scrambler), and apply the cyclic-interval test on the target side. Because
`inv_perm_s` is independent of `s_bits`, the source-side kernel is _only_
using `(offsets_s, ones_s)` to drive a per-row deterministic-but-distinct
window through col-space — i.e. cumulative-degree induced phase shifts
implementing a low-discrepancy distributed allocation.

This explains the empirical strength of `phase_router` on `moe_bench` and
explains why intrinsic-decode kernels cannot reproduce it: they replace
the entire low-discrepancy allocator with a per-row support-driven
sampler, which on identical supports degenerates to "every row picks
from the same set".

### Four-kernel ablation family

The RS work then becomes valuable precisely because it isolates which
parts of the original kernel's behaviour come from support structure vs.
global occupancy geometry. The kernels currently in tree:

| kernel                   | walks                                   | decodes via         | preserves                                 |
| ------------------------ | --------------------------------------- | ------------------- | ----------------------------------------- |
| `phase_router`           | `[offsets_s[j], + d_j)` in physical     | `inv_perm_s[p]`     | global-discrepancy / occupancy allocator  |
| `phase_router_rs` (add)  | `[0, d_j)` in rank-space + per-row φ    | `select1(row, r')`  | intrinsic support transport               |
| `phase_router_rs_affine` | `[0, d_j)` in rank-space + per-row a, b | `select1(row, r')`  | intrinsic support transport, stronger mix |
| `phase_router_rs_hybrid` | `[offsets_s[j], + d_j)` in rank-space   | `select1(row, g%d)` | occupancy walk _and_ intrinsic decode     |

The hybrid is the kernel that operationally reconciles the two
viewpoints: it preserves the cumulative-degree global walk (and hence
the low-discrepancy distributed allocation) while _also_ respecting
support topology row-by-row. On identical-support workloads it inherits
the collapse property of the other RS kernels (because `select1(row, r)`
is independent of `j` when supports are equal); on support-diverse
workloads it should track or outperform `phase_router` by tying
allocation to the actual support of each row.

### Diagnostic framework (Phase 2)

To make these regime distinctions empirically falsifiable rather than
anecdotal, Phase 2 adds:

| artefact                       | role                                                                                                                  |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `src/workloads.rs`             | four canonical profiles: `dense_uniform` / `dense_diverse` / `sparse_diverse` / `block_local`                         |
| `src/metrics.rs`               | discrepancy, naive O(n²) DFT power spectrum, low-frequency energy share, pairwise candidate Jaccard, support validity |
| `examples/spectral_probe`      | sweeps 5 methods × 4 workloads, emits `spectral_probe.csv`                                                            |
| extended `moe_bench_rs`        | sweeps 5 methods × 4 workloads × (headroom, k, scale)                                                                 |
| extended `quality_probe`       | per-workload mean/CV/peak/corr/survival comparison                                                                    |
| extended `tests/invariants.rs` | hybrid-kernel mirror tests + cross-profile fan-out / validity / dedup                                                 |

The expectation that this framework should reveal:

- `phase_router` wins on `dense_uniform` (its native regime).
- All RS kernels collapse on `dense_uniform` to comparable poor numbers
  (this is the structural-impossibility regime).
- `dense_diverse` / `sparse_diverse` / `block_local` should let the
  hybrid kernel _match_ or _beat_ `phase_router` because intrinsic decode
  starts paying off when supports actually differ.
- Pairwise candidate Jaccard should drop for the hybrid relative to
  the additive/affine variants on diverse-support workloads, signalling
  that the global walk plus intrinsic decode are jointly contributing.

## Phase 2 — empirical results (consolidated in `dev/findings_phase2.md`)

The full diagnostic sweep ran on 4 workloads × 5 methods. Headline:

- **`dense_uniform` is the only regime where `phase_router` strongly
  dominates.** Survival 0.897 vs ~0.44 for all RS kernels at headroom 1.0×.
  CV 1.05 vs ~2.30. corr(L,c) 0.88 vs ~0.44.
- On `dense_diverse`, `sparse_diverse`, `block_local`, all three RS
  kernels match `phase_router` within 1–3 pp on every metric, and a
  different RS variant is best on each of those workloads. The RS family
  also keeps `valid = 1.0` (support-preserving) where `phase_router` sits
  at ≈ 0.30.
- The hybrid collapses to the additive variant on `dense_uniform` by an
  analytic identity:
  ```
  ones_s[j] = d for all j  ⇒  offsets_s[j] = j·d
                           ⇒  offsets_s[j] mod d = 0   for all j
  ```
  i.e. the cumulative-degree phase is identically zero on
  identical-support workloads, completing the structural-impossibility
  argument mechanistically.
- The naive O(n²) DFT yields a clean separator: low-frequency energy
  share **quadruples** for the RS kernels on `dense_uniform` (~0.51 vs
  ~0.13 for `phase_router`) and is at hash-baseline parity everywhere
  else. This is the cleanest direct measurement of the
  _discrepancy-control_ mechanism that motivated the
  _low-discrepancy distributed occupancy allocator_ reframing.
- RS kernels are 5–10× slower at large N (261–581 ms vs 36–72 ms at
  N=4096, k=2). Dominated by reservoir RNG + `select1` dispatch.

The full mechanism / preservation table, the spectral signature, the
"support entropy substitutes for routing entropy" framing, and the
suggested next-step research directions are all in
`dev/findings_phase2.md`. That document is the intended interpretive
record for the paper; this file remains the engineering log.

The conceptual companion to `findings_phase2.md` is
`dev/support_ensembles.md`. It lifts the discussion from "named
workloads" to **support ensembles** (probability measures over
bit-matrices) and the decomposition `Q(R, E ; φ) = E_{S~E,σ}[φ(R(S,σ))]`.
The key conclusion: the controlling object is _relational structure
between supports_ (pairwise overlap, column coherence, discrepancy),
not per-row statistics. Per-row entropy in particular cannot
distinguish `dense_uniform` from `dense_diverse` — a result that
deliberately blocks the most natural-but-wrong abstraction. That note
is the prerequisite for any future `src/ensembles.rs` module; until a
control-variable experiment identifies the controlling scalar `q_φ`,
the trait surface is intentionally **not** committed to.

## Open follow-ups

- **Word-prefix popcount index** (`Vec<u16>` per row, or one `Vec<u32>`
  keyed by `j*nb_words + w`). For `nb_words` in the range encountered by
  the MoE benchmark (16–64), the linear word-scan inside `select1` is
  already cheap, but at `n ≥ 16384` an O(log nb_words) prefix lookup
  becomes worthwhile.
- **Vigna full broadword `select_in_word`**. The current portable
  fallback is O(k); the standard broadword version is O(1) in 6 ops. Only
  matters on hosts without BMI2 (very old x86, ARM without equivalent
  intrinsics). Note: ARM has `RBIT + CLZ` which gives a similar O(1)
  effect via `rbit(x).trailing_zeros()` after isolating the k-th bit; can
  be added behind a `#[cfg(target_arch = "aarch64")]` if needed.
- **Word-local iteration.** Right now each iteration calls `select1`
  from scratch, which re-scans words from the start. We can cache the
  active word and walk set bits via `x &= x - 1`, avoiding the
  `popcount` accumulation on hot rows.
- **Option B (intrinsic both sides)** — Phase 6 of `dev/plan.md`. Would
  require redefining the overlap test entirely. Defer until the
  hybrid version's quality numbers are settled.
- **Python bindings for the new kernel.** Trivial wrapper change once we
  decide to expose it; left out to keep the public Python surface stable.

## Decision log

- **Feature gating.** `rank-select` is **not** in default features. The
  default build path is unchanged so the existing branch can be merged
  to main without disturbing downstream consumers (the Python wheel,
  `examples/moe_bench.rs`, etc.).
- **Signature compatibility.** `phase_router_rs(_affine)` accepts the
  same `(s_bits, t_bits, n, nb_words, k, col_perm_s, col_perm_t, seed)`
  signature as `phase_router`, even though `col_perm_s` is no longer
  used to walk the source. This is intentional — it keeps benchmarks and
  call sites trivially swappable, and folds `col_perm_s[j]` into the
  per-row phase seed so callers who do supply a permutation get
  meaningfully different routes.
- **Roll-our-own select.** No external dependency on `sucds` / `bitm`.
  The bitsupport module is ~250 lines including tests and is the right
  place for future broadword optimisation work tied to this paper.
- **Tests use deterministic seeds.** No `proptest` yet. Easy to add
  later if the property surface grows.
