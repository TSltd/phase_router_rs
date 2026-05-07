# Plan: Integrating Rank/Select into the Fused OLBIO Kernel

The goal is not merely to “optimize” the current kernel.

The deeper goal is:

> replace global occupancy flattening with intrinsic support-coordinate transport.

Concretely:

Current kernel:

```text
global cyclic occupancy space
→ inverse permutation decode
→ interval overlap
```

Target kernel:

```text
intrinsic support coordinates
→ affine/phase transport in rank space
→ select() decode
→ overlap
```

This removes:

- destructive left-alignment semantics
- global permutation dependence
- random indirection lookups

while preserving:

- deterministic balancing
- bounded fan-out
- SIMD friendliness
- broadword compatibility

---

# Phase 0 — Preserve Current Behavior First

Before modifying semantics:

## Add golden behavioral tests

You need:

- deterministic reproducibility
- degree/load properties
- routing validity
- concentration guarantees

locked down first.

---

## Test Set A — Existing Invariants

### A1. Determinism

```rust
same_seed_same_routes()
different_seed_different_routes()
```

Verify:

```rust
phase_router(...) == phase_router(...)
```

---

### A2. Fan-Out Bound

For every row:

```rust
routes[row].len() <= k
```

and:

```rust
unique(routes[row])
```

---

### A3. Degree Preservation

Empirical test:

Generate random:

- source degrees
- target degrees

Measure:

```text
corr(target_degree, observed_load)
```

Should remain high.

---

### A4. Load Concentration

Measure:

```text
max_load / mean_load
variance(loads)
p95(load)
```

Compare against:

- hash routing
- current OLBIO

---

### A5. Sparse Edge Validity

Every emitted edge must satisfy:

```text
S[i,j] = 1
T[j,i] = 1
```

under transformed semantics.

This test becomes essential once transforms stop being explicit matrices.

---

# Phase 1 — Introduce Explicit Support Views

This is the key architectural change.

---

# Step 1.1 — Add Sparse Support Abstraction

Create:

```rust
struct SupportView<'a> {
    words: &'a [u64],
    degree: usize,
}
```

Purpose:

- expose select/rank semantics
- hide raw bitvector details
- enable broadword evolution later

---

# Step 1.2 — Implement Basic select()

Start naïve first.

```rust
fn select1(words: &[u64], k: usize) -> usize
```

Returns:

```text
position of kth set bit
```

Naive implementation:

- word scan
- popcount accumulation
- trailing-zero iteration

Do NOT optimize first.

Correctness first.

---

# Step 1.3 — Add rank()

```rust
fn rank1(words: &[u64], pos: usize) -> usize
```

Returns:

```text
# of set bits before pos
```

This enables:

- inverse coordinate mapping
- future occupancy-space analytics

---

# Step 1.4 — Exhaustive Correctness Tests

For random bitvectors:

```rust
for k in 0..popcount:
    pos = select1(x,k)
    assert(rank1(x,pos) == k)
```

This is critical.

---

# Phase 2 — Replace Global Occupancy Flattening

This is the real semantic transition.

---

# Current Logic

Currently:

```rust
p ∈ [s_start, s_start+s_len)
col = inv_perm_s[p]
```

This means:

- occupancy coordinates are globally linearized.

---

# New Logic

Instead:

```rust
r = local occupancy index
r' = phase transform(r)
col = select1(row_support, r')
```

No global permutation required.

---

# Step 2.1 — Local Rank Iteration

Replace:

```rust
let mut p = s_start;
for _ in 0..s_len
```

with:

```rust
for r in 0..degree
```

where:

```text
r = intrinsic occupancy coordinate
```

---

# Step 2.2 — Affine Phase Transport

Initially:

```rust
r_prime = (r + phase) % degree
```

Later experiment with:

```rust
r_prime = (a*r + b) % degree
```

where:

```text
gcd(a, degree) == 1
```

This gives:

- deterministic pseudorandomization
- invertibility
- stronger mixing

---

# Step 2.3 — Decode via select()

```rust
col = select1(row_words, r_prime)
```

Now supports retain:

- spacing
- locality
- topology

---

# Phase 3 — Redesign Target Overlap Logic

Current overlap:

```rust
interval overlap in global occupancy space
```

This must evolve.

---

# Option A (Recommended Initially)

Keep target rows represented as:

```text
cyclic occupancy intervals
```

only replacing source-side decoding.

This allows:

- partial migration
- apples-to-apples comparison

---

# Option B (Full Intrinsic Transport)

Represent BOTH sides intrinsically:

```text
source support coordinates
target support coordinates
```

Then overlap becomes:

```text
support intersection after affine transport
```

This is mathematically cleaner but larger scope.

---

# Phase 4 — Broadword Optimization Pass

Only after correctness + metrics stabilize.

---

# Step 4.1 — Broadword select()

Replace naive select with:

- popcount ladders
- broadword select
- BMI2 PDEP/PEXT where available

Potential implementations:

- Vigna broadword select
- Harley-Seal methods
- BMI2-assisted decode

---

# Step 4.2 — Word-Local Iteration

Current:

```rust
select1(..., r)
```

may rescan repeatedly.

Instead:

- cache active word
- iterate local set bits
- maintain rolling rank window

This becomes:

- nearly branchless
- cache-local
- SIMD-friendly

---

# Step 4.3 — Compressed Support Metadata

Store:

```rust
struct SupportIndex {
    word_prefix_popcounts: Vec<u16>,
}
```

This accelerates:

- select
- rank
- interval queries

---

# Phase 5 — Comparative Evaluation

This is where the project becomes scientifically interesting.

---

# Metrics to Measure

## Routing Quality

- max load
- load variance
- entropy
- collision rate
- dropped tokens
- support preservation

---

## Structural Preservation

Measure:

- pairwise distance distributions
- locality retention
- autocorrelation
- spectral characteristics

before/after transport.

This is VERY important.

---

## Systems Performance

Measure:

- memory bandwidth
- branch misses
- cache misses
- IPC
- runtime

Especially compare:

```text
inv_perm_s lookup
vs
select() decoding
```

because this is the real systems tradeoff.

---

# Phase 6 — Advanced Experiments

After stable implementation.

---

# Experiment A — Eliminate Permutations Entirely

Test:

```text
affine occupancy transforms only
```

No global permutation arrays.

Potentially:

- better locality
- lower memory traffic
- cleaner theory

---

# Experiment B — Hierarchical Supports

Represent supports:

```text
word-level blocks
→ local occupancy coordinates
```

This could become:

- extremely cache-friendly
- GPU-friendly
- SIMD-native

---

# Experiment C — Implicit Routing

Do not materialize transformed supports at all.

Represent routes parametrically:

```text
(base phase, affine coefficients, support descriptor)
```

Intersection becomes:

```text
constraint solving over occupancy coordinates
```

Potentially very powerful.

---

# Recommended Immediate Milestone

## Minimal Viable Rank/Select OLBIO

Implement ONLY:

### Replace:

```rust
col = inv_perm_s[p]
```

with:

```rust
col = select1(row_bits, r_prime)
```

using:

```rust
r_prime = (r + phase) % degree
```

Keep:

- existing target interval logic
- reservoir sampling
- parallelism
- concentration machinery

This isolates:

> whether intrinsic occupancy transport improves structure preservation and routing quality.

That is the cleanest first experiment.
