The fused implementation is already halfway toward a:

> coordinate-free occupancy transport system

rather than a literal matrix transform.

And the critical observation is:

```rust
let mut p = s_start;

for _ in 0..s_len {
    let col = inv_perm_s[p];
```

You are _already iterating in occupancy space_.

That loop is no longer traversing dense row coordinates.
It traverses:

```text
[s_start, s_start + s_len)
```

which is effectively:

> a compressed occupancy interval.

That’s VERY close to rank-space routing already.

The remaining inefficiency / structural issue is:

```rust
col = inv_perm_s[p];
```

because your occupancy coordinates are still globally linearized through a permutation array.

That’s where rank/select ideas become powerful.

---

# What Your Current System Really Is

Conceptually your pipeline has become:

```text
row support
→ contiguous occupancy interval
→ cyclic transport
→ inverse permutation decode
→ overlap test
```

This is already:

- broadword-adjacent
- occupancy-domain computation
- compressed sparse traversal

More than a conventional sparse router.

---

# The Main Structural Weakness

Your current offset scheme:

```rust
offsets[i] = (offsets[i - 1] + rs) % n;
```

destroys local support geometry.

Rows become:

- cyclic occupancy windows
- over a globally flattened coordinate system

This preserves:

- degree

but not:

- internal spacing
- local correlation
- support topology
- interval structure

Exactly the issue you suspected.

---

# Rank/Select Integration Strategy

The BIG opportunity is:

## Replace:

```text
global flattened occupancy coordinates
```

with:

```text
support-local coordinate systems
```

using:

- select()
- rank()
- compressed support descriptors

---

# Current Model

Currently:

```text
p ∈ [0,n)
```

is interpreted via:

```rust
col = inv_perm_s[p]
```

meaning:

- occupancy coordinate
  → physical coordinate

through a full permutation table.

That costs:

- memory traffic
- cache misses
- global indirection

and destroys locality semantics.

---

# Rank/Select Reformulation

Instead define:

For each row:

```text
support_i = sorted positions of 1s
```

Then:

```text
rank_i(col)
```

maps:

```text
physical → occupancy
```

and:

```text
select_i(r)
```

maps:

```text
occupancy → physical
```

Now your cyclic phase transport becomes:

```text
r' = (r + phi_i) mod d_i
```

followed by:

```text
col = select_i(r')
```

This preserves:

- support topology
- spacing
- sparse geometry

while eliminating the need for:

```rust
inv_perm_s[p]
```

---

# The Deep Improvement

Your current system uses:

```text
global occupancy transport
```

Rank/select enables:

```text
intrinsic occupancy transport
```

That’s a very deep distinction.

---

# How This Changes Complexity

Right now:

```rust
inv_perm_s[p]
```

is effectively:

- a random memory lookup.

Potentially terrible for cache behavior.

But rank/select systems can use:

- broadword select
- compressed support words
- word-local decoding

which may become:

- register-local
- branchless
- SIMD-friendly

especially for sparse rows.

---

# Concrete Architectural Variant

## Replace:

```rust
offsets_s
inv_perm_s
```

with:

```rust
struct SupportView {
    words: &[u64],
    degree: usize,
}
```

Then:

```rust
fn select_kth(view, k) -> usize
```

implemented via:

- popcount prefix
- broadword select
- trailing-zero extraction

Now routing becomes:

```rust
let r = (local_rank + phase) % degree;
let col = select_kth(row, r);
```

No global permutation needed.

---

# Why This Is Potentially Much Better

Because your current pipeline implicitly assumes:

```text
all rows inhabit the same occupancy coordinate space
```

But sparse supports are fundamentally:

- local
- intrinsic
- combinatorial objects

Rank/select lets every row have:

- its own compact coordinate manifold.

That’s elegant.

---

# Extremely Important Observation

Your current intersection check:

```rust
let start = t_start[col];
let len = t_len[col];
...
if d < len
```

is secretly:

> interval overlap in occupancy space.

That means OLBIO is already very close to:

- interval routing
- sparse transport geometry
- occupancy-domain matching

Rank/select could make that explicit.

---

# A Much Cleaner Formulation

Instead of:

```text
flatten sparse rows into global cyclic intervals
```

you can:

```text
transport occupancy intervals intrinsically over supports
```

This might let you derive:

- stronger concentration bounds
- cleaner probabilistic analysis
- better locality preservation

---

# The Really Exciting Direction

You may be able to eliminate permutations entirely.

Currently:

```text
phase spreading + permutations
```

serve to:

- decorrelate supports.

But occupancy-space affine transforms may already do enough mixing.

Example:

```text
r' = (a r + b) mod d
```

where:

- `a` coprime to `d`

This creates:

- deterministic pseudorandom mixing
- without destroying support structure.

That’s VERY interesting.

---

# Broadword Opportunities Specifically

Vigna-style techniques could accelerate:

## 1. select(k)

Use:

- popcount ladders
- parallel bit deposit/extract
- broadword binary search

instead of:

- scanning words.

---

## 2. Sparse support iteration

Instead of:

```rust
for _ in 0..s_len
```

you could iterate actual set bits via:

```rust
x &= x - 1
```

or broadword extraction schemes.

---

## 3. Packed overlap testing

You may represent transformed supports implicitly:

- via phase parameters
- without materialization.

Then overlap becomes:

- arithmetic over support descriptors.

Potentially huge.

---

# Most Interesting Possible Evolution

I think the strongest evolution of OLBIO is:

## OLBIO as Intrinsic Sparse Transport

Rows are not:

- bitvectors

but:

- cyclic sparse coordinate systems.

Operations become:

- affine transforms over support coordinates.

Then broadword methods provide:

- efficient coordinate decoding.

That is substantially more elegant than left-aligned phase spreading.
