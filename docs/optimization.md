# 🧠 Where you are right now

You’ve got:

- ✅ correct pipeline
- ✅ parallel structure (Rayon)
- ✅ clean memory layout
- ❌ still leaving performance on the table

---

# 🚀 The next 3 high-impact upgrades (in order)

These are not minor tweaks—each one can give **real speedups**.

---

# 1️⃣ Kill per-row allocations (biggest win)

Right now you’re doing things like:

```rust
let mut temp = vec![0u64; nb_words];
let mut rotated = vec![0u64; nb_words];
```

inside parallel loops.

---

## Why this hurts

- alloc/free per row
- cache misses
- allocator contention under Rayon

---

## Fix: reuse buffers per thread

### Pattern:

```rust
use rayon::prelude::*;

out.par_chunks_mut(nb_words).for_each_init(
    || {
        // thread-local buffers
        (
            vec![0u64; nb_words],
            vec![0u64; nb_words],
            vec![0u64; nb_words],
        )
    },
    |(temp, rotated, permuted), (i, dst_row)| {
        // use temp/rotated/permuted here
        // no allocation inside loop
    },
);
```

---

👉 This alone can easily give **2–4× speedup**

---

# 2️⃣ Avoid building temporary full rows when possible

Right now:

```text
left-align → temp
→ rotate → rotated
→ permute → dst
```

---

## Better idea (later optimization):

- generate bits _on the fly_ during rotation
- skip `temp` entirely

---

👉 harder, but reduces memory traffic further

---

# 3️⃣ Optimize candidate extraction (hot path)

This loop is critical:

```rust
while m != 0 {
    let b = m.trailing_zeros();
    candidates.push(...);
    m &= m - 1;
}
```

---

## Improvements:

### A. Preallocate

```rust
let mut candidates = Vec::with_capacity(64);
```

---

### B. Reuse vector (same trick as above)

Avoid realloc per row.

---

### C. Optional: stack buffer

If `k` is small:

```rust
let mut candidates = [0usize; 512]; // stack
```

---

👉 This is often a **hidden bottleneck**

---

# ⚡ 4️⃣ Parallelize `T_final` safely (important)

Right now it’s sequential.

---

## Problem:

Writing into `out[dst_row * nb_words + ...]` causes:

- write contention
- unsafe parallelization

---

## Better approach:

Flip the loop:

```text
for dst_row:
    gather contributions
```

---

👉 makes it:

- parallelizable
- cache-friendly

---
