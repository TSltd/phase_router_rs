# **Phase Router (Rust Implementation)**

A **high-performance, deterministic load-balancing kernel** for constructing **balanced bipartite routings** using cyclic phase arithmetic.

This crate provides a Rust implementation of the **Orthogonal Load-Balanced Incidence Operator (OLBIO)**—a fast, reproducible method for distributing uneven workloads across fixed-capacity targets.

---

## 🚀 What this does

Given two inputs:

- **Sources** with varying weights (rows)
- **Targets** with varying capacities (columns)

The Phase Router computes a sparse assignment:

```
O ∈ {0,1}^{N×N}
```

such that:

- Each row has **at most `k` outputs** (bounded fan-out)
- Column loads are **balanced in expectation**
- Routing is **deterministic given a seed**

---

## 🧠 Intuition

The algorithm:

1. **Left-aligns** row mass (preserving row degrees)
2. **Spreads mass across a cyclic phase space**
3. **Applies independent permutations**
4. **Intersects two transformed matrices**
5. **Extracts up to `k` connections per row**

This produces a **low-skew, degree-weighted routing** without solving a global optimization problem.

---

## ⚡ Key properties

- **Deterministic**
  Same input + seed → identical output

- **Bounded fan-out**

  ```
  ∑_j O[i,j] ≤ k
  ```

- **Load-balanced (statistically)**

  ```
  E[L_j] ∝ t_j
  ```

- **No coordination required**
  Fully local, batch computation

- **Zero intermediate matrices**
  Fused pipeline uses O(n) arithmetic — no bit-packed matrix materialization

---

## 🏗️ Pipeline

The implementation uses a **fully fused pipeline** that eliminates all intermediate matrices:

```
1. Precompute O(n) arrays:
   - row offsets (cumulative popcount prefix sums)
   - row ones counts (popcount per row)
   - inverse source permutation

2. For each row j (parallel):
   - iterate over candidate columns via cyclic range
   - O(1) arithmetic check for intersection
   - partial Fisher-Yates selection of top-k
```

### Design principles

- **No matrix materialization** — S_final, T_final, and transpose are all eliminated
- **O(1) per candidate** — cyclic range membership via arithmetic, not bit operations
- **Memory: O(n)** — only 5 small arrays, no n×n matrices
- **Embarrassingly parallel** — each row is fully independent
- **Cache-friendly** — sequential access to small precomputed arrays

---

## 📦 Crate structure

```text
src/
├── lib.rs        # public API + tests
├── bitops.rs     # bit-level utilities (fill_rotated_bits, permute, rotate)
├── core.rs       # pipeline stages (offsets, row_ones, cyclic range, legacy builders)
└── router.rs     # top-level routing (fused + legacy)

benches/
└── bench.rs      # Criterion benchmarks (fused vs legacy, per-stage)

examples/
└── bench.rs      # Quick CLI benchmark with timing table

docs/
└── optimization.md  # Optimization analysis and notes
```

---

## 🔧 Usage

```rust
use phase_router_rs::router::phase_router;

let routes = phase_router(
    &s_bits,
    &t_bits,
    n,
    nb_words,
    k,
    &col_perm_s,
    &col_perm_t,
    seed,
);
```

### Inputs

- `s_bits`, `t_bits`: bit-packed matrices (`Vec<u64>`)
- `n`: matrix size
- `nb_words`: `(n + 63) / 64`
- `k`: max outputs per row
- `col_perm_*`: column permutations
- `seed`: deterministic seed

### Output

- `Vec<i32>` of size `n * k`
- Each row contains up to `k` column indices (`-1` if empty)

---

## 📊 Benchmarks

Run the quick CLI benchmark:

```bash
cargo run --release --example bench
```

Run Criterion benchmarks with statistical analysis:

```bash
cargo bench
```

### Results (Aspire 5750, fused vs legacy matrix-based)

| N    | Fused (min) | Legacy (min) | Speedup  |
| ---- | ----------- | ------------ | -------- |
| 64   | 0.042 ms    | 0.105 ms     | **2.5×** |
| 128  | 0.080 ms    | 0.193 ms     | **2.4×** |
| 256  | 0.279 ms    | 0.477 ms     | **1.7×** |
| 512  | 0.832 ms    | 1.623 ms     | **2.0×** |
| 1024 | 3.385 ms    | 6.537 ms     | **1.9×** |
| 2048 | 13.83 ms    | 28.93 ms     | **2.1×** |
| 4096 | 60.62 ms    | 126.9 ms     | **2.1×** |

---

## 🧪 Current status

- ✅ Correctness validated (basic tests)
- ✅ Deterministic routing
- ✅ Parallel execution via Rayon
- ✅ Fully fused pipeline (no intermediate matrices)
- ✅ ~2× speedup over matrix-based approach
- ✅ Benchmark suite (CLI + Criterion)

---

## 🚀 Optimizations applied

- [x] Thread-local buffer reuse via `for_each_init` (eliminated per-row allocations)
- [x] Fused left-align + rotate (`fill_rotated_bits` — direct cyclic bit-range fill)
- [x] Parallel `T_final` construction (gather pattern)
- [x] Candidate vector reuse + partial Fisher-Yates shuffle
- [x] **Fully fused pipeline** — eliminates all n×n matrices, uses O(1) cyclic range checks

### Remaining opportunities

- [ ] SIMD acceleration (`std::arch`) for hot loops
- [ ] Optional `unsafe` fast paths (bounds check elimination)
- [ ] Python bindings via `pyo3`
- [ ] Benchmark suite vs C++ implementation

---

## 🎯 Target use cases

This implementation is best suited for:

- **ML inference routing** (micro-batch load balancing)
- **Distributed data partitioning**
- **Cache / shard rebalancing**
- **Bioinformatics (k-mer distribution)**
- **Graph partitioning**

---

## 🧠 When to use this

Use Phase Router when:

- You have **skewed workloads**
- You can process in **batches**
- You need **fast recomputation**
- You want **deterministic behavior**

Avoid when:

- You need real-time per-item decisions
- You have multi-dimensional constraints (CPU + RAM + etc.)
- You require strict optimality guarantees

---

## 📊 Conceptual comparison

| Method           | Speed      | Balance  | Deterministic | Global State |
| ---------------- | ---------- | -------- | ------------- | ------------ |
| Hashing          | ⭐⭐⭐⭐⭐ | ⭐⭐     | ✓             | ✗            |
| Greedy           | ⭐⭐       | ⭐⭐⭐⭐ | ✓             | ✓            |
| **Phase Router** | ⭐⭐⭐⭐   | ⭐⭐⭐⭐ | ✓             | ✗            |

---

## 📜 License

TBD

---

## 🤝 Contributing

Contributions are welcome, especially around:

- SIMD optimization
- real-world benchmarking
- Python/FFI bindings
- integration examples

---

## 🧾 Summary

Phase Router is:

> A fast, deterministic, low-skew routing primitive built on cyclic phase arithmetic.

This Rust implementation delivers:

- **~2× speedup** via a fully fused pipeline with zero matrix materialization
- **O(n) memory** instead of O(n²)
- **Embarrassingly parallel** row-independent computation
- **Deterministic, reproducible** routing from any seed

---
