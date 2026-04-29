# **Phase Router (Rust Implementation)**

A **high-performance, deterministic load-balancing kernel** for constructing **balanced bipartite routings** using cyclic phase arithmetic.

This crate provides a Rust implementation of the **Orthogonal Load-Balanced Incidence Operator (OLBIO)**—a fast, reproducible method for distributing uneven workloads across fixed-capacity targets.

---

## What this does

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

## Intuition

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
  Fused pipeline — no bit-packed matrix materialization

- **Work proportional to active mass** — iterates only over nonzero row degrees

- **Order-independent behavior** — input ordering does not create geometric hotspots

---

## Pipeline

The implementation uses a **fully fused pipeline** that eliminates all intermediate matrices:

```
1. Precompute O(n) arrays:
   - row offsets (cumulative popcount prefix sums)
   - row ones counts (popcount per row)
   - inverse source permutation

2. For each row j (parallel):
   - Iterate only positions where S'[j,*] = 1 (cyclic interval)
   - Map positions through inverse permutation
   - Evaluate T'[j,col] via arithmetic interval test (no matrix lookup)
   - Reservoir sample up to k matches (O(k) memory)
```

### Design principles

- **No materialized S′ / T′ matrices** — all phase transforms are evaluated analytically at query time
- **O(1) per candidate** — intersection reduced to arithmetic interval checks (no bitwise scans)
- **Memory: O(n)** — only 5 small arrays, no n×n matrices
- **Embarrassingly parallel** — each row is fully independent
- **Cache-friendly** — sequential access to small precomputed arrays

---

## Crate structure

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

## Usage

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

## Benchmarks

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
| 64   | 0.037 ms    | 0.126 ms     | **3.4×** |
| 128  | 0.136 ms    | 0.312 ms     | **2.3×** |
| 256  | 0.349 ms    | 0.467 ms     | **1.3×** |
| 512  | 0.905 ms    | 1.578 ms     | **1.7×** |
| 1024 | 3.333 ms    | 6.280 ms     | **1.9×** |
| 2048 | 13.07 ms    | 27.49 ms     | **2.1×** |
| 4096 | 54.14 ms    | 123.9 ms     | **2.3×** |

(Min times shown; median times from Criterion are ~10–20% higher)

---

## 🧪 Current status

- ✅ Correctness validated (basic tests)
- ✅ Deterministic routing
- ✅ Parallel execution via Rayon
- ✅ Fully fused pipeline (no intermediate matrices)
- ✅ ~2–3× speedup over matrix-based approach
- ✅ Benchmark suite (CLI + Criterion)

---

## Optimizations applied

### Structural optimizations

- Fully fused pipeline — eliminates all n×n matrices
- Analytical phase evaluation (no S′ / T′ materialization)
- Reservoir sampling — O(k) memory, no candidate buffer

### Micro-optimizations

- Modulo-free cyclic iteration
- Branchless cyclic range check
- Precomputed T lookups
- Thread-local buffer reuse
-

### Remaining opportunities

- [ ] SIMD acceleration (`std::arch`) for hot loops
- [ ] Optional `unsafe` fast paths (bounds check elimination)
- [ ] Python bindings via `pyo3`
- [ ] Benchmark suite vs C++ implementation

---

## Target use cases

This implementation is best suited for:

- **ML inference routing** (micro-batch load balancing)
- **Distributed data partitioning**
- **Cache / shard rebalancing**
- **Bioinformatics (k-mer distribution)**
- **Graph partitioning**

---

## 🧠 Mixture-of-Experts (MoE) Routing

Phase Router is particularly well-suited to **capacity-constrained expert routing** in Mixture-of-Experts models, where tokens must be dispatched to experts with heterogeneous capacity limits.

### The problem

In MoE inference, each token is routed to _k_ experts. Experts have hard capacity limits — tokens that exceed a limit are **dropped**, wasting compute and degrading quality. Standard hash routing distributes load uniformly, ignoring capacity differences between experts. This causes low-capacity experts to overflow while high-capacity experts sit idle.

### Why Phase Router helps

The cyclic phase embedding produces assignments where **expected load on each expert is proportional to its capacity** — by construction. No post-hoc rebalancing or auxiliary loss is needed.

```
Hash:          E[load_j] = k           (uniform, ignores capacity)
Phase Router:  E[load_j] ∝ capacity_j  (capacity-aware by construction)
```

### Benchmark results

We benchmark against uniform hash routing on N=1024 experts with heterogeneous capacities (10% at 8×, 20% at 2×, rest at 1×). Metric: **token survival rate** (fraction of routed tokens not dropped).

| Scenario                           | Phase Router | Hash  | Advantage  |
| ---------------------------------- | ------------ | ----- | ---------- |
| Tight capacity (1.0× headroom)     | 87.9%        | 79.2% | **+8.7%**  |
| Practical capacity (1.2× headroom) | 89.8%        | 79.2% | **+10.5%** |
| High fan-out (k=16)                | 96.6%        | 77.6% | **+19.0%** |
| Large scale (N=4096)               | 91.1%        | 79.0% | **+12.1%** |

Key findings:

- **10–19% higher token survival** across all tested configurations
- **Advantage grows with fan-out _k_** — at k=16, Phase Router delivers 96.6% vs 77.6% (+19pp)
- **~40% less overprovisioning needed** — Phase Router hits 90% survival at 1.2× headroom; hash needs ~2×
- **Consistent across scale** — 10–14% advantage from N=256 to N=4096
- **Advantage emerges with heterogeneity** — near-identical at uniform capacity, +10.5% at strong heterogeneity

### When it matters most

The advantage is largest when:

- Expert capacities are **heterogeneous** (mixed GPU types, variable batch budgets)
- Token drops are **expensive** (require recomputation or degrade model quality)
- Overprovisioning budget is **limited** (can't afford 2× headroom)
- Fan-out _k_ is **moderate to large** (k ≥ 4, as in Switch Transformer / GShard)

> For the full benchmark methodology and results, see [`docs/comparison.md`](docs/comparison.md).

---

## When to use this

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

## Conceptual comparison

| Method           | Speed      | Balance  | Deterministic | Global State |
| ---------------- | ---------- | -------- | ------------- | ------------ |
| Hashing          | ⭐⭐⭐⭐⭐ | ⭐⭐     | ✓             | ✗            |
| Greedy           | ⭐⭐       | ⭐⭐⭐⭐ | ✓             | ✓            |
| **Phase Router** | ⭐⭐⭐⭐   | ⭐⭐⭐⭐ | ✓             | ✗            |

---

## License

MIT

---

## Contributing

Contributions are welcome, especially around:

- SIMD optimization
- real-world benchmarking
- Python/FFI bindings
- integration examples

---

## Summary

Phase Router is:

> A fast, deterministic, low-skew routing primitive built on cyclic phase arithmetic.

This Rust implementation delivers:

- **~2× speedup** via a fully fused pipeline with zero matrix materialization
- **O(n) memory** instead of O(n²)
- **Embarrassingly parallel** row-independent computation
- **Deterministic, reproducible** routing from any seed

This enables fast, repeatable routing decisions in systems where traditional hashing causes load imbalance and greedy methods are too expensive.

---
