# **Phase Router**

A **deterministic, capacity-aware routing kernel** that reduces dropped work in load-balanced systems.

> Trades microseconds of routing for milliseconds of saved compute.

Hashing assumes all targets are equal.

Real systems are not equal.

Phase Router aligns load with capacity → fewer drops.

```
Phase Router: 92.5% survival
Hash:         76.4% survival
```

Phase Router reduces dropped tokens by 10–19% in realistic MoE settings.

This crate provides a Rust implementation of a **phase-based routing algorithm** that distributes uneven workloads across fixed-capacity targets without global coordination.

---

## Why this matters

Hash routing is fast, but blind to capacity.

In capacity-constrained systems (e.g. Mixture-of-Experts):

- Overloaded targets drop work
- Dropped work wastes compute and degrades quality

Phase Router trades microseconds of routing for fewer dropped tokens.

### The tradeoff

- Hash routing: ~40× faster
- Phase Router: ~2×–3× fewer dropped tokens

In real systems:

- Routing = microseconds
- Token processing = milliseconds (GPU)

So avoiding drops is often **cheaper than routing faster**.

### Rule of thumb

Phase Router is beneficial when:

```
cost_of_dropped_work >> cost_of_routing
```

This is true in most ML inference and training pipelines.

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

Routing reduces to intersection of intervals on a circle (O(1) per candidate),
instead of scanning or materializing matrices.

This produces a **low-skew, degree-weighted routing** without solving a global optimization problem.

---

## Key properties

### Core guarantees:

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

### Systems properties:

- **No coordination required**
  Fully local, batch computation

- **Zero intermediate matrices**
  Fused pipeline — no bit-packed matrix materialization

- **Work proportional to active mass** — iterates only over nonzero row degrees (skips zeros entirely)

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
- **Highly parallel** — each row is fully independent
- **Cache-friendly** — sequential access to small precomputed arrays

---

## Crate structure

```text
src/
├── lib.rs        # public API + tests
├── core.rs       # pipeline stages (offsets, row_ones, inverse perm, cyclic range)
├── router.rs     # fused phase router (sole implementation)
└── python.rs     # PyO3 bindings (thin wrapper, no logic duplication)

benches/
└── bench.rs      # Criterion benchmarks (Phase Router vs hash, k sweep)

examples/
├── bench.rs      # Quick CLI benchmark (Phase Router vs hash timing)
└── moe_bench.rs  # MoE capacity-constrained benchmark (4 experiments)

python/
└── phase_router.py  # High-level Python API (bit-packing, analysis)

scripts/
├── demo.py       # Interactive demo with plots
└── plot_moe.py   # MoE benchmark plot generation
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

Run the quick CLI benchmark (Phase Router vs hash routing):

```bash
cargo run --release --example bench
```

Run Criterion benchmarks with statistical analysis:

```bash
cargo bench
```

The benchmarks compare **Phase Router vs uniform hash routing** across sizes (N=64–4096) and fan-out values (k=1–16).

Hash routing is faster, but Phase Router uses additional compute to align load with capacity.

At small batch sizes the overhead is modest (~1–3×), while at large scale it grows — but this cost is typically outweighed by reduced dropped work in capacity-constrained systems.

See the [MoE comparison](#-mixture-of-experts-moe-routing) for quality results.

Run the full MoE capacity-constrained benchmark:

```bash
cargo run --release --example moe_bench
```

---

## 🧪 Current status

- ✅ Correctness validated (basic tests)
- ✅ Deterministic routing
- ✅ Parallel execution via Rayon
- ✅ Fully fused pipeline (no intermediate matrices)
- ✅ Benchmark suite vs hash routing (CLI + Criterion)
- ✅ Python bindings (PyO3 + maturin)
- ✅ MoE capacity-constrained benchmarks

---

## Optimizations applied

### Structural optimizations

- Fully fused pipeline — eliminates all n×n matrices
- Analytical phase evaluation (no S′ / T′ materialization)
- Reservoir sampling — O(k) memory, no candidate buffer
- Python bindings via `pyo3` + `maturin`

### Micro-optimizations

- Modulo-free cyclic iteration
- Branchless cyclic range check
- Precomputed T lookups
- Thread-local buffer reuse

### Remaining opportunities

- SIMD acceleration (`std::arch`) for hot loops
- Optional `unsafe` fast paths (bounds check elimination)
- Benchmark suite vs C++ implementation

---

## Target use cases

This implementation is best suited for:

- **ML inference routing** (micro-batch load balancing)
- **Distributed data partitioning**
- **Cache / shard rebalancing**
- **Bioinformatics (k-mer distribution)**
- **Graph partitioning**

---

## Mixture-of-Experts (MoE) Routing

### The problem

Naive routing wastes capacity.

### The consequence

Dropped tokens = wasted compute.

### The fix

Phase Router aligns load with capacity by construction.

```
Hash:          E[load_j] = k           (uniform, ignores capacity)
Phase Router:  E[load_j] ∝ capacity_j  (capacity-aware by construction)
```

This emerges because both source mass and target capacity are embedded into the same cyclic phase space, and routing corresponds to interval intersection in that space.

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

## Python Bindings

The Rust kernel is exposed to Python via [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs). The bindings are a **thin wrapper** — zero logic duplication, GIL released during compute.

### Install

```bash
pip install maturin
maturin develop --release
```

### Quick start (high-level API)

```python
import numpy as np
import phase_router_rs

# Bit-packed source/target matrices (flat u64 arrays)
n = 1024
nb_words = (n + 63) // 64
s_bits = np.ones(n * nb_words, dtype=np.uint64) * 0xFFFFFFFFFFFFFFFF
t_bits = np.ones(n * nb_words, dtype=np.uint64) * 0xFFFFFFFFFFFFFFFF

# phase_router_auto generates permutations from seed internally
routes = phase_router_rs.phase_router_auto(s_bits, t_bits, n, k=4, seed=42)
# routes: np.ndarray shape (n, k), dtype int32
# routes[i] → up to k target indices for source i (-1 = empty)
```

### Low-level API (bring your own permutations)

```python
col_perm_s = np.random.permutation(n).astype(np.uint64)
col_perm_t = np.random.permutation(n).astype(np.uint64)

routes = phase_router_rs.phase_router(
    s_bits, t_bits, n, nb_words, k=4,
    col_perm_s, col_perm_t, seed=42,
)
```

### Performance notes

- **GIL is released** during Rust compute — Rayon parallelism works fully
- **Returns 2D array** `(n, k)` — no reshape needed
- **Use contiguous arrays**: `np.ascontiguousarray(arr, dtype=np.uint64)` to avoid silent copies
- Routing cost is amortized by avoiding dropped-token recomputation

### Demo: Phase Router vs Hash Routing

A full interactive demo compares Phase Router against uniform hash routing on 512 experts with heterogeneous capacities:

```bash
python scripts/demo.py
```

The demo produces:

1. **Load distribution stats** — mean, std, CV, max/mean for both methods
2. **Load–capacity correlation** — Phase Router targets ≈ 1.0, hash ≈ 0.0
3. **Per-expert load tables** — top/bottom experts by capacity with load alignment
4. **Token survival** after capacity enforcement — Phase Router drops fewer tokens
5. **Capacity vs Load plot** (`plots/capacity_vs_load.png`) — Phase Router load rises diagonally with capacity; hash stays flat
6. **Load/Capacity ratio plot** (`plots/load_capacity_ratio.png`) — Phase Router holds a flat ratio (balanced); hash slopes downward (overloads small experts, wastes large ones)

The high-level Python API (`python/phase_router.py`) handles bit-packing and permutation generation so users never touch u64 arrays:

```python
from phase_router import route, hash_route, compute_loads

routes = route(target_capacities=[1,1,1,8,8,2,2,1], n=8, k=4, seed=42)
loads = compute_loads(routes, n=8)
```

The demo shows that the Phase Router aligns load with capacity, while hashing ignores it.

Phase Router approximates E[load_j] ∝ capacity_j under a hard fan-out constraint (k),
which slightly compresses extreme values.

---

## License

MIT

---

## Contributing

Contributions are welcome, especially around:

- SIMD optimization
- real-world benchmarking
- integration examples

---

## Summary

Phase Router is:

> A fast, deterministic, low-skew routing primitive built on cyclic phase arithmetic which aligns load with capacity — without coordination or optimization..

This Rust implementation delivers:

- \*\*Optimized fully fused pipeline with zero matrix materialization
- **O(n) memory** instead of O(n²)
- **Highly parallel** row-independent computation
- **Deterministic, reproducible** routing from any seed

This enables fast, repeatable routing decisions in systems where traditional hashing causes load imbalance and greedy methods are too expensive.

---
