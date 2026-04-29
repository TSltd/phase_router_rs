# Phase Router vs Hash Routing: Capacity-Constrained MoE Benchmark

## Summary

We compare the **Phase Router** against **uniform hash routing** in a capacity-constrained Mixture-of-Experts (MoE) setting. The key finding:

> **Phase Router achieves 10–19% higher token survival than hash routing under tight capacity limits, with the advantage growing as capacity heterogeneity and fan-out increase.**

At 1.2× headroom with strong heterogeneous expert capacities (10% at 8×, 20% at 2×, rest at 1×):

| Method           | Token Survival | Tokens Dropped |
| ---------------- | -------------- | -------------- |
| **Phase Router** | **89.8%**      | ~209 / 2048    |
| Uniform Hash     | 79.2%          | ~425 / 2048    |

Phase Router routes **2× fewer dropped tokens** because it distributes load proportional to capacity by construction, while hash routing distributes uniformly regardless of capacity.

---

## Experimental Setup

- **N** tokens, each routed to **k** experts from a pool of **N** experts
- Each expert has a **capacity limit** (hard cap on tokens it can accept)
- Tokens exceeding an expert's capacity are **dropped**
- **Metric**: Token survival rate = assigned tokens / (N × k)

### Phase Router

Encodes source demands and target capacities as bit-packed matrices. The cyclic phase embedding creates intervals on a ring of size N, with intersection sizes proportional to source demand × target capacity. Column permutations randomize the assignment structure.

### Uniform Hash

Each token independently hashes to k experts chosen uniformly at random. Fast (O(Nk)) but capacity-blind — ignores expert capacity entirely.

### Capacity Profiles

| Profile | Description                       |
| ------- | --------------------------------- |
| Uniform | All experts equal capacity        |
| Mild    | 20% of experts have 3× capacity   |
| Strong  | 10% have 8×, 20% have 2×, rest 1× |
| Extreme | 5% have 16×, rest 1×              |

---

## Experiment 1: Headroom Sweep

**Setup**: N=1024, k=2, strong heterogeneous capacity. Sweep capacity headroom from 1.0× (exact fit) to 2.0× (double capacity).

| Headroom | Phase Router | Hash  | Δ          |
| -------- | ------------ | ----- | ---------- |
| 1.00×    | 87.9%        | 79.2% | **+8.7%**  |
| 1.10×    | 89.1%        | 79.2% | **+9.8%**  |
| 1.20×    | 89.8%        | 79.2% | **+10.5%** |
| 1.50×    | 92.4%        | 80.9% | **+11.5%** |
| 2.00×    | 97.8%        | 92.5% | **+5.3%**  |

**Key insight**: Phase Router reaches 90% survival at 1.2× headroom. Hash routing needs nearly 2× headroom to reach the same level. In production, this means **40% less overprovisioning** with Phase Router.

At extreme headroom (2×), both methods converge toward 100% — but the gap at practical operating points (1.0–1.5×) is substantial.

---

## Experiment 2: Fan-out (k) Sweep

**Setup**: N=1024, headroom=1.2×, strong heterogeneous capacity. Sweep k ∈ {1, 2, 4, 8, 16}.

| k   | Phase Router | Hash  | Δ          |
| --- | ------------ | ----- | ---------- |
| 1   | 85.4%        | 70.6% | **+14.9%** |
| 2   | 89.8%        | 79.2% | **+10.5%** |
| 4   | 91.4%        | 74.6% | **+16.8%** |
| 8   | 95.9%        | 78.9% | **+17.1%** |
| 16  | 96.6%        | 77.6% | **+19.0%** |

**Key insight**: Phase Router's advantage **grows with k**. At k=16 (common in large MoE models like Switch Transformer), Phase Router achieves 96.6% survival vs 77.6% for hash — a **19 percentage point gap**. Higher fan-out amplifies the structural benefit of capacity-proportional routing.

---

## Experiment 3: Scale Sweep

**Setup**: k=2, headroom=1.2×, strong heterogeneous capacity. Sweep N ∈ {256, 512, 1024, 2048, 4096}.

| N    | Phase Router | Hash  | Δ      | PR Time | Hash Time |
| ---- | ------------ | ----- | ------ | ------- | --------- |
| 256  | 91.6%        | 77.7% | +13.9% | 0.27 ms | 0.02 ms   |
| 512  | 92.1%        | 80.6% | +11.5% | 0.96 ms | 0.03 ms   |
| 1024 | 89.8%        | 79.2% | +10.5% | 3.86 ms | 0.09 ms   |
| 2048 | 90.1%        | 78.7% | +11.4% | 11.7 ms | 0.14 ms   |
| 4096 | 91.1%        | 79.0% | +12.1% | 34.1 ms | 0.24 ms   |

**Key insight**: The survival advantage is **consistent across scale** at 10–14%. Phase Router is ~40× slower than hash, but this is the classic quality-vs-speed tradeoff. For MoE inference where token drops require expensive recomputation or degrade model quality, the routing cost is amortized by avoiding wasted compute on dropped tokens.

---

## Experiment 4: Heterogeneity Sweep

**Setup**: N=1024, k=2, headroom=1.2×. Sweep capacity heterogeneity.

| Profile                  | Phase Router | Hash  | Δ          |
| ------------------------ | ------------ | ----- | ---------- |
| Uniform                  | 89.4%        | 89.2% | +0.2%      |
| Mild (20% @ 3×)          | 87.5%        | 78.8% | **+8.7%**  |
| Strong (10%@8× + 20%@2×) | 89.8%        | 79.2% | **+10.5%** |
| Extreme (5% @ 16×)       | 83.0%        | 74.8% | **+8.2%**  |

**Key insight**: With uniform capacity, both methods perform equally (as expected — hash is optimal when all experts are identical). Phase Router's advantage **emerges with heterogeneity** and peaks at strong heterogeneity (+10.5%). At extreme heterogeneity, the advantage is slightly smaller because the Phase Router's bit matrix saturates (can't represent >N bits per row, capping the encoded capacity ratio).

---

## Why Phase Router Wins

The advantage stems from a fundamental difference in how load is distributed:

| Property           | Phase Router                  | Hash Routing               |
| ------------------ | ----------------------------- | -------------------------- |
| Load distribution  | Proportional to capacity      | Uniform (ignores capacity) |
| Coordination       | Structured (cyclic embedding) | Independent random choices |
| Capacity awareness | Built into the routing        | None                       |
| Determinism        | Fully deterministic per seed  | Deterministic per seed     |

Hash routing sends ~k tokens to each expert regardless of capacity. Low-capacity experts overflow; high-capacity experts are underutilized. The Phase Router's cyclic phase embedding creates intersection sizes proportional to target capacity, naturally routing more tokens to experts that can handle them.

**Formally**: For target j with capacity proportional to t_j:

- Phase Router: E[load_j] ∝ t_j (by construction of the cyclic embedding)
- Hash: E[load_j] = k (uniform, regardless of t_j)

---

## When to Use Phase Router

**Use Phase Router when**:

- Expert capacities are heterogeneous (different GPU memory, different batch sizes)
- Token drops are expensive (require recomputation or degrade quality)
- Overprovisioning budget is limited
- Deterministic, reproducible routing is required
- Fan-out k is moderate to large (k ≥ 2)

**Use hash routing when**:

- All experts have equal capacity
- Latency is the primary concern (hash is ~40× faster)
- Routing quality is secondary to throughput
- The system can tolerate 10–20% token drops

---

## Reproducing These Results

```bash
# Run benchmark
cargo run --release --example moe_bench

# Generate plots
python scripts/plot_moe.py
```

Results are written to `moe_results.csv`. Plots are saved to `plots/`.

---

## Plots

See `plots/` directory:

- `moe_headroom.png` — Survival vs headroom (hero figure)
- `moe_k_sweep.png` — Survival vs fan-out k
- `moe_scale.png` — Survival vs N
- `moe_hetero.png` — Survival vs heterogeneity
- `moe_combined.png` — 2×2 summary panel
