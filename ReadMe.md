# **Phase Router**

A **deterministic, capacity-factor-invariant balance primitive** for
sparse routing (Mixture-of-Experts, shard dispatch, request routing).

> **Phase routing converts balance from an optimisation objective
> into a geometric invariant.**

Switch-style top-k routers treat load balance as a soft penalty on a
global gate statistic, kept honest by an auxiliary loss and a tuned
**capacity factor `cf`**. Remove either and the gate collapses: load
CV explodes 10×, drop rate jumps to ~45 %.

Phase routing makes balance a **structural property of the
cyclic-phase intersection** — no aux loss, no capacity factor, no
gate over experts to collapse. The trade-off is a 5 – 12 %
cross-entropy cost: phase routing is **not** a better semantic
router than top-k; it is a structural **balancing substrate**.

---

## What this actually is

For an MoE layer routing `N` tokens to `E` capacity-bounded experts,
phase routing returns indices `idx ∈ {-1, 0, …, E-1}^{N×k}` such that:

- **Capacity respected:** `Σ_i 𝟙[idx[i,:] = j] ≤ C_j` for every
  expert `j`.
- **Fan-out respected:** each row holds up to `k` distinct experts.
- **Load proportional to capacity by construction:**
  `E[L_j] ∝ t_j`, with `O(N^-1/2)` discreteness fluctuations.
- **Deterministic and reproducible** from a single seed.

The selection rule is a function of `(N, E, k, base_density, seed)`
only — it does **not** consume gate logits. That is why it requires no capacity-factor or auxiliary-loss tuning (no `α`, no `cf`) and nothing
to collapse (no learned gate over experts).

---

## Headline results

### 1. Capacity-factor invariance (the strongest claim)

On a 32-expert TinyStories MoE, sweeping `cf ∈ {1.00, 1.25, 1.50, 2.00}`:

| router    | val_ce          | val_cv          | val_drop         | val_contig     | route_ms    |
| --------- | --------------- | --------------- | ---------------- | -------------- | ----------- |
| top-k     | 4.712–4.766     | 0.144 → 0.221   | 11.79 % → 0.00 % | 0.110 → 0.141  | 10.7 – 11.9 |
| **phase** | **5.286–5.287** | **0.0997 flat** | **0.58 % flat**  | **0.120 flat** | 7.3 – 10.0  |

Phase's (CE, CV, drop, contig) tuple varies by **< 0.05 %** across
the entire cf range. Top-k swings 5 – 50 %. The capacity-factor
knob is empirically inert for phase.

### 2. Anti-collapse without the aux loss

At 8 experts, dropping the auxiliary load-balance loss (`α = 0`):

| router         | val_cv    | val_drop    |
| -------------- | --------- | ----------- |
| top-k aux=0.01 | 0.106     | 8.5 %       |
| top-k aux=0    | **0.879** | **47.0 %**  |
| **phase**      | **0.036** | **0.000 %** |

Phase has no aux-loss hyperparameter to drop — the load distribution
is structural and cannot collapse without modifying the kernel itself.

### 3. Higher token survival under heterogeneous capacity (baseline)

Against capacity-aware uniform **hash** routing on `N = 1024` shards
with skewed capacities (10 % at 8×, 20 % at 2×, rest at 1×):

| scenario                  | phase | hash  | advantage    |
| ------------------------- | ----- | ----- | ------------ |
| tight (1.0× headroom)     | 87.9% | 79.2% | **+8.7 pp**  |
| practical (1.2× headroom) | 89.8% | 79.2% | **+10.5 pp** |
| high fan-out (k = 16)     | 96.6% | 77.6% | **+19.0 pp** |
| large scale (N = 4096)    | 91.1% | 79.0% | **+12.1 pp** |

Full benchmark methodology in [`docs/comparison.md`](docs/comparison.md).

---

## The honest trade-off

Phase routing is **strictly worse than top-k on validation
cross-entropy** at every scale we tested:

| scale      | top-k best CE | phase CE | gap     |
| ---------- | ------------- | -------- | ------- |
| 8 experts  | 4.845         | 5.181    | +6.9 %  |
| 32 experts | 4.712         | 5.286    | +12.2 % |

The gap **widens with `E`** because phase selection ignores gate
affinity — a token whose top-affinity expert is at index 17 might be
routed to expert 9 instead.

This isn't a bug, it's the design. **Balance and specialisation are
antagonistic objectives under sparse routing:** top-k allows semantic
concentration in the gate (specialisation) but destabilises occupancy;
phase suppresses concentration (stable occupancy) but weakens
specialisation. Phase picks balance and pays the affinity tax.

We tried the obvious hybrid (top-k for affinity, phase-style quotas,
soft fall-through). It **fails catastrophically** — 62 % drop at
32 experts, because the learned gate collapses onto `k + overflow`
effective experts and the fall-through has nowhere to land. Phase's
invariances are **structural**, not portable into hybrids that retain
a learned gate over experts. See §6 of [`docs/paper.md`](docs/paper.md).

## Conceptual distinction

> Top-k:
> balance enforced by optimization pressure
>
> Phase:
> balance enforced by geometric construction

---

## When to use

Phase routing is the right choice when **balance and reproducibility
matter more than per-token affinity**:

- **Heterogeneous-capacity shard dispatch** — mixed-GPU MoE inference;
  capacity-proportional load by construction.
- **Inference at tight `cf`** — provision exactly at the quantum
  `N · k / E` without worrying about cf interactions.
- **Reproducibility-critical pipelines** — bit-deterministic given a
  seed; no aux-loss coefficient distorting the optimisation
  objective.
- **Anti-collapse-critical pipelines** — long-running deployments
  where gate collapse would be an SLO failure. Phase cannot collapse.

Phase routing is the **wrong** choice when:

- per-token expert specialisation dominates the value (5 – 12 % CE
  cost is unacceptable);
- you can tolerate aux-loss tuning and capacity-factor tuning anyway;
- the batch shape is so small that the 2 – 3 ms / forward routing
  saving is irrelevant.

---

## Algorithm intuition

Phase routing is the cyclic-phase intersection of two bit-packed
matrices encoding per-source mass and per-target capacity:

1. **Cyclic embedding.** Row `i` is a contiguous interval
   `(φ_i, φ_i + s_i) mod N` on a ring of size `N`, with
   `φ_i = Σ_{r<i} s_r mod N`. The cyclic embedding removes
   geometric bias — every starting position is occupied with equal
   expected density across `i`.
2. **Independent column permutations** are applied to the source and
   target matrices `S → S'`, `T → T'`.
3. **Intersection** `O = S' ∧ (T')^T` produces the routed
   assignment. The cyclic mixing gives `E[L_j] ∝ t_j` by
   construction.
4. **Fused arithmetic evaluation** replaces matrix materialisation
   with a wrapping-arithmetic interval test (`d = (x − start) mod
N; keep iff d < len`). Total working memory drops from `O(N²)`
   to `O(N)`.
5. **Reservoir sampling** picks up to `k` qualifying matches per row
   in `O(k)` memory.

That cyclic-phase intersection is the geometric invariant. No
optimisation step, no gate-statistics monitoring loop, no penalty
coefficient — balance is a property of the construction.

---

## Crate structure

```text
src/
├── lib.rs        # public API + tests
├── core.rs       # pipeline stages (offsets, row_ones, inverse perm, cyclic range)
├── router.rs     # fused phase router (sole implementation)
└── python.rs     # PyO3 bindings (thin wrapper, no logic duplication)

benches/
└── bench.rs      # Criterion benchmarks (phase vs hash, k sweep)

examples/
├── bench.rs            # quick CLI benchmark
└── moe_bench.rs        # MoE capacity-constrained benchmark (4 experiments)

python/
└── phase_router.py     # high-level Python API (bit-packing, analysis)

scripts/
├── demo.py             # interactive demo with plots
└── plot_moe.py         # MoE benchmark plot generation

docs/
├── paper.md            # full write-up (cf-invariance, anti-collapse, negative result)
└── comparison.md       # hash-routing baseline methodology
```

---

## Usage

### Rust

```rust
use phase_router_rs::router::phase_router;

let routes = phase_router(
    &s_bits, &t_bits, n, nb_words, k,
    &col_perm_s, &col_perm_t, seed,
);
// routes: Vec<i32> of size n * k; -1 marks empty slots.
```

For the MoE case where every expert has equal capacity, prefer the
specialised path:

```rust
use phase_router_rs::phase_router_uniform_dispatch;

let routes = phase_router_uniform_dispatch(
    n, e, k, base_density, seed, oversample,
);
// routes: Vec<i32> of size n * k.
```

### Python

```bash
pip install maturin
maturin develop --release
```

```python
import numpy as np
import phase_router_rs

# High-level (auto-generated permutations from seed)
routes = phase_router_rs.phase_router_auto(s_bits, t_bits, n, k=4, seed=42)

# MoE uniform-dispatch fast path
routes = phase_router_rs.phase_router_uniform_dispatch(
    n=8192, e=32, k=2, base_density=0.3, seed=42, oversample=4,
)
# routes: np.ndarray shape (n, k), dtype int32, -1 = empty slot
```

The GIL is released during Rust compute, so Rayon parallelism works
fully under Python. See `python/phase_router.py` for a higher-level
wrapper that handles bit-packing and permutation generation.

---

## Benchmarks

Quick CLI benchmark (phase vs hash):

```bash
cargo run --release --example bench
```

Criterion benchmarks with statistical analysis:

```bash
cargo bench
```

MoE capacity-constrained benchmark (the table above):

```bash
cargo run --release --example moe_bench
python scripts/plot_moe.py
```

Routing wall-time at `N = 8192, E = 32, k = 2` on an A10G:

| router                     | route_time (ms / fwd) |
| -------------------------- | --------------------: |
| top-k + softmax + capacity |                  11.9 |
| **phase uniform dispatch** |               **8.5** |

Phase is ~28 % cheaper to invoke than the top-k stack at this scale.
At small batch sizes the kernel overhead is more visible (~1 – 3×
slower than raw hash), but capacity-aware behaviour reduces dropped
work by enough to dominate the trade-off.

---

## Negative result: composing top-k with phase fails

A natural-looking hybrid — top-k for affinity, phase-style per-expert
quotas, soft fall-through to the 2nd/3rd/4th choice on overflow —
fails catastrophically at 32 experts:

| metric   | top-k (cf=1.0) | phase (cf=1.0) | balanced (cf=1.0) |
| -------- | -------------: | -------------: | ----------------: |
| val_ce   |          4.766 |          5.287 |             4.787 |
| val_cv   |          0.144 |          0.100 |         **1.551** |
| val_drop |        11.79 % |         0.58 % |       **61.91 %** |

The mechanism: the learned gate collapses onto ~8 of 32 experts
(`perm_entropy = 0.61`), the `top-(k+overflow) = top-4` candidate set
lies entirely inside the popular subset, and the fall-through has
nowhere to land. `cf = 2.0` does not rescue it: drop is still
51.7 %.

Phase routing is immune because it has no gate to collapse. Phase's
invariances are **structural** — not portable into any hybrid that
retains a learned gate over experts. Full analysis in §6 of
[`docs/paper.md`](docs/paper.md).

---

## Conceptual comparison

| primitive        | balance source             | causal?    | hyperparams    | failure mode          |
| ---------------- | -------------------------- | ---------- | -------------- | --------------------- |
| top-k + aux + cf | aux loss + capacity factor | yes        | `α`, `cf`, `k` | gate collapse w/o aux |
| Expert Choice    | per-expert pick            | **no**     | `C`, `k`       | breaks AR semantics   |
| BASE layers      | linear assignment          | yes (slow) | none           | `O(N²)` per layer     |
| hash routing     | none (random)              | yes        | none           | capacity-blind        |
| **phase (ours)** | bipartite construction     | yes        | `base_density` | CE cost ≈ 5 – 12 %    |

The most relevant comparison is BASE: it also gives up a learned
gate and accepts that balance is the contribution. We retain the
gate as a routing **weight** (the softmax distribution is gathered at
the chosen indices and renormalised), so phase routing plugs into the
same training loop as top-k; we only replace the selection rule.

---

## Status

- ✅ Correctness validated (unit + property tests in `tests/invariants.rs`)
- ✅ Deterministic routing from a seed
- ✅ Parallel execution via Rayon
- ✅ Fully fused pipeline (no intermediate matrices, `O(N)` memory)
- ✅ Python bindings (PyO3 + maturin), GIL released
- ✅ MoE training experiments at 8 and 32 experts (TinyStories,
  cf-invariance reproduced)
- ✅ Negative result: greedy top-k + phase hybrid documented

---

## Further reading

- [`docs/paper.md`](docs/paper.md) — full write-up: cf-invariance,
  anti-collapse, the BalancedRouter negative result, balance ↔
  specialisation antagonism, geometric-invariant framing.
- [`docs/comparison.md`](docs/comparison.md) — hash-routing baseline
  methodology and full survival tables.
- `dev/findings_stress_sweep.md`, `dev/findings_cf_sweep_v2.md`,
  `dev/findings_ensemble_probe.md` — primary-source findings notes
  from each sweep.

---

## License

MIT

---

## Contributing

Contributions are welcome, especially around:

- SIMD acceleration of the cyclic-interval inner loop
- learned load-shaping heads layered _under_ phase (the natural
  follow-up to the BalancedRouter negative result)
- real-world integration examples (sharding, request routing)

---

## Summary

> Phase routing is a deterministic discrepancy-minimising balance
> primitive whose operating behaviour is invariant to capacity
> factor, immune to gate collapse by construction, and stable under
> provisioning sweeps — at a 5 – 12 % cross-entropy cost that grows
> with `E`. The contribution is the **substrate**, not the loss
> number.

Balance and specialisation are antagonistic under sparse routing;
phase routing picks balance and pays the affinity tax.

---

<img src="https://repo-view-counter.repo-view-counter-bitpackedphaserouter.workers.dev/track?repo=phase_router_rs&cb=1" width="1" height="1" style="display:none;" alt="">
