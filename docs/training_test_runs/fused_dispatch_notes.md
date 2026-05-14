# Fused `phase_router_uniform_dispatch` — Implementation Notes

**Date:** 2026-05-14
**Commit:** TBD (post-cf-sweep)

## Motivation

The capacity-factor sweep (`dev/findings_cf_sweep.md`) showed phase routing was
~22% slower than top-k on A10G (14k vs 17.7k tok/s), with route_time of
~34ms/forward vs top-k's ~10.5ms. Local profiling traced ~22ms of that gap to
**pure Python overhead** sitting between the kernel and the model:

| stage                           |    py time |
| ------------------------------- | ---------: |
| `_build_t_bits` (uniform)       |      ~5 ms |
| `_build_s_bits_uniform`         |      ~5 ms |
| `phase_router_auto` (kernel)    |     ~12 ms |
| per-row unique-pick Python loop |     ~12 ms |
| **total**                       | **~34 ms** |

Only the kernel call (12 ms) was actual algorithmic work; the other 22 ms was
NumPy bit-twiddling and a token-by-token `set()` loop, both holding the GIL.

## Solution

Fuse all four stages into a single GIL-released Rust call:

```python
routes = phase_router_rs.phase_router_uniform_dispatch(
    n_tokens, n_experts, k, base_density, seed, oversample=4,
)  # → (n_tokens, k) int32, unique-per-row, [-1, n_experts)
```

### Internals (see `src/router.rs::phase_router_uniform_dispatch`)

1. **Pad** `n_tokens` up to a multiple of `n_experts` (`N_pad`,
   `width = N_pad / n_experts`).
2. **Build uniform bit matrices** in-place: each row is the same prototype,
   left-aligned with `ones_e = round(base_density * width)` (target) or
   `s_ones = round(base_density * N_pad)` (source) low bits set.
   _Cost:_ one `memcpy`-loop per matrix, no per-bit work.
3. **Shuffle** both column permutations from `ChaCha8Rng(seed)`.
4. **Call** the existing `phase_router(...)` kernel with
   `k_kernel = max(k * oversample, k + 1)`.
5. **Map + dedupe in parallel** with `rayon::par_chunks_mut(k)`:
   for each row, walk its `k_kernel` kernel columns, convert
   `column // width → expert id`, and emit the first `k` unique experts
   into the output. Uses a `u128` seen-bitmask for `n_experts ≤ 128`
   (no allocation per row) and a `Vec<bool>` fallback otherwise.

The GIL is released for the entire call.

## Verification

### Rust unit tests (`src/lib.rs`)

- `uniform_dispatch_shape_and_uniqueness` — checks (n,k) shape, per-row uniqueness,
  range `[0, n_experts)`, and that load-CV across experts < 0.10.
- `uniform_dispatch_determinism` — same seed → identical output.
- `uniform_dispatch_n_not_divisible` — `n_tokens=1000, n_experts=8` works.

All 4 tests pass under `cargo test --release`.

### Local CPU smoke (`N=2048, E=8, k=2`)

```
old path (Python glue + auto): 134.34±38.46 ms
new fused dispatch:             33.28±52.02 ms
speedup: 4.04x
```

- 0 rows with duplicates
- per-expert counts within ±5% of mean (CV 0.0463)
- determinism verified across two calls

### End-to-end `PhaseRouter.forward`

```
PhaseRouter.forward median: 52.32 ms
TopKRouter.forward median:  48.60 ms
```

**CPU parity reached.** Previously phase was ~3× slower than top-k.

## API surface

```rust
// src/router.rs
pub fn phase_router_uniform_dispatch(
    n_tokens: usize,
    n_experts: usize,
    k: usize,
    base_density: f64,
    oversample: usize,
    seed: u64,
) -> Vec<i32>;
```

```python
# Python (via PyO3)
phase_router_rs.phase_router_uniform_dispatch(
    n_tokens: int,
    n_experts: int,
    k: int,
    base_density: float,
    seed: int,
    oversample: int = 4,           # kw-default
) -> np.ndarray  # shape (n_tokens, k), dtype=int32
```

The Python signature deliberately uses scalar args only (no NumPy buffers in,
single NumPy buffer out) — this is the cleanest possible PyO3 boundary and
allows the Rust side to choose the optimal bit-packing layout without
round-tripping through NumPy.

## `train/routers.py` integration

`PhaseRouter._select` now branches on `capacity_mode`:

- `"uniform"` (production) → one call to `phase_router_uniform_dispatch`.
- `"ema"` (kept for ablation) → original NumPy/`phase_router_auto` path
  with the per-row unique-pick loop.

The EMA path is still useful for future studies of demand-shaped capacity but
is not currently used by any production config.

## Next step

Re-run the cf sweep on Modal A10G — expected outcome:

- phase route_time: **~34 ms → ~12 ms** per forward (parity with top-k kernel time)
- phase throughput: **~14k tok/s → ~17–18k tok/s** (parity with top-k)
- all CE / CV / drop numbers from `findings_cf_sweep.md` unchanged
  (this is a pure plumbing change — no algorithmic change to selection)
