//! Property-level invariants applied to every router variant.
//!
//! These tests check *behaviour*, not bit-equality between kernels: the
//! rank/select kernel produces structurally different routes from the
//! original `phase_router` (this is intended, see `dev/plan.md`).
//!
//! Run all kernels:
//!   cargo test --features rank-select
//!
//! Run only the original kernel:
//!   cargo test

use phase_router_rs::router::phase_router;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[cfg(feature = "rank-select")]
use phase_router_rs::router_rs::{
    phase_router_rs, phase_router_rs_affine, phase_router_rs_hybrid,
};

// ── workload helpers ─────────────────────────────────────────────────────

/// Build a bit-packed matrix where row `i` has the first `ones_per_row[i]`
/// bits set (matches the convention used by `examples/moe_bench.rs`).
fn build_bits(ones_per_row: &[usize], n: usize, nb_words: usize) -> Vec<u64> {
    let mut bits = vec![0u64; n * nb_words];
    for i in 0..n {
        let ones = ones_per_row[i].min(n);
        for b in 0..ones {
            bits[i * nb_words + b / 64] |= 1u64 << (b % 64);
        }
    }
    bits
}

/// Heterogeneous capacity scenario consistent with the README/MoE bench.
struct Workload {
    n: usize,
    nb_words: usize,
    k: usize,
    s_bits: Vec<u64>,
    t_bits: Vec<u64>,
    col_perm_s: Vec<usize>,
    col_perm_t: Vec<usize>,
    rel_caps: Vec<f64>,
    seed: u64,
}

fn make_workload(n: usize, k: usize, seed: u64) -> Workload {
    let nb_words = (n + 63) / 64;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Strong heterogeneity: 10% at 8x, 20% at 2x, rest at 1x.
    let mut rel_caps = vec![1.0f64; n];
    let t1 = (n as f64 * 0.1) as usize;
    let t2 = (n as f64 * 0.2) as usize;
    for v in rel_caps.iter_mut().take(t1) {
        *v = 8.0;
    }
    for v in rel_caps.iter_mut().skip(t1).take(t2) {
        *v = 2.0;
    }
    rel_caps.shuffle(&mut rng);

    let total: f64 = rel_caps.iter().sum();
    let mean = total / n as f64;

    // Source: uniform 30% density.
    let base_density = 0.3;
    let s_density = (base_density * n as f64).round() as usize;
    let s_ones = vec![s_density; n];
    let s_bits = build_bits(&s_ones, n, nb_words);

    // Target density ∝ relative capacity.
    let t_ones: Vec<usize> = rel_caps
        .iter()
        .map(|&c| {
            ((c / mean) * base_density * n as f64)
                .round()
                .max(1.0)
                .min(n as f64) as usize
        })
        .collect();
    let t_bits = build_bits(&t_ones, n, nb_words);

    let mut col_perm_s: Vec<usize> = (0..n).collect();
    let mut col_perm_t: Vec<usize> = (0..n).collect();
    col_perm_s.shuffle(&mut rng);
    col_perm_t.shuffle(&mut rng);

    Workload {
        n,
        nb_words,
        k,
        s_bits,
        t_bits,
        col_perm_s,
        col_perm_t,
        rel_caps,
        seed,
    }
}

// ── invariant checkers ──────────────────────────────────────────────────

fn check_fanout_bound(routes: &[i32], n: usize, k: usize) {
    assert_eq!(routes.len(), n * k);
    for (j, row) in routes.chunks(k).enumerate() {
        let assigned = row.iter().filter(|&&c| c >= 0).count();
        assert!(
            assigned <= k,
            "row {j}: {assigned} assignments exceeds fan-out {k}"
        );
        for &c in row {
            if c >= 0 {
                assert!(
                    (c as usize) < n,
                    "row {j}: column index {c} out of range [0, {n})"
                );
            }
        }
    }
}

fn check_dedup_within_row(routes: &[i32], k: usize) {
    for (j, row) in routes.chunks(k).enumerate() {
        let mut seen = std::collections::HashSet::new();
        for &c in row {
            if c >= 0 {
                assert!(
                    seen.insert(c),
                    "row {j}: column {c} appears twice (rank/select kernel \
                     enforces unique columns per row by construction)"
                );
            }
        }
    }
}

fn loads_per_target(routes: &[i32], n: usize) -> Vec<u32> {
    let mut loads = vec![0u32; n];
    for &c in routes {
        if c >= 0 {
            loads[c as usize] += 1;
        }
    }
    loads
}

/// Pearson correlation between two equal-length f64 sequences.
fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for i in 0..x.len() {
        let a = x[i] - mx;
        let b = y[i] - my;
        num += a * b;
        dx2 += a * a;
        dy2 += b * b;
    }
    if dx2 == 0.0 || dy2 == 0.0 {
        return 0.0;
    }
    num / (dx2.sqrt() * dy2.sqrt())
}

// ── tests on the original kernel (regression baseline) ──────────────────

#[test]
fn phase_router_is_deterministic() {
    let w = make_workload(256, 4, 0xA);
    let a = phase_router(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    let b = phase_router(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    assert_eq!(a, b);
}

#[test]
fn phase_router_seed_changes_routes() {
    let w = make_workload(256, 4, 0xA);
    let a = phase_router(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, 1,
    );
    let b = phase_router(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, 2,
    );
    assert_ne!(a, b, "different seeds should produce different routes");
}

#[test]
fn phase_router_fanout_bound() {
    let w = make_workload(512, 4, 1);
    let routes = phase_router(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    check_fanout_bound(&routes, w.n, w.k);
}

#[test]
fn phase_router_dedup() {
    let w = make_workload(512, 4, 2);
    let routes = phase_router(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    check_dedup_within_row(&routes, w.k);
}

#[test]
fn phase_router_load_correlates_with_capacity() {
    // Strong heterogeneity → load should track capacity.
    let w = make_workload(1024, 4, 3);
    let routes = phase_router(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    let loads = loads_per_target(&routes, w.n);
    let loads_f: Vec<f64> = loads.iter().map(|&l| l as f64).collect();
    let r = pearson(&loads_f, &w.rel_caps);
    // The README claims phase_router targets r ≈ 1.0; we ask only for > 0.5
    // to keep the test robust to tuning changes.
    assert!(
        r > 0.5,
        "capacity-load correlation {r:.3} below 0.5; phase_router should align load with capacity"
    );
}

// ── tests on the rank/select kernel (additive) ──────────────────────────

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_is_deterministic() {
    let w = make_workload(256, 4, 0xA);
    let a = phase_router_rs(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    let b = phase_router_rs(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    assert_eq!(a, b);
}

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_seed_changes_routes() {
    let w = make_workload(256, 4, 0xA);
    let a = phase_router_rs(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, 1,
    );
    let b = phase_router_rs(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, 2,
    );
    assert_ne!(a, b);
}

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_fanout_bound() {
    let w = make_workload(512, 4, 11);
    let routes = phase_router_rs(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    check_fanout_bound(&routes, w.n, w.k);
}

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_dedup() {
    // Each occupancy index r' maps to a distinct physical column via
    // select1, so the additive variant must produce unique columns per row.
    let w = make_workload(512, 4, 12);
    let routes = phase_router_rs(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    check_dedup_within_row(&routes, w.k);
}

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_emits_columns_inside_source_support() {
    // Crucial validity check under transformed semantics: every emitted
    // column must be a 1-bit in the source row's *original* support.
    let w = make_workload(256, 4, 13);
    let routes = phase_router_rs(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    for (j, row) in routes.chunks(w.k).enumerate() {
        let row_words = &w.s_bits[j * w.nb_words..(j + 1) * w.nb_words];
        for &c in row {
            if c >= 0 {
                let cu = c as usize;
                let bit_set = (row_words[cu / 64] >> (cu % 64)) & 1 == 1;
                assert!(
                    bit_set,
                    "row {j}: emitted column {cu} not in source support"
                );
            }
        }
    }
}

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_load_correlates_with_capacity() {
    let w = make_workload(1024, 4, 14);
    let routes = phase_router_rs(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    let loads = loads_per_target(&routes, w.n);
    let loads_f: Vec<f64> = loads.iter().map(|&l| l as f64).collect();
    let r = pearson(&loads_f, &w.rel_caps);
    assert!(
        r > 0.3,
        "rank/select additive: capacity-load correlation {r:.3} too low \
         (target side preserved → some correlation expected)"
    );
}

// ── tests on the rank/select kernel (affine) ────────────────────────────

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_affine_is_deterministic() {
    let w = make_workload(256, 4, 0xB);
    let a = phase_router_rs_affine(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    let b = phase_router_rs_affine(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    assert_eq!(a, b);
}

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_affine_fanout_bound() {
    let w = make_workload(512, 4, 21);
    let routes = phase_router_rs_affine(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    check_fanout_bound(&routes, w.n, w.k);
}

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_affine_dedup() {
    // Affine `r' = a*r + b mod d` with gcd(a,d)=1 is a bijection on [0,d),
    // so each call to select1 produces a distinct column.
    let w = make_workload(512, 4, 22);
    let routes = phase_router_rs_affine(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    check_dedup_within_row(&routes, w.k);
}

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_affine_emits_columns_inside_source_support() {
    let w = make_workload(256, 4, 23);
    let routes = phase_router_rs_affine(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    for (j, row) in routes.chunks(w.k).enumerate() {
        let row_words = &w.s_bits[j * w.nb_words..(j + 1) * w.nb_words];
        for &c in row {
            if c >= 0 {
                let cu = c as usize;
                let bit_set = (row_words[cu / 64] >> (cu % 64)) & 1 == 1;
                assert!(bit_set, "row {j}: emitted column {cu} not in source support");
            }
        }
    }
}

#[cfg(feature = "rank-select")]
#[test]
fn affine_differs_from_additive() {
    // Sanity: the two rank/select variants should not collapse into the
    // same routes — they encode different rank-space transports.
    let w = make_workload(256, 4, 31);
    let add = phase_router_rs(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    let aff = phase_router_rs_affine(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    assert_ne!(add, aff);
}

#[cfg(feature = "rank-select")]
#[test]
fn rank_select_differs_from_phase_router() {
    // The whole point: rank/select should produce *structurally different*
    // routes from the global-permutation kernel. If they ever match, either
    // the workload is degenerate or one kernel has silently degraded into
    // the other.
    let w = make_workload(256, 4, 41);
    let pr = phase_router(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    let rs = phase_router_rs(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    assert_ne!(pr, rs);
}

// ── tests on the rank/select kernel (hybrid) ────────────────────────────

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_hybrid_is_deterministic() {
    let w = make_workload(256, 4, 0xC);
    let a = phase_router_rs_hybrid(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    let b = phase_router_rs_hybrid(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    assert_eq!(a, b);
}

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_hybrid_seed_changes_routes() {
    let w = make_workload(256, 4, 0xC);
    let a = phase_router_rs_hybrid(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, 1,
    );
    let b = phase_router_rs_hybrid(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, 2,
    );
    assert_ne!(a, b);
}

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_hybrid_fanout_bound() {
    let w = make_workload(512, 4, 51);
    let routes = phase_router_rs_hybrid(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    check_fanout_bound(&routes, w.n, w.k);
}

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_hybrid_dedup() {
    // The hybrid walks `g` over `d_j` consecutive integers in `[0, n)`,
    // producing `d_j` distinct values of `g mod d_j` only when `d_j` divides
    // `n` or when the walk is short enough to avoid wrap-around alias. In
    // general some `r` values may repeat, but `select1(row, r)` for the same
    // `r` returns the same physical column, so duplicates can be deduped at
    // most by reservoir sampling (which still emits unique columns when k is
    // small relative to the number of distinct candidates). For the standard
    // workload (d ≈ 0.3*n, n large, k=4) we expect dedup to hold.
    let w = make_workload(512, 4, 52);
    let routes = phase_router_rs_hybrid(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    check_dedup_within_row(&routes, w.k);
}

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_hybrid_emits_columns_inside_source_support() {
    let w = make_workload(256, 4, 53);
    let routes = phase_router_rs_hybrid(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    for (j, row) in routes.chunks(w.k).enumerate() {
        let row_words = &w.s_bits[j * w.nb_words..(j + 1) * w.nb_words];
        for &c in row {
            if c >= 0 {
                let cu = c as usize;
                let bit_set = (row_words[cu / 64] >> (cu % 64)) & 1 == 1;
                assert!(
                    bit_set,
                    "row {j}: emitted column {cu} not in source support"
                );
            }
        }
    }
}

#[cfg(feature = "rank-select")]
#[test]
fn phase_router_rs_hybrid_load_correlates_with_capacity() {
    let w = make_workload(1024, 4, 54);
    let routes = phase_router_rs_hybrid(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    let loads = loads_per_target(&routes, w.n);
    let loads_f: Vec<f64> = loads.iter().map(|&l| l as f64).collect();
    let r = pearson(&loads_f, &w.rel_caps);
    assert!(
        r > 0.3,
        "rank/select hybrid: capacity-load correlation {r:.3} too low"
    );
}

#[cfg(feature = "rank-select")]
#[test]
fn hybrid_differs_from_additive_and_affine_and_phase_router() {
    let w = make_workload(256, 4, 55);
    let pr = phase_router(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    let add = phase_router_rs(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    let aff = phase_router_rs_affine(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    let hyb = phase_router_rs_hybrid(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, w.k, &w.col_perm_s, &w.col_perm_t, w.seed,
    );
    assert_ne!(hyb, pr, "hybrid should not equal phase_router");
    assert_ne!(hyb, add, "hybrid should not equal additive RS");
    assert_ne!(hyb, aff, "hybrid should not equal affine RS");
}

// ── structural-impossibility sanity check ───────────────────────────────
//
// On an *identical-support* workload (every row has the same dense
// contiguous support `[0, d)`), all intrinsic-decode kernels visit the same
// per-row candidate stream `select1(row, r) = r`, so they differ only in
// the order of candidates and the per-row reservoir RNG. The Jaccard
// overlap between any two rows' candidate sets is therefore exactly 1.0.
// The hybrid walks `g` from `offsets_s[j]` and so visits the *same* set of
// candidates per row (the entire support); only the reservoir RNG breaks
// the tie. This test asserts that property explicitly so future regressions
// can't silently undo it.

#[cfg(feature = "rank-select")]
#[test]
fn intrinsic_decode_kernels_visit_full_support_on_dense_uniform() {
    use phase_router_rs::workloads::Profile;

    let n = 128;
    let k = 8;
    let w = Profile::DenseUniform.build(n, 0.3, 0.3, 0xCAFE);

    // Use k large enough that all candidates fit in the reservoir for a
    // small support — then routes are exactly the candidate stream.
    let routes_add = phase_router_rs(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, k, &w.col_perm_s, &w.col_perm_t, 7,
    );
    let routes_aff = phase_router_rs_affine(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, k, &w.col_perm_s, &w.col_perm_t, 7,
    );
    let routes_hyb = phase_router_rs_hybrid(
        &w.s_bits, &w.t_bits, w.n, w.nb_words, k, &w.col_perm_s, &w.col_perm_t, 7,
    );

    // Every intrinsic-decode kernel must produce only columns in the shared
    // support `[0, d)` and never anything outside it.
    let d = ((0.3 * n as f64).round()) as usize;
    for routes in [&routes_add, &routes_aff, &routes_hyb] {
        for &c in routes.iter() {
            if c >= 0 {
                assert!(
                    (c as usize) < d,
                    "kernel emitted column {c} outside identical support [0, {d})"
                );
            }
        }
    }
}

// ── invariants across all four canonical workload profiles ──────────────

#[cfg(feature = "rank-select")]
#[test]
fn fanout_and_validity_across_workloads() {
    use phase_router_rs::workloads::Profile;

    for profile in Profile::ALL {
        let w = profile.build(256, 0.3, 0.3, 0x99);

        for (name, routes) in [
            (
                "phase_router",
                phase_router(
                    &w.s_bits, &w.t_bits, w.n, w.nb_words, 4,
                    &w.col_perm_s, &w.col_perm_t, 5,
                ),
            ),
            (
                "rs_additive",
                phase_router_rs(
                    &w.s_bits, &w.t_bits, w.n, w.nb_words, 4,
                    &w.col_perm_s, &w.col_perm_t, 5,
                ),
            ),
            (
                "rs_affine",
                phase_router_rs_affine(
                    &w.s_bits, &w.t_bits, w.n, w.nb_words, 4,
                    &w.col_perm_s, &w.col_perm_t, 5,
                ),
            ),
            (
                "rs_hybrid",
                phase_router_rs_hybrid(
                    &w.s_bits, &w.t_bits, w.n, w.nb_words, 4,
                    &w.col_perm_s, &w.col_perm_t, 5,
                ),
            ),
        ] {
            check_fanout_bound(&routes, w.n, 4);
            check_dedup_within_row(&routes, 4);

            // Validity is required for all RS kernels; phase_router by
            // construction emits columns from the source support too
            // (it walks `inv_perm_s[p]` over physical positions inside
            // the row's offset window and the support is dense in this
            // workload). We just enforce the universal property.
            for (j, row) in routes.chunks(4).enumerate() {
                let row_words = &w.s_bits[j * w.nb_words..(j + 1) * w.nb_words];
                for &c in row {
                    if c >= 0 {
                        let cu = c as usize;
                        let bit_set = (row_words[cu / 64] >> (cu % 64)) & 1 == 1;
                        assert!(
                            bit_set,
                            "[{}/{:?}] row {j}: emitted column {cu} not in source support",
                            name, profile,
                        );
                    }
                }
            }
        }
    }
}
