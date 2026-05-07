//! Rank/select OLBIO router (intrinsic occupancy transport, Option A).
//!
//! Replaces the source-side `inv_perm_s[p]` global indirection with a
//! per-row `select1(s_bits[j], r')` decode, where `r'` is a phase-mixed
//! occupancy index in `[0, d_j)`. Target side is **unchanged** — same
//! cyclic occupancy interval test as the original `phase_router`. This
//! isolates the empirical question:
//!
//! > does intrinsic occupancy transport on the source side improve
//! > structural / load-balancing properties before any broadword work?
//!
//! Three variants are provided:
//!
//! - [`phase_router_rs`]         — additive shift  `r' = (r + φ_j) mod d_j`
//! - [`phase_router_rs_affine`]  — affine          `r' = (a_j·r + b_j) mod d_j`,
//!                                 with `gcd(a_j, d_j) = 1`. This is the
//!                                 "eliminate global permutations" experiment
//!                                 from `dev/plan.md` Phase 5.
//! - [`phase_router_rs_hybrid`]  — preserves the original kernel's
//!                                 cumulative-degree phase from `offsets_s`
//!                                 *and* decodes intrinsically via `select1`.
//!                                 `r' = (offsets_s[j] + i) mod d_j;
//!                                  col = select1(row, r')`, for `i = 0..d_j`.
//!
//! The hybrid is the kernel that empirically reconciles the two designs:
//! intrinsic-decode kernels collapse on identical-support workloads
//! (see `dev/rank_select_progress.md`), but the original `phase_router` owes
//! its low-discrepancy load distribution to the global occupancy walk over
//! `offsets_s`, not to support structure. The hybrid keeps that walk while
//! eliminating the support-independent `inv_perm_s` indirection.
//!
//! All three variants take the same signature as [`crate::router::phase_router`]
//! to make head-to-head benches trivial. `col_perm_s` is *accepted* (so
//! `examples/moe_bench.rs` etc. compile unchanged) but used **only** to mix
//! the seed for per-row phase derivation — it is not used to walk the support.

use crate::bitsupport::{coprime_multiplier, mix64, select1};
use crate::core::{compute_offsets, compute_row_ones};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

// ── shared target-side overlap test ──────────────────────────────────────

/// Branchless cyclic interval membership: is `p_t` in `[start, start+len) mod n`?
#[inline]
fn in_cyclic_interval(p_t: usize, start: usize, len: usize, n: usize) -> bool {
    let mut d = p_t.wrapping_sub(start);
    if p_t < start {
        d = d.wrapping_add(n);
    }
    d < len
}

// ── additive variant ─────────────────────────────────────────────────────

/// Hybrid rank/select router with **additive** rank-space transport.
///
/// `r' = (r + φ_j) mod d_j`, `φ_j = mix64(seed XOR j) mod d_j`.
///
/// Target side is identical to [`crate::router::phase_router`].
pub fn phase_router_rs(
    s_bits: &[u64],
    t_bits: &[u64],
    n: usize,
    nb_words: usize,
    k: usize,
    col_perm_s: &[usize],
    col_perm_t: &[usize],
    seed: u64,
) -> Vec<i32> {
    debug_assert_eq!(s_bits.len(), n * nb_words);
    debug_assert_eq!(t_bits.len(), n * nb_words);
    debug_assert_eq!(col_perm_s.len(), n);
    debug_assert_eq!(col_perm_t.len(), n);

    // Target-side precompute identical to the original kernel.
    let offsets_t = compute_offsets(t_bits, n, nb_words);
    let ones_t = compute_row_ones(t_bits, n, nb_words);
    let ones_s = compute_row_ones(s_bits, n, nb_words);

    let mut routes = vec![-1i32; n * k];

    routes
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(j, row_out)| {
            let degree = ones_s[j];
            if degree == 0 {
                return;
            }
            let row_words = &s_bits[j * nb_words..(j + 1) * nb_words];

            // Per-row phase: deterministic, derived without a global permutation.
            // We mix `col_perm_s[j]` into the seed so callers who explicitly
            // randomise per-row phases via the source permutation still get
            // distinct routes from those who do not.
            let phase_seed = seed
                .wrapping_add(j as u64)
                .wrapping_add(col_perm_s[j] as u64);
            let phi = (mix64(phase_seed) as usize) % degree;
            let p_t = col_perm_t[j];

            let mut rng = ChaCha8Rng::seed_from_u64(phase_seed.wrapping_add(0xC0FFEE));
            let mut count: usize = 0;

            for r in 0..degree {
                let mut r_prime = r + phi;
                if r_prime >= degree {
                    r_prime -= degree;
                }
                let col = select1(row_words, r_prime as u32);

                let start = offsets_t[col];
                let len = ones_t[col];
                if in_cyclic_interval(p_t, start, len, n) {
                    if count < k {
                        row_out[count] = col as i32;
                    } else {
                        let rr = rng.gen_range(0..=count);
                        if rr < k {
                            row_out[rr] = col as i32;
                        }
                    }
                    count += 1;
                }
            }
        });

    routes
}

// ── affine variant (eliminates the global permutation idea) ─────────────

/// Hybrid rank/select router with **affine** rank-space transport.
///
/// `r' = (a_j · r + b_j) mod d_j`, with `gcd(a_j, d_j) = 1`. This is the
/// "Phase 5" experiment in `dev/plan.md`: per-row affine mixers replace the
/// global source-side permutation entirely. `col_perm_s` is consulted only
/// to perturb the `(a_j, b_j)` derivation seed.
pub fn phase_router_rs_affine(
    s_bits: &[u64],
    t_bits: &[u64],
    n: usize,
    nb_words: usize,
    k: usize,
    col_perm_s: &[usize],
    col_perm_t: &[usize],
    seed: u64,
) -> Vec<i32> {
    debug_assert_eq!(s_bits.len(), n * nb_words);
    debug_assert_eq!(t_bits.len(), n * nb_words);
    debug_assert_eq!(col_perm_s.len(), n);
    debug_assert_eq!(col_perm_t.len(), n);

    let offsets_t = compute_offsets(t_bits, n, nb_words);
    let ones_t = compute_row_ones(t_bits, n, nb_words);
    let ones_s = compute_row_ones(s_bits, n, nb_words);

    let mut routes = vec![-1i32; n * k];

    routes
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(j, row_out)| {
            let degree = ones_s[j];
            if degree == 0 {
                return;
            }
            let row_words = &s_bits[j * nb_words..(j + 1) * nb_words];

            let row_seed = seed
                .wrapping_add(j as u64)
                .wrapping_add(col_perm_s[j] as u64);
            let a = coprime_multiplier(degree, row_seed ^ 0xA1F1_F2F3_F4F5_F6F7);
            let b = (mix64(row_seed ^ 0xB0B1_B2B3_B4B5_B6B7) as usize) % degree;
            let p_t = col_perm_t[j];

            let mut rng = ChaCha8Rng::seed_from_u64(row_seed ^ 0xC0FF_EE_C0FFEE);
            let mut count: usize = 0;

            for r in 0..degree {
                // r_prime = (a*r + b) mod d. Both a < d and b < d, r < d, so
                // (a*r + b) fits in usize for any realistic n.
                let r_prime = (a.wrapping_mul(r).wrapping_add(b)) % degree;
                let col = select1(row_words, r_prime as u32);

                let start = offsets_t[col];
                let len = ones_t[col];
                if in_cyclic_interval(p_t, start, len, n) {
                    if count < k {
                        row_out[count] = col as i32;
                    } else {
                        let rr = rng.gen_range(0..=count);
                        if rr < k {
                            row_out[rr] = col as i32;
                        }
                    }
                    count += 1;
                }
            }
        });

    routes
}

// ── hybrid variant (preserves global occupancy walk + intrinsic decode) ─

/// **Hybrid** rank/select router. This is the kernel that reconciles the
/// occupancy-walk geometry of the original [`crate::router::phase_router`]
/// with the intrinsic-decode property of the rank/select variants.
///
/// Routine, per source row `j`:
/// ```text
/// degree = popcount(s_bits[j])
/// phi    = offsets_s[j] mod degree               // cumulative-degree phase
/// for i in 0..degree {
///     r   = (i + phi) mod degree                 // bijective rank-space walk
///     col = select1(s_bits[j], r)                // intrinsic decode
///     // overlap test against (offsets_t[col], ones_t[col], p_t = col_perm_t[j])
/// }
/// ```
///
/// Why this works:
///
/// - The phase `offsets_s[j] mod d_j` is the *cumulative-degree* marker of
///   row `j` in the original kernel's global walk over physical positions
///   `[offsets_s[j], offsets_s[j] + d_j) mod n`. It is the property that
///   makes `phase_router` a *low-discrepancy distributed occupancy
///   allocator*: rows with adjacent cumulative-degree positions get
///   adjacent (mod d_j) phases, threading the rank-space walks across rows
///   in a deterministic, low-discrepancy way.
/// - The decode `col = select1(row, r)` is intrinsic to the row support,
///   so on workloads with diverse per-row supports it differentiates rows
///   that the original `inv_perm_s × offsets_s` indirection cannot
///   naturally distinguish (since `inv_perm_s` is support-independent).
///
/// Note: a naive "`r = g mod d_j` while walking `g` cyclically through
/// `[0, n)` for `d_j` steps" would *not* be bijective on rank-space when
/// `d_j ∤ n` (the wraparound at `n` skips residues). Using the additive
/// form here is both bijective by construction and equivalent on the
/// `n mod d_j == 0` regime.
///
/// `col_perm_s` is consumed (signature compatibility) but not used to walk
/// the source: the hybrid is permutation-free on the source side, like the
/// affine variant.
pub fn phase_router_rs_hybrid(
    s_bits: &[u64],
    t_bits: &[u64],
    n: usize,
    nb_words: usize,
    k: usize,
    col_perm_s: &[usize],
    col_perm_t: &[usize],
    seed: u64,
) -> Vec<i32> {
    debug_assert_eq!(s_bits.len(), n * nb_words);
    debug_assert_eq!(t_bits.len(), n * nb_words);
    debug_assert_eq!(col_perm_s.len(), n);
    debug_assert_eq!(col_perm_t.len(), n);

    // Source-side and target-side precompute identical to the original.
    let offsets_s = compute_offsets(s_bits, n, nb_words);
    let offsets_t = compute_offsets(t_bits, n, nb_words);
    let ones_t = compute_row_ones(t_bits, n, nb_words);
    let ones_s = compute_row_ones(s_bits, n, nb_words);

    let mut routes = vec![-1i32; n * k];

    routes
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(j, row_out)| {
            let degree = ones_s[j];
            if degree == 0 {
                return;
            }
            let row_words = &s_bits[j * nb_words..(j + 1) * nb_words];
            let p_t = col_perm_t[j];

            // Reservoir RNG seeded mirroring the additive variant for
            // determinism; col_perm_s[j] still folds into the seed for
            // callers who supply a non-identity source permutation.
            let row_seed = seed
                .wrapping_add(j as u64)
                .wrapping_add(col_perm_s[j] as u64);
            let mut rng = ChaCha8Rng::seed_from_u64(row_seed ^ 0xC0FF_EE_C0FFEE);

            // Cumulative-degree-driven phase. This is the row's marker in the
            // global occupancy walk used by the original kernel; expressed in
            // rank-space it is a per-row additive shift, but unlike the
            // additive variant, the shift is *not* a hash of the row index —
            // it is the cross-row low-discrepancy quantity carried by
            // `offsets_s` (see module-level docs).
            let phi = offsets_s[j] % degree;
            let mut count: usize = 0;

            for i in 0..degree {
                // Bijective rank-space walk: r = (i + phi) mod degree.
                let mut r = i + phi;
                if r >= degree {
                    r -= degree;
                }
                let col = select1(row_words, r as u32);

                let start = offsets_t[col];
                let len = ones_t[col];
                if in_cyclic_interval(p_t, start, len, n) {
                    if count < k {
                        row_out[count] = col as i32;
                    } else {
                        let rr = rng.gen_range(0..=count);
                        if rr < k {
                            row_out[rr] = col as i32;
                        }
                    }
                    count += 1;
                }
            }
        });

    routes
}

// ── module-local sanity tests ────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitsupport::popcount_words;

    fn dense(n: usize, nb_words: usize) -> Vec<u64> {
        let mut v = vec![0u64; n * nb_words];
        for i in 0..n {
            for b in 0..n {
                v[i * nb_words + b / 64] |= 1u64 << (b % 64);
            }
        }
        v
    }

    #[test]
    fn additive_runs_and_fills_for_dense_input() {
        let n = 64;
        let nb = (n + 63) / 64;
        let s = dense(n, nb);
        let t = dense(n, nb);
        let perm: Vec<usize> = (0..n).collect();
        let k = 4;
        let routes = phase_router_rs(&s, &t, n, nb, k, &perm, &perm, 42);
        assert_eq!(routes.len(), n * k);
        // With dense input, every source row should be filled to k.
        for row in routes.chunks(k) {
            assert!(row.iter().all(|&c| c >= 0 && (c as usize) < n));
        }
    }

    #[test]
    fn affine_runs_and_fills_for_dense_input() {
        let n = 64;
        let nb = (n + 63) / 64;
        let s = dense(n, nb);
        let t = dense(n, nb);
        let perm: Vec<usize> = (0..n).collect();
        let k = 4;
        let routes = phase_router_rs_affine(&s, &t, n, nb, k, &perm, &perm, 42);
        assert_eq!(routes.len(), n * k);
        for row in routes.chunks(k) {
            assert!(row.iter().all(|&c| c >= 0 && (c as usize) < n));
        }
    }

    #[test]
    fn determinism_additive() {
        let n = 32;
        let nb = (n + 63) / 64;
        let s = dense(n, nb);
        let t = dense(n, nb);
        let perm: Vec<usize> = (0..n).collect();
        let a = phase_router_rs(&s, &t, n, nb, 3, &perm, &perm, 99);
        let b = phase_router_rs(&s, &t, n, nb, 3, &perm, &perm, 99);
        assert_eq!(a, b);
    }

    #[test]
    fn determinism_affine() {
        let n = 32;
        let nb = (n + 63) / 64;
        let s = dense(n, nb);
        let t = dense(n, nb);
        let perm: Vec<usize> = (0..n).collect();
        let a = phase_router_rs_affine(&s, &t, n, nb, 3, &perm, &perm, 99);
        let b = phase_router_rs_affine(&s, &t, n, nb, 3, &perm, &perm, 99);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_rows_are_skipped() {
        // Source row 0 has zero ones → no routes for that row.
        let n = 8;
        let nb = 1;
        let mut s = vec![0u64; n * nb];
        for i in 1..n {
            for b in 0..n {
                s[i * nb] |= 1u64 << b;
            }
        }
        let t = dense(n, nb);
        let perm: Vec<usize> = (0..n).collect();
        let routes = phase_router_rs(&s, &t, n, nb, 2, &perm, &perm, 1);
        assert!(routes[0..2].iter().all(|&c| c == -1));
    }

    #[test]
    fn _unused_offsets_s_is_not_called() {
        // Smoke test: the rank/select kernel does not depend on
        // compute_offsets for the source side, because it iterates
        // intrinsic rank-space.
        let n = 16;
        let nb = 1;
        let mut s = vec![0u64; n * nb];
        // Pathological row offsets in physical space: only row 0 has ones.
        s[0] = 0xFFFF;
        let t = dense(n, nb);
        let perm: Vec<usize> = (0..n).collect();
        // Should not panic, should emit some route for row 0.
        let routes = phase_router_rs(&s, &t, n, nb, 1, &perm, &perm, 7);
        assert!(routes[0] >= 0);
    }

    #[test]
    fn popcount_consistency_with_core_row_ones() {
        // Sanity: our bitsupport popcount matches the existing core helper.
        let n = 32;
        let nb = 1;
        let s = dense(n, nb);
        let core_ones = crate::core::compute_row_ones(&s, n, nb);
        for i in 0..n {
            let row = &s[i * nb..(i + 1) * nb];
            assert_eq!(popcount_words(row) as usize, core_ones[i]);
        }
    }
}
