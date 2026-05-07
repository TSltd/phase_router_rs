//! Workload taxonomy for diagnostic benchmarking.
//!
//! The original `phase_router` performed strongly on the MoE benchmark, but
//! that benchmark has a hidden structural property: every source row has the
//! same dense contiguous support `[0, d)`. On such an *identical-support*
//! workload, intrinsic-decode kernels (rank/select with additive or affine
//! transport) are **structurally** incapable of differentiating rows — every
//! row produces the same candidate set under `select1`, so reservoir sampling
//! erases ordering. See `dev/rank_select_progress.md` for the full argument.
//!
//! To disentangle this, we expose four parameterised workload profiles. They
//! all share the same return shape so examples / tests / benches can iterate
//! over them uniformly.
//!
//! | profile           | per-row degree | per-row support             |
//! | ----------------- | -------------- | --------------------------- |
//! | `dense_uniform`   | constant `d`   | identical `[0, d)`          |
//! | `dense_diverse`   | constant `d`   | random size-`d` subset      |
//! | `sparse_diverse`  | heavy-tailed   | random subset of that size  |
//! | `block_local`     | constant `d`   | contiguous `[o_i, o_i + d)` |
//!
//! All randomness is deterministic via SplitMix64 (`crate::bitsupport::mix64`).
//! No `rand` dependency in this module.
//!
//! The bit layout matches the convention used elsewhere in this crate:
//! `bits[i * nb_words .. (i + 1) * nb_words]` is row `i`'s packed support.

use crate::bitsupport::mix64;

// ── shared output type ───────────────────────────────────────────────────

/// All workload constructors return this. `bits.len() == n * nb_words`.
pub struct Workload {
    pub n: usize,
    pub nb_words: usize,
    /// Source bit matrix.
    pub s_bits: Vec<u64>,
    /// Target bit matrix (capacity-shaped on the target side).
    pub t_bits: Vec<u64>,
    /// Per-target relative capacity (driving `t_bits` density).
    pub rel_caps: Vec<f64>,
    /// Source column permutation (as required by the router signature).
    pub col_perm_s: Vec<usize>,
    /// Target column permutation (as required by the router signature).
    pub col_perm_t: Vec<usize>,
    /// Human-readable label, e.g. `"dense_uniform(d=307)"`.
    pub label: String,
}

// ── helpers ──────────────────────────────────────────────────────────────

#[inline]
fn set_bit(bits: &mut [u64], i: usize, nb_words: usize, b: usize) {
    bits[i * nb_words + b / 64] |= 1u64 << (b % 64);
}

/// Deterministic Fisher–Yates over `[0, n)` using SplitMix64.
fn shuffle_range(n: usize, mut seed: u64) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        seed = mix64(seed);
        let j = (seed as usize) % (i + 1);
        v.swap(i, j);
    }
    v
}

/// Strong heterogeneity capacity profile: 10% × 8.0 + 20% × 2.0 + rest × 1.0.
/// Shuffled deterministically so target densities are not contiguous.
fn strong_hetero_caps(n: usize, seed: u64) -> Vec<f64> {
    let mut c = vec![1.0f64; n];
    let t1 = (n as f64 * 0.1) as usize;
    let t2 = (n as f64 * 0.2) as usize;
    for v in c.iter_mut().take(t1) {
        *v = 8.0;
    }
    for v in c.iter_mut().skip(t1).take(t2) {
        *v = 2.0;
    }
    let perm = shuffle_range(n, seed);
    let mut out = vec![0.0f64; n];
    for i in 0..n {
        out[perm[i]] = c[i];
    }
    out
}

/// Build the **target** bit matrix from `rel_caps` so target density tracks
/// capacity (matches the convention used by `examples/moe_bench.rs`).
fn build_target_bits(rel_caps: &[f64], n: usize, nb_words: usize, base_density: f64) -> Vec<u64> {
    let total: f64 = rel_caps.iter().sum();
    let mean = total / n as f64;
    let mut bits = vec![0u64; n * nb_words];
    for i in 0..n {
        let ones = ((rel_caps[i] / mean) * base_density * n as f64)
            .round()
            .max(1.0)
            .min(n as f64) as usize;
        for b in 0..ones {
            set_bit(&mut bits, i, nb_words, b);
        }
    }
    bits
}

// ── public profiles ──────────────────────────────────────────────────────

/// **Identical-support workload.** Every source row has support `[0, d)`,
/// where `d = round(density * n)`. This is the original `moe_bench` shape;
/// it is the *pathological* case for intrinsic-decode kernels because all
/// rows produce identical `select1` candidate streams.
pub fn dense_uniform(n: usize, density: f64, base_density: f64, seed: u64) -> Workload {
    let nb_words = (n + 63) / 64;
    let d = ((density * n as f64).round() as usize).clamp(1, n);
    let mut s_bits = vec![0u64; n * nb_words];
    for i in 0..n {
        for b in 0..d {
            set_bit(&mut s_bits, i, nb_words, b);
        }
    }

    let rel_caps = strong_hetero_caps(n, seed ^ 0xC0CA_C01A_BEEF_5EED);
    let t_bits = build_target_bits(&rel_caps, n, nb_words, base_density);

    let col_perm_s = shuffle_range(n, seed ^ 0xA1A1_A1A1);
    let col_perm_t = shuffle_range(n, seed ^ 0xB2B2_B2B2);

    Workload {
        n,
        nb_words,
        s_bits,
        t_bits,
        rel_caps,
        col_perm_s,
        col_perm_t,
        label: format!("dense_uniform(d={d})"),
    }
}

/// **Same degree, diverse supports.** Every row has degree `d`, but the
/// concrete `d` columns chosen for each row are a deterministic random
/// subset (via Fisher–Yates from a per-row SplitMix64 seed). This isolates
/// "diversity of support" from "diversity of degree" when comparing kernels.
pub fn dense_diverse(n: usize, density: f64, base_density: f64, seed: u64) -> Workload {
    let nb_words = (n + 63) / 64;
    let d = ((density * n as f64).round() as usize).clamp(1, n);
    let mut s_bits = vec![0u64; n * nb_words];
    for i in 0..n {
        let row_seed = mix64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let perm = shuffle_range(n, row_seed);
        for b in 0..d {
            set_bit(&mut s_bits, i, nb_words, perm[b]);
        }
    }

    let rel_caps = strong_hetero_caps(n, seed ^ 0xC0CA_C01A_BEEF_5EED);
    let t_bits = build_target_bits(&rel_caps, n, nb_words, base_density);

    let col_perm_s = shuffle_range(n, seed ^ 0xA1A1_A1A1);
    let col_perm_t = shuffle_range(n, seed ^ 0xB2B2_B2B2);

    Workload {
        n,
        nb_words,
        s_bits,
        t_bits,
        rel_caps,
        col_perm_s,
        col_perm_t,
        label: format!("dense_diverse(d={d})"),
    }
}

/// **Heavy-tailed degree, diverse supports.** Per-row degrees are drawn from
/// a discretised Pareto (truncated power law) with parameter `tail_alpha`
/// and per-row supports are uniform random subsets of that size.
///
/// The mean degree is approximately `mean_density * n`; rows can range from
/// `1` to `n`. Useful for stressing kernels on skewed-degree workloads
/// without coupling the skew to capacity.
pub fn sparse_diverse(
    n: usize,
    mean_density: f64,
    tail_alpha: f64,
    base_density: f64,
    seed: u64,
) -> Workload {
    let nb_words = (n + 63) / 64;

    // Discretised Pareto over [1, n]. CDF: 1 - (x_min / x)^alpha. We pick
    // x_min so the *mean* matches `mean_density * n` (mean = α·x_min/(α-1)
    // for α > 1), then sample by inverting the CDF on a uniform.
    let target_mean = (mean_density * n as f64).max(1.0);
    let alpha = tail_alpha.max(1.05);
    let x_min = target_mean * (alpha - 1.0) / alpha;

    let mut s_bits = vec![0u64; n * nb_words];
    for i in 0..n {
        let row_seed = mix64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        // Uniform u in (0, 1) (avoid 0 to keep finite x).
        let u = ((row_seed >> 11) as f64 / (1u64 << 53) as f64).clamp(1e-9, 1.0 - 1e-9);
        let xf = x_min / (1.0 - u).powf(1.0 / alpha);
        let d = (xf.round() as usize).clamp(1, n);

        let perm = shuffle_range(n, mix64(row_seed));
        for b in 0..d {
            set_bit(&mut s_bits, i, nb_words, perm[b]);
        }
    }

    let rel_caps = strong_hetero_caps(n, seed ^ 0xC0CA_C01A_BEEF_5EED);
    let t_bits = build_target_bits(&rel_caps, n, nb_words, base_density);

    let col_perm_s = shuffle_range(n, seed ^ 0xA1A1_A1A1);
    let col_perm_t = shuffle_range(n, seed ^ 0xB2B2_B2B2);

    Workload {
        n,
        nb_words,
        s_bits,
        t_bits,
        rel_caps,
        col_perm_s,
        col_perm_t,
        label: format!("sparse_diverse(mean={target_mean:.0},α={alpha:.2})"),
    }
}

/// **Constant degree, contiguous block at row-dependent offset.** Each row
/// has a contiguous support `[o_i, o_i + d) mod n` where `o_i` is derived
/// from `mix64(seed ^ i)`. Tests whether kernels can exploit per-row
/// translational symmetry of the support.
pub fn block_local(n: usize, density: f64, base_density: f64, seed: u64) -> Workload {
    let nb_words = (n + 63) / 64;
    let d = ((density * n as f64).round() as usize).clamp(1, n);
    let mut s_bits = vec![0u64; n * nb_words];
    for i in 0..n {
        let row_seed = mix64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let o = (row_seed as usize) % n;
        for off in 0..d {
            let col = (o + off) % n;
            set_bit(&mut s_bits, i, nb_words, col);
        }
    }

    let rel_caps = strong_hetero_caps(n, seed ^ 0xC0CA_C01A_BEEF_5EED);
    let t_bits = build_target_bits(&rel_caps, n, nb_words, base_density);

    let col_perm_s = shuffle_range(n, seed ^ 0xA1A1_A1A1);
    let col_perm_t = shuffle_range(n, seed ^ 0xB2B2_B2B2);

    Workload {
        n,
        nb_words,
        s_bits,
        t_bits,
        rel_caps,
        col_perm_s,
        col_perm_t,
        label: format!("block_local(d={d})"),
    }
}

// ── enumerable handle for examples/tests ─────────────────────────────────

/// Identifier for the four canonical profiles. Useful when an example wants
/// to sweep all of them by name without hard-coding closures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    DenseUniform,
    DenseDiverse,
    SparseDiverse,
    BlockLocal,
}

impl Profile {
    pub const ALL: [Profile; 4] = [
        Profile::DenseUniform,
        Profile::DenseDiverse,
        Profile::SparseDiverse,
        Profile::BlockLocal,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Profile::DenseUniform => "dense_uniform",
            Profile::DenseDiverse => "dense_diverse",
            Profile::SparseDiverse => "sparse_diverse",
            Profile::BlockLocal => "block_local",
        }
    }

    /// Build the workload for this profile with sensible defaults.
    /// `density` is the source density; targets get `base_density` modulated
    /// by `rel_caps`.
    pub fn build(self, n: usize, density: f64, base_density: f64, seed: u64) -> Workload {
        match self {
            Profile::DenseUniform => dense_uniform(n, density, base_density, seed),
            Profile::DenseDiverse => dense_diverse(n, density, base_density, seed),
            Profile::SparseDiverse => sparse_diverse(n, density, 1.5, base_density, seed),
            Profile::BlockLocal => block_local(n, density, base_density, seed),
        }
    }
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitsupport::popcount_words;

    fn assert_basics(w: &Workload) {
        assert_eq!(w.s_bits.len(), w.n * w.nb_words);
        assert_eq!(w.t_bits.len(), w.n * w.nb_words);
        assert_eq!(w.col_perm_s.len(), w.n);
        assert_eq!(w.col_perm_t.len(), w.n);
        assert_eq!(w.rel_caps.len(), w.n);

        // Permutations must be valid.
        let mut seen_s = vec![false; w.n];
        for &p in &w.col_perm_s {
            assert!(p < w.n);
            assert!(!seen_s[p]);
            seen_s[p] = true;
        }
        let mut seen_t = vec![false; w.n];
        for &p in &w.col_perm_t {
            assert!(p < w.n);
            assert!(!seen_t[p]);
            seen_t[p] = true;
        }

        // Every row should have at least one bit set.
        for i in 0..w.n {
            let row = &w.s_bits[i * w.nb_words..(i + 1) * w.nb_words];
            assert!(popcount_words(row) > 0, "row {i} is empty");
        }
    }

    #[test]
    fn dense_uniform_has_identical_supports() {
        let w = dense_uniform(128, 0.3, 0.3, 42);
        assert_basics(&w);
        let row0 = &w.s_bits[0..w.nb_words].to_vec();
        for i in 1..w.n {
            let rowi = &w.s_bits[i * w.nb_words..(i + 1) * w.nb_words];
            assert_eq!(rowi, row0.as_slice());
        }
    }

    #[test]
    fn dense_diverse_has_constant_degree_diverse_supports() {
        let w = dense_diverse(128, 0.3, 0.3, 42);
        assert_basics(&w);
        let d0 = popcount_words(&w.s_bits[0..w.nb_words]);
        let mut equal_count = 0;
        for i in 1..w.n {
            let row = &w.s_bits[i * w.nb_words..(i + 1) * w.nb_words];
            assert_eq!(popcount_words(row), d0, "row {i} degree differs");
            if row == &w.s_bits[0..w.nb_words] {
                equal_count += 1;
            }
        }
        // It would be astronomically unlikely for any row to coincide with row 0.
        assert!(equal_count < w.n / 4, "supports look insufficiently random");
    }

    #[test]
    fn sparse_diverse_has_variable_degree() {
        let w = sparse_diverse(256, 0.3, 1.5, 0.3, 42);
        assert_basics(&w);
        let d0 = popcount_words(&w.s_bits[0..w.nb_words]);
        let mut found_diff = false;
        for i in 1..w.n {
            let row = &w.s_bits[i * w.nb_words..(i + 1) * w.nb_words];
            if popcount_words(row) != d0 {
                found_diff = true;
                break;
            }
        }
        assert!(found_diff, "all rows have the same degree");
    }

    #[test]
    fn block_local_supports_are_contiguous_cyclically() {
        let n = 128;
        let w = block_local(n, 0.25, 0.3, 42);
        assert_basics(&w);
        let d = ((0.25 * n as f64).round()) as usize;
        for i in 0..w.n {
            let row = &w.s_bits[i * w.nb_words..(i + 1) * w.nb_words];
            assert_eq!(popcount_words(row) as usize, d);

            // Find any set bit, then walk d positions forward, all set;
            // and one past the block (one before too) must be unset.
            let mut set_positions = Vec::with_capacity(d);
            for b in 0..n {
                if (row[b / 64] >> (b % 64)) & 1 == 1 {
                    set_positions.push(b);
                }
            }
            assert_eq!(set_positions.len(), d);
            // Cyclically contiguous: there is some rotation under which the
            // positions are 0,1,...,d-1.
            let start = set_positions[0];
            let normalised: Vec<usize> = set_positions
                .iter()
                .map(|&p| (p + n - start) % n)
                .collect();
            let mut sorted = normalised.clone();
            sorted.sort_unstable();
            // After rotation the set should be exactly {0..d-1} OR a wrap
            // case where it splits into [0..a] + [b..n-1]. We test by
            // computing the "circular gap": exactly one gap of size n-d.
            let mut all = set_positions.clone();
            all.sort_unstable();
            let mut max_gap = 0usize;
            for k in 0..d {
                let cur = all[k];
                let nxt = all[(k + 1) % d];
                let gap = if nxt > cur {
                    nxt - cur
                } else {
                    nxt + n - cur
                };
                if gap > max_gap {
                    max_gap = gap;
                }
            }
            // The largest gap between set bits, cyclically, must be (n - d) + 1.
            assert_eq!(max_gap, n - d + 1, "row {i}: not cyclically contiguous");
            let _ = sorted;
        }
    }

    #[test]
    fn profile_enum_round_trip() {
        for p in Profile::ALL {
            let w = p.build(64, 0.3, 0.3, 7);
            assert_basics(&w);
            assert!(w.label.starts_with(p.name()));
        }
    }

    #[test]
    fn determinism_same_seed_same_workload() {
        for p in Profile::ALL {
            let a = p.build(64, 0.25, 0.3, 99);
            let b = p.build(64, 0.25, 0.3, 99);
            assert_eq!(a.s_bits, b.s_bits);
            assert_eq!(a.t_bits, b.t_bits);
            assert_eq!(a.col_perm_s, b.col_perm_s);
            assert_eq!(a.col_perm_t, b.col_perm_t);
        }
    }
}
