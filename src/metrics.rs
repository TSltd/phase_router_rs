//! Diagnostic metrics for routing quality.
//!
//! These are deliberately naive (no FFT crate, no SIMD). We are looking for
//! *signals* that distinguish routing regimes — low-discrepancy distributed
//! occupancy allocation vs. support-driven random sampling vs. uniform hashing
//! — not for high-throughput online monitoring. Asymptotically:
//!
//! - [`discrepancy`]                       — `O(n)`
//! - [`power_spectrum`]                    — `O(n²)` (direct DFT)
//! - [`low_freq_energy_share`]             — `O(n²)` (calls `power_spectrum`)
//! - [`pairwise_candidate_overlap`]        — `O(P · max_row_size)` for `P` sampled pairs
//! - [`support_autocorrelation_preservation`] — `O(n · k)`
//!
//! All metric functions are **deterministic** given their inputs (no internal
//! randomness except for `pairwise_candidate_overlap`, which seeds a
//! SplitMix64 from the caller-provided seed).

use crate::bitsupport::mix64;

// ── discrepancy ──────────────────────────────────────────────────────────

/// Summary of how loads deviate from the expected uniform value.
#[derive(Debug, Clone, Copy)]
pub struct Discrepancy {
    /// `max_i |loads[i] - expected|`.
    pub max_abs_dev: f64,
    /// `(max_i loads[i] - min_i loads[i]) / mean(loads)`.
    pub range_over_mean: f64,
    /// `std(loads) / mean(loads)`. Matches the CV used in `quality_probe`.
    pub cv: f64,
    /// `max(loads) / mean(loads)`. Hot-spot ratio.
    pub max_over_mean: f64,
}

/// Computes load-discrepancy summary statistics relative to `expected`.
///
/// `expected` is the uniform-distribution target (`n * k / n = k` for the
/// canonical setup). `loads.len() == n`, integer load counts.
pub fn discrepancy(loads: &[u32], expected: f64) -> Discrepancy {
    if loads.is_empty() {
        return Discrepancy {
            max_abs_dev: 0.0,
            range_over_mean: 0.0,
            cv: 0.0,
            max_over_mean: 0.0,
        };
    }
    let n = loads.len() as f64;
    let mut sum = 0.0f64;
    let mut sumsq = 0.0f64;
    let mut max_l = u32::MIN;
    let mut min_l = u32::MAX;
    let mut max_abs_dev = 0.0f64;
    for &l in loads {
        let lf = l as f64;
        sum += lf;
        sumsq += lf * lf;
        if l > max_l {
            max_l = l;
        }
        if l < min_l {
            min_l = l;
        }
        let dev = (lf - expected).abs();
        if dev > max_abs_dev {
            max_abs_dev = dev;
        }
    }
    let mean = sum / n;
    let var = (sumsq / n) - mean * mean;
    let std = if var > 0.0 { var.sqrt() } else { 0.0 };
    let range = (max_l - min_l) as f64;
    let range_over_mean = if mean > 0.0 { range / mean } else { 0.0 };
    let cv = if mean > 0.0 { std / mean } else { 0.0 };
    let max_over_mean = if mean > 0.0 { max_l as f64 / mean } else { 0.0 };
    Discrepancy {
        max_abs_dev,
        range_over_mean,
        cv,
        max_over_mean,
    }
}

// ── DFT power spectrum ───────────────────────────────────────────────────

/// Direct `O(n²)` DFT of a real signal. Returns `n` magnitude-squared values.
///
/// The DC bin (index 0) is included; consumers may want to drop it.
pub fn power_spectrum(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f64;
    let two_pi_over_n = 2.0 * std::f64::consts::PI / nf;
    let mut out = vec![0.0f64; n];
    for k in 0..n {
        let theta_k = two_pi_over_n * k as f64;
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (t, &v) in values.iter().enumerate() {
            let theta = theta_k * t as f64;
            re += v * theta.cos();
            im -= v * theta.sin();
        }
        out[k] = re * re + im * im;
    }
    out
}

/// Fraction of total non-DC spectral energy concentrated in the lowest
/// `low_band` frequency bins (excluding DC). High share = strong large-scale
/// structure; low share = white-noise-like.
///
/// `low_band` is clamped to `[1, n/2]`.
pub fn low_freq_energy_share(values: &[f64], low_band: usize) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    // Demean to suppress DC and the floating-point noise it produces in
    // non-DC bins of the O(n²) DFT. After demeaning, a truly constant
    // signal becomes all-zero → power_spectrum returns all zeros →
    // total == 0 → we return 0.0 below. This does not affect the
    // *shape* of the non-DC spectrum for any other input, since the DFT
    // is linear and demeaning only zeroes the k=0 bin analytically.
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let demeaned: Vec<f64> = values.iter().map(|&v| v - mean).collect();
    let spec = power_spectrum(&demeaned);
    // Use [1, n/2] as the meaningful spectrum (Nyquist-symmetric for real input).
    let nyq = n / 2;
    let band = low_band.clamp(1, nyq.max(1));
    let mut low = 0.0f64;
    let mut total = 0.0f64;
    for k in 1..=nyq {
        let p = spec[k];
        total += p;
        if k <= band {
            low += p;
        }
    }
    if total > 0.0 {
        low / total
    } else {
        0.0
    }
}

// ── pairwise candidate overlap ──────────────────────────────────────────

/// Summary statistics for pairwise overlap between row candidate sets.
#[derive(Debug, Clone, Copy)]
pub struct OverlapStats {
    pub mean_jaccard: f64,
    pub max_jaccard: f64,
    pub mean_intersection_over_union: f64,
    pub samples: usize,
}

/// Sample `sample_pairs` random pairs of rows and compute Jaccard overlap of
/// their emitted candidate sets (treating `routes_per_row[j]` as a set; -1
/// entries are ignored). Returns `(mean, max, mean_iou, samples)`.
///
/// On the `dense_uniform` workload this metric should approach 1.0 for
/// intrinsic-decode kernels (rows visit identical candidate sets) and stay
/// well below 1.0 for the original `phase_router` (different row offsets
/// produce different per-row windows of the global walk).
pub fn pairwise_candidate_overlap(
    routes_per_row: &[Vec<i32>],
    sample_pairs: usize,
    seed: u64,
) -> OverlapStats {
    let n = routes_per_row.len();
    if n < 2 || sample_pairs == 0 {
        return OverlapStats {
            mean_jaccard: 0.0,
            max_jaccard: 0.0,
            mean_intersection_over_union: 0.0,
            samples: 0,
        };
    }

    let mut state = seed;
    let mut sum_j = 0.0f64;
    let mut max_j = 0.0f64;
    let mut taken = 0usize;

    for _ in 0..sample_pairs {
        state = mix64(state);
        let i = (state as usize) % n;
        state = mix64(state);
        let mut j = (state as usize) % n;
        if j == i {
            j = (j + 1) % n;
        }

        // Build sets (small Vec; routes are at most k entries).
        let mut a: Vec<i32> = routes_per_row[i].iter().copied().filter(|&c| c >= 0).collect();
        let mut b: Vec<i32> = routes_per_row[j].iter().copied().filter(|&c| c >= 0).collect();
        a.sort_unstable();
        a.dedup();
        b.sort_unstable();
        b.dedup();
        if a.is_empty() && b.is_empty() {
            continue;
        }

        // |A ∩ B|
        let mut inter = 0usize;
        let (mut ia, mut ib) = (0usize, 0usize);
        while ia < a.len() && ib < b.len() {
            if a[ia] == b[ib] {
                inter += 1;
                ia += 1;
                ib += 1;
            } else if a[ia] < b[ib] {
                ia += 1;
            } else {
                ib += 1;
            }
        }
        let union = a.len() + b.len() - inter;
        if union == 0 {
            continue;
        }
        let jacc = inter as f64 / union as f64;
        sum_j += jacc;
        if jacc > max_j {
            max_j = jacc;
        }
        taken += 1;
    }

    let mean_j = if taken > 0 { sum_j / taken as f64 } else { 0.0 };
    OverlapStats {
        mean_jaccard: mean_j,
        max_jaccard: max_j,
        mean_intersection_over_union: mean_j, // alias kept for clarity
        samples: taken,
    }
}

// ── support autocorrelation preservation ────────────────────────────────

/// Fraction of routed columns that lie in the *same row's* original support.
///
/// All in-tree routers should report 1.0 here by construction; this is a
/// regression sentinel for any future router that emits columns outside the
/// row support (which would silently break the model semantics).
pub fn support_validity(
    routes_per_row: &[Vec<i32>],
    s_bits: &[u64],
    nb_words: usize,
) -> f64 {
    let n = routes_per_row.len();
    if n == 0 {
        return 0.0;
    }
    let mut total = 0usize;
    let mut valid = 0usize;
    for j in 0..n {
        let row = &s_bits[j * nb_words..(j + 1) * nb_words];
        for &c in &routes_per_row[j] {
            if c < 0 {
                continue;
            }
            total += 1;
            let cu = c as usize;
            let bit_set = (row[cu / 64] >> (cu % 64)) & 1 == 1;
            if bit_set {
                valid += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        valid as f64 / total as f64
    }
}

/// Mean fraction, over all rows `j`, of routed columns whose adjacent
/// position `(c+1) mod n` is also in row `j`'s support. Approximates
/// "did the router preserve the local-block character of the source"; for a
/// `block_local` workload, kernels that walk supports contiguously will score
/// near 1.0, kernels that scatter will score close to the row density.
pub fn support_autocorrelation_preservation(
    routes_per_row: &[Vec<i32>],
    s_bits: &[u64],
    n: usize,
    nb_words: usize,
) -> f64 {
    if n == 0 || routes_per_row.is_empty() {
        return 0.0;
    }
    let mut total = 0usize;
    let mut neighbours = 0usize;
    for (j, row_routes) in routes_per_row.iter().enumerate() {
        let row = &s_bits[j * nb_words..(j + 1) * nb_words];
        for &c in row_routes {
            if c < 0 {
                continue;
            }
            total += 1;
            let cu = (c as usize + 1) % n;
            let bit_set = (row[cu / 64] >> (cu % 64)) & 1 == 1;
            if bit_set {
                neighbours += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        neighbours as f64 / total as f64
    }
}

// ── helpers ──────────────────────────────────────────────────────────────

/// Build `routes_per_row` (`Vec<Vec<i32>>`) from a flat `Vec<i32>` of length
/// `n*k` produced by any router in this crate.
pub fn split_routes(routes: &[i32], n: usize, k: usize) -> Vec<Vec<i32>> {
    debug_assert_eq!(routes.len(), n * k);
    routes.chunks(k).map(|c| c.to_vec()).collect()
}

/// Per-target loads (count of routed entries per column in `[0, n)`).
pub fn loads_per_target(routes: &[i32], n: usize) -> Vec<u32> {
    let mut l = vec![0u32; n];
    for &c in routes {
        if c >= 0 && (c as usize) < n {
            l[c as usize] += 1;
        }
    }
    l
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrepancy_uniform_load() {
        let loads = vec![4u32; 32];
        let d = discrepancy(&loads, 4.0);
        assert_eq!(d.max_abs_dev, 0.0);
        assert_eq!(d.range_over_mean, 0.0);
        assert_eq!(d.cv, 0.0);
        assert!((d.max_over_mean - 1.0).abs() < 1e-12);
    }

    #[test]
    fn discrepancy_one_hot_spike() {
        let mut loads = vec![1u32; 16];
        loads[0] = 16; // big spike
        let mean = (16 + 15) as f64 / 16.0;
        let d = discrepancy(&loads, 1.0);
        assert!(d.max_over_mean > 1.0);
        assert!(d.cv > 0.0);
        assert!((d.max_abs_dev - 15.0).abs() < 1e-12);
        let _ = mean;
    }

    #[test]
    fn power_spectrum_dc_only_for_constant() {
        let v = vec![1.0f64; 16];
        let s = power_spectrum(&v);
        assert!(s[0] > 0.0);
        for k in 1..16 {
            assert!(s[k] < 1e-9, "non-DC bin {k} not zero: {}", s[k]);
        }
    }

    #[test]
    fn power_spectrum_pure_tone_concentrated() {
        // x[t] = cos(2π·3·t/16) → energy at k=3 (and k=13 by symmetry).
        let n = 16;
        let v: Vec<f64> = (0..n)
            .map(|t| (2.0 * std::f64::consts::PI * 3.0 * t as f64 / n as f64).cos())
            .collect();
        let s = power_spectrum(&v);
        let total: f64 = s[1..].iter().sum();
        let target = s[3] + s[n - 3];
        assert!(target / total > 0.99, "tone not concentrated: {target}/{total}");
    }

    #[test]
    fn low_freq_share_constant_signal() {
        // Constant signal: all energy in DC; non-DC band is zero → share = 0/0 → 0.
        let v = vec![3.0f64; 32];
        let s = low_freq_energy_share(&v, 3);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn low_freq_share_low_tone_high() {
        let n = 64;
        let v: Vec<f64> = (0..n)
            .map(|t| (2.0 * std::f64::consts::PI * 1.0 * t as f64 / n as f64).cos())
            .collect();
        let s = low_freq_energy_share(&v, 3);
        assert!(s > 0.95, "expected near-1.0, got {s}");
    }

    #[test]
    fn pairwise_overlap_identical_rows_is_one() {
        let row = vec![0i32, 1, 2, 3];
        let routes = vec![row.clone(); 16];
        let r = pairwise_candidate_overlap(&routes, 32, 7);
        assert!(r.samples > 0);
        assert!((r.mean_jaccard - 1.0).abs() < 1e-12);
        assert!((r.max_jaccard - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pairwise_overlap_disjoint_rows_is_zero() {
        let routes: Vec<Vec<i32>> = (0..8).map(|j| vec![j as i32 * 4, j as i32 * 4 + 1]).collect();
        let r = pairwise_candidate_overlap(&routes, 32, 9);
        assert!(r.samples > 0);
        assert!(r.mean_jaccard < 1e-12);
        assert!(r.max_jaccard < 1e-12);
    }

    #[test]
    fn split_and_loads_round_trip() {
        let routes = vec![0i32, 1, 2, -1, 0, 0, 2, 3];
        let n = 4;
        let k = 2;
        let split = split_routes(&routes, n, k);
        assert_eq!(split.len(), 4);
        let l = loads_per_target(&routes, n);
        assert_eq!(l, vec![3, 1, 2, 1]);
    }

    #[test]
    fn support_validity_perfect_when_routed_inside_support() {
        let n = 8;
        let nb = 1;
        let mut s = vec![0u64; n * nb];
        for i in 0..n {
            for b in 0..4 {
                s[i] |= 1u64 << b;
            }
        }
        let routes_per_row: Vec<Vec<i32>> = (0..n).map(|_| vec![0, 1, 2, 3]).collect();
        let v = support_validity(&routes_per_row, &s, nb);
        assert!((v - 1.0).abs() < 1e-12);

        // One offending route at column 7 (not in support).
        let mut bad = routes_per_row.clone();
        bad[0][0] = 7;
        let v2 = support_validity(&bad, &s, nb);
        assert!(v2 < 1.0 && v2 > 0.9);
    }
}
