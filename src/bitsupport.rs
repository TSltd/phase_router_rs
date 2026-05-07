//! Bit-support primitives: `rank1`, `select1`, `select_in_word`, `SupportView`.
//!
//! These are the in-tree replacements for the `inv_perm_s` global indirection
//! used by the original phase router. They expose a row's bitvector as an
//! intrinsic occupancy coordinate space, where:
//!
//! - `rank1(words, p)`  maps physical position → occupancy index
//! - `select1(words, k)` maps occupancy index  → physical position
//!
//! `select_in_word` uses BMI2 `PDEP` when available (with one cached runtime
//! check) and falls back to a portable `tzcnt`-loop otherwise. A faster
//! Vigna-style broadword select is intentionally *not* implemented here yet:
//! the PDEP path on x86_64 already gives O(1) word-local select, and the
//! portable fallback is small and obviously correct, which is what we want
//! while we are still validating semantics.

use std::sync::atomic::{AtomicU8, Ordering};

// ── select_in_word ───────────────────────────────────────────────────────

/// Returns the bit position (0..64) of the `k`-th set bit in `x` (0-indexed).
/// Returns 64 if `x` has fewer than `k+1` set bits.
#[inline]
pub fn select_in_word(x: u64, k: u32) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_bmi2() {
            // SAFETY: gated by runtime CPUID check.
            return unsafe { select_in_word_bmi2(x, k) };
        }
    }
    select_in_word_portable(x, k)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "bmi2")]
unsafe fn select_in_word_bmi2(x: u64, k: u32) -> u32 {
    use std::arch::x86_64::_pdep_u64;
    // PDEP deposits a single 1 bit into x's set positions; tzcnt then
    // gives the physical bit position of the k-th set bit.
    let bit = _pdep_u64(1u64 << k, x);
    bit.trailing_zeros()
}

/// Portable fallback: O(k) but trivially correct.
#[inline]
fn select_in_word_portable(mut x: u64, mut k: u32) -> u32 {
    while x != 0 {
        if k == 0 {
            return x.trailing_zeros();
        }
        x &= x - 1;
        k -= 1;
    }
    64
}

// Cache the BMI2 detection so the hot path is one relaxed atomic load.
#[cfg(target_arch = "x86_64")]
fn has_bmi2() -> bool {
    static CACHE: AtomicU8 = AtomicU8::new(0); // 0 = unknown, 1 = no, 2 = yes
    let v = CACHE.load(Ordering::Relaxed);
    if v != 0 {
        return v == 2;
    }
    let detected = std::is_x86_feature_detected!("bmi2");
    CACHE.store(if detected { 2 } else { 1 }, Ordering::Relaxed);
    detected
}

// ── multi-word rank/select ───────────────────────────────────────────────

/// Total set-bit count of a packed bit slice.
#[inline]
pub fn popcount_words(words: &[u64]) -> u32 {
    words.iter().map(|w| w.count_ones()).sum()
}

/// `rank1(words, p)` = number of set bits in positions `[0, p)`.
///
/// Linear in `p / 64`. For small `nb_words` (the MoE regime) this fits in
/// a handful of cycles per call; if larger supports become routine, swap
/// in a word-prefix popcount index (Phase 4 in `dev/plan.md`).
#[inline]
pub fn rank1(words: &[u64], pos: usize) -> u32 {
    let full_words = pos / 64;
    let rem_bits = pos % 64;
    let mut count: u32 = 0;
    let limit = full_words.min(words.len());
    for w in &words[..limit] {
        count += w.count_ones();
    }
    if rem_bits > 0 && full_words < words.len() {
        let mask = (1u64 << rem_bits) - 1;
        count += (words[full_words] & mask).count_ones();
    }
    count
}

/// `select1(words, k)` = position of the `k`-th set bit (0-indexed).
///
/// Panics if `k` ≥ popcount of `words`. The router never calls this with
/// out-of-range `k` because `r' = (...) mod d_j` always satisfies
/// `r' < d_j = popcount`.
#[inline]
pub fn select1(words: &[u64], mut k: u32) -> usize {
    for (w_idx, &w) in words.iter().enumerate() {
        let pc = w.count_ones();
        if pc > k {
            return w_idx * 64 + select_in_word(w, k) as usize;
        }
        k -= pc;
    }
    panic!(
        "select1: k={} out of range for words with popcount {}",
        k,
        popcount_words(words)
    );
}

// ── SupportView ──────────────────────────────────────────────────────────

/// A borrowed view over a row's packed bit support, exposing
/// occupancy-domain operations.
///
/// The `degree` is cached at construction so that hot loops can iterate
/// `0..degree` without repeated popcounts.
#[derive(Copy, Clone)]
pub struct SupportView<'a> {
    pub words: &'a [u64],
    pub degree: u32,
}

impl<'a> SupportView<'a> {
    #[inline]
    pub fn new(words: &'a [u64]) -> Self {
        Self {
            words,
            degree: popcount_words(words),
        }
    }

    /// Construct from a precomputed degree (avoids redundant popcount).
    #[inline]
    pub fn with_degree(words: &'a [u64], degree: u32) -> Self {
        Self { words, degree }
    }

    /// Occupancy → physical.
    #[inline]
    pub fn select(&self, k: u32) -> usize {
        debug_assert!(k < self.degree, "select out of range");
        select1(self.words, k)
    }

    /// Physical → occupancy (count of 1s strictly before `pos`).
    #[inline]
    pub fn rank(&self, pos: usize) -> u32 {
        rank1(self.words, pos)
    }
}

// ── affine helpers (consumed by router_rs.rs) ───────────────────────────

/// Splitmix64 — fast deterministic mixing. Used for per-row phase / multiplier
/// derivation so that rank-space transport does not need a global permutation.
#[inline]
pub fn mix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[inline]
pub fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Returns an `a` in `[1, d)` with `gcd(a, d) == 1`. Deterministic given `seed`.
///
/// For `d <= 1` returns 1 (the identity multiplier). The expected number of
/// trials is `d / phi(d) <= O(log log d)` so this terminates quickly.
#[inline]
pub fn coprime_multiplier(d: usize, seed: u64) -> usize {
    if d <= 1 {
        return 1;
    }
    let mut s = seed;
    loop {
        s = mix64(s);
        let a = (s as usize) % d;
        if a >= 1 && gcd(a, d) == 1 {
            return a;
        }
    }
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn select_in_word_handpicked() {
        let x: u64 = (1 << 0) | (1 << 5) | (1 << 7) | (1 << 33) | (1 << 63);
        assert_eq!(select_in_word(x, 0), 0);
        assert_eq!(select_in_word(x, 1), 5);
        assert_eq!(select_in_word(x, 2), 7);
        assert_eq!(select_in_word(x, 3), 33);
        assert_eq!(select_in_word(x, 4), 63);
    }

    #[test]
    fn select_in_word_zero_returns_64() {
        assert_eq!(select_in_word_portable(0, 0), 64);
    }

    #[test]
    fn select_in_word_portable_matches_dispatcher() {
        // Both code paths must produce identical results on every bit.
        let mut rng = ChaCha8Rng::seed_from_u64(0xA1A2A3A4);
        for _ in 0..512 {
            let x: u64 = rng.gen();
            let pc = x.count_ones();
            for k in 0..pc {
                let a = select_in_word(x, k);
                let b = select_in_word_portable(x, k);
                assert_eq!(a, b, "mismatch on x={:016x} k={}", x, k);
            }
        }
    }

    #[test]
    fn rank1_handpicked() {
        // bits set at 0, 5, 7, 33, 63
        let words = vec![(1u64 << 0) | (1 << 5) | (1 << 7) | (1 << 33) | (1 << 63)];
        assert_eq!(rank1(&words, 0), 0);
        assert_eq!(rank1(&words, 1), 1);
        assert_eq!(rank1(&words, 5), 1);
        assert_eq!(rank1(&words, 6), 2);
        assert_eq!(rank1(&words, 8), 3);
        assert_eq!(rank1(&words, 33), 3);
        assert_eq!(rank1(&words, 34), 4);
        assert_eq!(rank1(&words, 64), 5);
    }

    #[test]
    fn rank_select_inverse_random() {
        // Core invariant: rank1(select1(x, k)) == k for every k < popcount.
        let mut rng = ChaCha8Rng::seed_from_u64(0xDEADBEEF);
        for trial in 0..32 {
            let nb = 1 + (trial as usize % 16);
            let words: Vec<u64> = (0..nb).map(|_| rng.gen()).collect();
            let pc = popcount_words(&words);
            for k in 0..pc {
                let p = select1(&words, k);
                assert_eq!(rank1(&words, p), k, "k={} p={}", k, p);
                // bit at p is actually set
                assert!(words[p / 64] & (1u64 << (p % 64)) != 0);
            }
        }
    }

    #[test]
    fn select_strictly_increasing() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x1234);
        let words: Vec<u64> = (0..8).map(|_| rng.gen()).collect();
        let pc = popcount_words(&words);
        let mut prev: i64 = -1;
        for k in 0..pc {
            let p = select1(&words, k) as i64;
            assert!(p > prev, "select must be strictly increasing");
            prev = p;
        }
    }

    #[test]
    fn boundary_all_zero_all_one() {
        assert_eq!(rank1(&[], 0), 0);
        assert_eq!(rank1(&[0u64; 4], 256), 0);
        let all = vec![!0u64; 2];
        assert_eq!(popcount_words(&all), 128);
        for k in 0..128 {
            assert_eq!(select1(&all, k), k as usize);
        }
        let single = vec![1u64 << 42, 0u64];
        assert_eq!(select1(&single, 0), 42);
    }

    #[test]
    fn support_view_roundtrip() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let words: Vec<u64> = (0..6).map(|_| rng.gen()).collect();
        let view = SupportView::new(&words);
        for k in 0..view.degree {
            let p = view.select(k);
            assert_eq!(view.rank(p), k);
        }
    }

    #[test]
    fn coprime_multiplier_is_coprime() {
        for d in 1..200usize {
            let a = coprime_multiplier(d, 0x123456);
            assert_eq!(gcd(a, d.max(1)), 1, "a={} d={}", a, d);
            if d > 1 {
                assert!(a >= 1 && a < d);
            }
        }
    }

    #[test]
    fn coprime_multiplier_is_deterministic() {
        for d in [3usize, 16, 64, 97, 128, 1000] {
            let a1 = coprime_multiplier(d, 42);
            let a2 = coprime_multiplier(d, 42);
            assert_eq!(a1, a2);
        }
    }
}
