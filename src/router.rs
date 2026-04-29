use crate::core::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Fully fused phase router — no intermediate matrices.
///
/// Uses O(n) precomputation, then a single parallel pass with:
/// - conditional-increment cyclic iteration (no modulo)
/// - branchless cyclic range check
/// - reservoir sampling (O(k) memory per row, no candidates Vec)
pub fn phase_router(
    s_bits: &[u64],
    t_bits: &[u64],
    n: usize,
    nb_words: usize,
    k: usize,
    col_perm_s: &[usize],
    col_perm_t: &[usize],
    seed: u64,
) -> Vec<i32> {
    // Precompute O(n) arrays
    let offsets_s = compute_offsets(s_bits, n, nb_words);
    let offsets_t = compute_offsets(t_bits, n, nb_words);
    let ones_s = compute_row_ones(s_bits, n, nb_words);
    let ones_t = compute_row_ones(t_bits, n, nb_words);
    let inv_perm_s = compute_inverse_perm(col_perm_s);

    // Fused T lookups: t_start[col] = offsets_t[n-1-col], t_len[col] = ones_t[n-1-col]
    // Eliminates subtraction + indirection in the hot loop
    let mut t_start = vec![0usize; n];
    let mut t_len = vec![0usize; n];
    for col in 0..n {
        let ti = n - 1 - col;
        t_start[col] = offsets_t[ti];
        t_len[col] = ones_t[ti];
    }

    let mut routes = vec![-1i32; n * k];

    routes
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(j, row_out)| {
            let s_start = offsets_s[j];
            let s_len = ones_s[j];
            let p_t = col_perm_t[j];

            // Reservoir sampling: maintain k best candidates in-place
            let mut rng = ChaCha8Rng::seed_from_u64(seed + j as u64);
            let mut count = 0usize; // total candidates seen

            // Cyclic iteration without modulo
            let mut p = s_start;

            for _ in 0..s_len {
                let col = inv_perm_s[p];

                // Branchless cyclic range check
                let start = t_start[col];
                let len = t_len[col];
                let mut d = p_t.wrapping_sub(start);
                if p_t < start {
                    d = d.wrapping_add(n);
                }
                // d < len handles len==0 naturally (d is always >= 0, so 0 < 0 is false)
                if d < len {
                    // Reservoir sampling (Vitter's Algorithm R)
                    if count < k {
                        row_out[count] = col as i32;
                    } else {
                        let r = rng.gen_range(0..=count);
                        if r < k {
                            row_out[r] = col as i32;
                        }
                    }
                    count += 1;
                }

                // Conditional increment (replaces % n)
                p += 1;
                if p == n {
                    p = 0;
                }
            }
        });

    routes
}

/// Legacy matrix-based phase router (kept for benchmarking comparison).
pub fn phase_router_legacy(
    s_bits: &[u64],
    t_bits: &[u64],
    n: usize,
    nb_words: usize,
    k: usize,
    col_perm_s: &[usize],
    col_perm_t: &[usize],
    seed: u64,
) -> Vec<i32> {
    let offsets_s = compute_offsets(s_bits, n, nb_words);
    let offsets_t = compute_offsets(t_bits, n, nb_words);

    let s_final = build_s_final(s_bits, &offsets_s, col_perm_s, n, nb_words);
    let t_final = build_t_final(t_bits, &offsets_t, col_perm_t, n, nb_words);

    let mut routes = vec![-1i32; n * k];

    routes
        .par_chunks_mut(k)
        .enumerate()
        .for_each_init(
            || Vec::with_capacity(n),
            |candidates, (i, row_out)| {
                candidates.clear();

                let srow = &s_final[i * nb_words..(i + 1) * nb_words];
                let trow = &t_final[i * nb_words..(i + 1) * nb_words];

                for w in 0..nb_words {
                    let mut m = srow[w] & trow[w];

                    while m != 0 {
                        let b = m.trailing_zeros() as usize;
                        candidates.push(w * 64 + b);
                        m &= m - 1;
                    }
                }

                let mut rng = ChaCha8Rng::seed_from_u64(seed + i as u64);
                let take = k.min(candidates.len());
                for j in 0..take {
                    let remaining = candidates.len() - j;
                    if remaining > 1 {
                        let idx = j + rng.gen_range(0..remaining);
                        candidates.swap(j, idx);
                    }
                    row_out[j] = candidates[j] as i32;
                }
            },
        );

    routes
}
