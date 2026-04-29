use crate::core::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

// Note: legacy matrix-based router removed — fused pipeline is the sole implementation.

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

    // Fused T lookups: t_start[col] = offsets_t[col], t_len[col] = ones_t[col]
    // Output column `col` directly represents target `col`, with load ∝ capacity.
    let mut t_start = vec![0usize; n];
    let mut t_len = vec![0usize; n];
    for col in 0..n {
        t_start[col] = offsets_t[col];
        t_len[col] = ones_t[col];
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

