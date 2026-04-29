use crate::core::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Fully fused phase router — no intermediate matrices.
///
/// For each row j, directly computes which columns satisfy both
/// S'[j,col]=1 and T'[j,col]=1 using O(1) arithmetic checks,
/// then selects up to k candidates.
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

    let mut routes = vec![-1i32; n * k];

    routes
        .par_chunks_mut(k)
        .enumerate()
        .for_each_init(
            || Vec::with_capacity(n),
            |candidates, (j, row_out)| {
                candidates.clear();

                let s_start = offsets_s[j];
                let s_len = ones_s[j];
                let p_t = col_perm_t[j];

                // Iterate over the s_len positions where S'[j,*]=1
                for idx in 0..s_len {
                    let p = (s_start + idx) % n;
                    let col = inv_perm_s[p];
                    let ti = n - 1 - col;

                    // O(1) check: is T'[j,col]=1?
                    if in_cyclic_range(p_t, offsets_t[ti], ones_t[ti], n) {
                        candidates.push(col);
                    }
                }

                // Partial Fisher-Yates: only shuffle k elements
                let mut rng = ChaCha8Rng::seed_from_u64(seed + j as u64);
                let take = k.min(candidates.len());
                for i in 0..take {
                    let remaining = candidates.len() - i;
                    if remaining > 1 {
                        let idx = i + rng.gen_range(0..remaining);
                        candidates.swap(i, idx);
                    }
                    row_out[i] = candidates[i] as i32;
                }
            },
        );

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
