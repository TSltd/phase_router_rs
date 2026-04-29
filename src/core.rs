use crate::bitops::*;
use rayon::prelude::*;

pub fn compute_offsets(bits: &[u64], n: usize, nb_words: usize) -> Vec<usize> {
    let mut offsets = vec![0usize; n];

    for i in 1..n {
        let prev = &bits[(i - 1) * nb_words..i * nb_words];
        let rs: usize = prev.iter().map(|w| w.count_ones() as usize).sum();
        offsets[i] = (offsets[i - 1] + rs) % n;
    }

    offsets
}

pub fn compute_row_ones(bits: &[u64], n: usize, nb_words: usize) -> Vec<usize> {
    (0..n)
        .map(|i| {
            bits[i * nb_words..(i + 1) * nb_words]
                .iter()
                .map(|w| w.count_ones() as usize)
                .sum()
        })
        .collect()
}

pub fn compute_inverse_perm(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

/// Check if `val` lies in the cyclic range [start, start+len) mod n.
#[inline(always)]
pub fn in_cyclic_range(val: usize, start: usize, len: usize, n: usize) -> bool {
    if len == 0 {
        return false;
    }
    if len >= n {
        return true;
    }
    let end = start + len;
    if end <= n {
        val >= start && val < end
    } else {
        val >= start || val < end - n
    }
}

// --- Legacy matrix-based functions (kept for benchmarking comparison) ---

pub fn build_s_final(
    s_bits: &[u64],
    offsets: &[usize],
    col_perm: &[usize],
    n: usize,
    nb_words: usize,
) -> Vec<u64> {
    let mut out = vec![0u64; n * nb_words];

    out.par_chunks_mut(nb_words)
        .enumerate()
        .for_each_init(
            || vec![0u64; nb_words],
            |rotated, (i, dst_row)| {
                let ones: usize = (0..nb_words)
                    .map(|w| s_bits[i * nb_words + w].count_ones() as usize)
                    .sum();

                fill_rotated_bits(rotated, n, offsets[i], ones);
                permute_columns_bits(rotated, dst_row, col_perm, n);
            },
        );

    out
}

pub fn build_t_final(
    t_bits: &[u64],
    offsets: &[usize],
    col_perm: &[usize],
    n: usize,
    nb_words: usize,
) -> Vec<u64> {
    // Phase 1: Build all permuted rows in parallel
    let mut t_permuted = vec![0u64; n * nb_words];

    t_permuted
        .par_chunks_mut(nb_words)
        .enumerate()
        .for_each_init(
            || vec![0u64; nb_words],
            |rotated, (i, dst)| {
                let ones: usize = (0..nb_words)
                    .map(|w| t_bits[i * nb_words + w].count_ones() as usize)
                    .sum();

                fill_rotated_bits(rotated, n, offsets[i], ones);
                permute_columns_bits(rotated, dst, col_perm, n);
            },
        );

    // Phase 2: Parallel transpose (gather pattern)
    let mut out = vec![0u64; n * nb_words];

    out.par_chunks_mut(nb_words)
        .enumerate()
        .for_each(|(j, dst_row)| {
            let j_word = j >> 6;
            let j_bit = j & 63;

            for i in 0..n {
                if (t_permuted[i * nb_words + j_word] >> j_bit) & 1 == 1 {
                    let dst_col = n - 1 - i;
                    let dw = dst_col >> 6;
                    let db = dst_col & 63;
                    dst_row[dw] |= 1u64 << db;
                }
            }
        });

    out
}
