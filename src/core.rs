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
        .for_each(|(i, dst_row)| {
            let mut ones = 0;
            for w in 0..nb_words {
                ones += s_bits[i * nb_words + w].count_ones() as usize;
            }

            let mut temp = vec![0u64; nb_words];

            let full = ones / 64;
            let rem = ones % 64;

            for w in 0..full {
                temp[w] = !0;
            }
            if rem > 0 {
                temp[full] = (1u64 << rem) - 1;
            }

            let mut rotated = vec![0u64; nb_words];
            rotate_bits_full(&temp, n, nb_words, offsets[i], &mut rotated);

            permute_columns_bits(&rotated, dst_row, col_perm, n);
        });

    out
}

pub fn build_t_final(
    t_bits: &[u64],
    offsets: &[usize],
    col_perm: &[usize],
    n: usize,
    nb_words: usize,
) -> Vec<u64> {
    let mut out = vec![0u64; n * nb_words];

    for i in 0..n {
        let mut ones = 0;
        for w in 0..nb_words {
            ones += t_bits[i * nb_words + w].count_ones() as usize;
        }

        let mut temp = vec![0u64; nb_words];

        let full = ones / 64;
        let rem = ones % 64;

        for w in 0..full {
            temp[w] = !0;
        }
        if rem > 0 {
            temp[full] = (1u64 << rem) - 1;
        }

        let mut rotated = vec![0u64; nb_words];
        rotate_bits_full(&temp, n, nb_words, offsets[i], &mut rotated);

        let mut permuted = vec![0u64; nb_words];
        permute_columns_bits(&rotated, &mut permuted, col_perm, n);

        for w in 0..nb_words {
            let mut m = permuted[w];

            while m != 0 {
                let b = m.trailing_zeros() as usize;
                let col = w * 64 + b;

                if col < n {
                    let dst_row = col;
                    let dst_col = n - 1 - i;

                    let dw = dst_col >> 6;
                    let db = dst_col & 63;

                    out[dst_row * nb_words + dw] |= 1 << db;
                }

                m &= m - 1;
            }
        }
    }

    out
}