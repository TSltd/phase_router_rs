use crate::core::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

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
    let offsets_s = compute_offsets(s_bits, n, nb_words);
    let offsets_t = compute_offsets(t_bits, n, nb_words);

    let s_final = build_s_final(s_bits, &offsets_s, col_perm_s, n, nb_words);
    let t_final = build_t_final(t_bits, &offsets_t, col_perm_t, n, nb_words);

    let mut routes = vec![-1i32; n * k];

    routes
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(i, row_out)| {
            let srow = &s_final[i * nb_words..(i + 1) * nb_words];
            let trow = &t_final[i * nb_words..(i + 1) * nb_words];

            let mut candidates = Vec::new();

            for w in 0..nb_words {
                let mut m = srow[w] & trow[w];

                while m != 0 {
                    let b = m.trailing_zeros() as usize;
                    candidates.push(w * 64 + b);
                    m &= m - 1;
                }
            }

            let mut rng = ChaCha8Rng::seed_from_u64(seed + i as u64);
            candidates.shuffle(&mut rng);

            for (idx, &c) in candidates.iter().take(k).enumerate() {
                row_out[idx] = c as i32;
            }
        });

    routes
}