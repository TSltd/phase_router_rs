use crate::core::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Build a left-aligned uniform bit-packed matrix: every row has the same
/// `ones_per_row` low-bit ones, packed into `nb_words` u64s.
///
/// Equivalent to the Python helper `_build_t_bits` / `_build_s_bits_uniform`
/// when every row has the same ones count — but implemented in pure Rust so
/// the GIL is released for the entire call.
fn build_uniform_bits(ones_per_row: usize, n: usize, nb_words: usize) -> Vec<u64> {
    let ones = ones_per_row.min(n);
    let full = ones / 64;
    let tail = ones - full * 64;

    // Prototype row, then tile.
    let mut proto = vec![0u64; nb_words];
    for w in 0..full {
        proto[w] = !0u64;
    }
    if tail > 0 {
        proto[full] = (1u64 << tail) - 1;
    }

    let mut out = vec![0u64; n * nb_words];
    for row in 0..n {
        out[row * nb_words..(row + 1) * nb_words].copy_from_slice(&proto);
    }
    out
}

/// Fused uniform MoE dispatch: builds the (uniform) s_bits/t_bits internally,
/// runs the phase-routing kernel with k * oversample columns, then maps kernel
/// column indices to expert ids (via expert-band tiling) and dedupes the first
/// `k` unique experts per token in a single parallel pass.
///
/// This collapses what used to be four pure-Python operations:
///   1. `_build_t_bits` (uniform target capacity)
///   2. `_build_s_bits_uniform` (uniform source demand)
///   3. `phase_router_rs.phase_router_auto(...)`
///   4. per-row unique-pick Python loop
/// into a single GIL-released Rust call.
///
/// # Arguments
/// * `n_tokens`     — original token count (output rows)
/// * `n_experts`    — number of experts (output value range)
/// * `k`            — experts per token (output cols)
/// * `base_density` — density used for both source demand and per-expert capacity
///                    band width (matches `PhaseRouter.base_density`)
/// * `oversample`   — kernel runs with `k_kernel = max(k * oversample, k + 1)`
///                    columns before per-row dedup
/// * `seed`         — deterministic seed (drives permutations + reservoir RNG)
///
/// Returns a flat `n_tokens * k` `Vec<i32>` of expert ids in `[0, n_experts)`,
/// with `-1` for unfilled slots.
pub fn phase_router_uniform_dispatch(
    n_tokens: usize,
    n_experts: usize,
    k: usize,
    base_density: f64,
    oversample: usize,
    seed: u64,
) -> Vec<i32> {
    assert!(n_tokens > 0 && n_experts > 0 && k > 0);

    // Pad N up to a multiple of n_experts so the band tiling is clean.
    // Matches PhaseRouter._select in train/routers.py.
    let mut width = (n_tokens / n_experts).max(1);
    let mut n_pad = width * n_experts;
    if n_pad < n_tokens {
        n_pad = (n_tokens / n_experts + 1) * n_experts;
        width = n_pad / n_experts;
    }
    let nb_words = (n_pad + 63) / 64;

    // Uniform ones counts.
    let ones_e = ((base_density * width as f64).round() as usize)
        .max(1)
        .min(n_pad);
    let s_ones = ((base_density * n_pad as f64).round() as usize)
        .max(1)
        .min(n_pad);

    let t_bits = build_uniform_bits(ones_e, n_pad, nb_words);
    let s_bits = build_uniform_bits(s_ones, n_pad, nb_words);

    // Deterministic permutations.
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut col_perm_s: Vec<usize> = (0..n_pad).collect();
    let mut col_perm_t: Vec<usize> = (0..n_pad).collect();
    col_perm_s.shuffle(&mut rng);
    col_perm_t.shuffle(&mut rng);

    let k_kernel = (k.saturating_mul(oversample)).max(k + 1).min(n_pad);

    let routes = phase_router(
        &s_bits, &t_bits, n_pad, nb_words, k_kernel, &col_perm_s, &col_perm_t, seed,
    );

    // routes: (n_pad, k_kernel) — only the first n_tokens rows correspond to
    // real tokens. Convert each kernel column to an expert id and take the
    // first k unique ones per row.
    let mut out = vec![-1i32; n_tokens * k];
    let use_bitmask = n_experts <= 128;

    out.par_chunks_mut(k)
        .enumerate()
        .for_each(|(row, slot_out)| {
            let row_in = &routes[row * k_kernel..(row + 1) * k_kernel];
            let mut j = 0usize;

            if use_bitmask {
                let mut seen: u128 = 0;
                for &col in row_in {
                    if col < 0 {
                        continue;
                    }
                    let c = col as usize;
                    if c >= n_pad {
                        continue;
                    }
                    let e = c / width;
                    if e >= n_experts {
                        continue;
                    }
                    let bit = 1u128 << e;
                    if seen & bit != 0 {
                        continue;
                    }
                    seen |= bit;
                    slot_out[j] = e as i32;
                    j += 1;
                    if j == k {
                        break;
                    }
                }
            } else {
                let mut seen = vec![false; n_experts];
                for &col in row_in {
                    if col < 0 {
                        continue;
                    }
                    let c = col as usize;
                    if c >= n_pad {
                        continue;
                    }
                    let e = c / width;
                    if e >= n_experts || seen[e] {
                        continue;
                    }
                    seen[e] = true;
                    slot_out[j] = e as i32;
                    j += 1;
                    if j == k {
                        break;
                    }
                }
            }
        });

    out
}


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

