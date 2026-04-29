/// Set bits at positions [from, to) in `dst`.
#[inline(always)]
fn set_bit_range(dst: &mut [u64], from: usize, to: usize) {
    if from >= to {
        return;
    }

    let first_word = from >> 6;
    let last_word = (to - 1) >> 6;
    let b_start = from & 63;
    let b_end = (to - 1) & 63;

    if first_word == last_word {
        let low = !0u64 << b_start;
        let high = if b_end == 63 {
            !0u64
        } else {
            (1u64 << (b_end + 1)) - 1
        };
        dst[first_word] |= low & high;
    } else {
        dst[first_word] |= !0u64 << b_start;
        for w in (first_word + 1)..last_word {
            dst[w] = !0u64;
        }
        let high = if b_end == 63 {
            !0u64
        } else {
            (1u64 << (b_end + 1)) - 1
        };
        dst[last_word] |= high;
    }
}

/// Directly compute the result of left-aligning `ones` bits then rotating by `offset`,
/// without materializing an intermediate left-aligned buffer.
///
/// Sets `ones` consecutive bits starting at position `offset`, wrapping at `n`.
#[inline(always)]
pub fn fill_rotated_bits(dst: &mut [u64], n: usize, offset: usize, ones: usize) {
    dst.fill(0);
    if ones == 0 || n == 0 {
        return;
    }

    let ones = ones.min(n);
    let start = offset % n;
    let end = start + ones;

    if end <= n {
        set_bit_range(dst, start, end);
    } else {
        set_bit_range(dst, start, n);
        set_bit_range(dst, 0, end - n);
    }
}

#[inline(always)]
pub fn rotate_bits_full(
    src: &[u64],
    n: usize,
    nb_words: usize,
    offset: usize,
    dst: &mut [u64],
) {
    if n == 0 {
        return;
    }

    let mask = if n % 64 == 0 {
        !0u64
    } else {
        (1u64 << (n % 64)) - 1
    };

    if offset == 0 {
        dst[0] = src[0] & mask;
        for w in 1..nb_words {
            dst[w] = src[w];
        }
        return;
    }

    if nb_words == 1 {
        dst[0] = ((src[0] << offset) | (src[0] >> (64 - offset))) & mask;
        return;
    }

    let word_shift = offset / 64;
    let bit_shift = offset % 64;

    for w in 0..nb_words {
        let src1 = (w + nb_words - word_shift) % nb_words;
        let src2 = (w + nb_words - word_shift - 1 + nb_words) % nb_words;

        let hi = if bit_shift == 0 {
            0
        } else {
            src[src2] >> (64 - bit_shift)
        };

        let lo = src[src1] << bit_shift;

        dst[w] = lo | hi;

        if w == nb_words - 1 {
            dst[w] &= mask;
        }
    }
}

#[inline(always)]
pub fn permute_columns_bits(src: &[u64], dst: &mut [u64], col_perm: &[usize], n: usize) {
    dst.fill(0);

    for j in 0..n {
        let src_j = col_perm[j];
        let src_w = src_j / 64;
        let src_b = src_j % 64;

        if (src[src_w] >> src_b) & 1 == 1 {
            let dst_w = j / 64;
            let dst_b = j % 64;
            dst[dst_w] |= 1 << dst_b;
        }
    }
}
