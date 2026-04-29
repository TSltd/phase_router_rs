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
