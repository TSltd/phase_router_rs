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
