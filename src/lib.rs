pub mod bitops;
pub mod core;
pub mod router;

#[cfg(test)]
mod tests {
    use crate::router::*;

    #[test]
    fn basic_run() {
        let n = 128;
        let nb = (n + 63) / 64;

        let s_bits = vec![0xFFFFFFFFFFFFFFFF; n * nb];
        let t_bits = vec![0xFFFFFFFFFFFFFFFF; n * nb];

        let col_perm: Vec<usize> = (0..n).collect();

        let routes = phase_router(
            &s_bits,
            &t_bits,
            n,
            nb,
            8,
            &col_perm,
            &col_perm,
            42,
        );

        assert_eq!(routes.len(), n * 8);
    }
}