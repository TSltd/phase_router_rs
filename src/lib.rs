pub mod core;
pub mod python;
pub mod router;

#[cfg(test)]
mod tests {
    use crate::router::*;
    use std::collections::HashSet;

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

    #[test]
    fn uniform_dispatch_shape_and_uniqueness() {
        let n_tokens = 1024;
        let n_experts = 8;
        let k = 2;
        let routes = phase_router_uniform_dispatch(
            n_tokens, n_experts, k, 0.3, 4, 42,
        );

        assert_eq!(routes.len(), n_tokens * k);

        // Per-row uniqueness + range
        for row in 0..n_tokens {
            let mut seen = HashSet::new();
            for j in 0..k {
                let e = routes[row * k + j];
                if e < 0 {
                    continue;
                }
                assert!(
                    (e as usize) < n_experts,
                    "expert id {} >= n_experts {}",
                    e,
                    n_experts
                );
                assert!(seen.insert(e), "duplicate expert {} in row {}", e, row);
            }
        }

        // Load CV should be very low — kernel balances by construction.
        let mut counts = vec![0u64; n_experts];
        for &e in &routes {
            if e >= 0 {
                counts[e as usize] += 1;
            }
        }
        let total: u64 = counts.iter().sum();
        let mean = total as f64 / n_experts as f64;
        let var: f64 = counts
            .iter()
            .map(|&c| (c as f64 - mean).powi(2))
            .sum::<f64>()
            / n_experts as f64;
        let cv = var.sqrt() / mean.max(1e-9);
        assert!(cv < 0.10, "load CV too high: {}", cv);
    }

    #[test]
    fn uniform_dispatch_determinism() {
        let a = phase_router_uniform_dispatch(512, 8, 2, 0.3, 4, 123);
        let b = phase_router_uniform_dispatch(512, 8, 2, 0.3, 4, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn uniform_dispatch_n_not_divisible() {
        // n_tokens not a multiple of n_experts → should still work,
        // padding handled internally.
        let routes = phase_router_uniform_dispatch(1000, 8, 2, 0.3, 4, 7);
        assert_eq!(routes.len(), 1000 * 2);
    }
}

