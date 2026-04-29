use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use phase_router_rs::router::phase_router;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn generate_test_data(
    n: usize,
    nb_words: usize,
    density: f64,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<usize>, Vec<usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let ones = (density * n as f64).round() as usize;
    let mut s_bits = vec![0u64; n * nb_words];
    let mut t_bits = vec![0u64; n * nb_words];

    for i in 0..n {
        for b in 0..ones.min(n) {
            s_bits[i * nb_words + b / 64] |= 1u64 << (b % 64);
            t_bits[i * nb_words + b / 64] |= 1u64 << (b % 64);
        }
    }

    let mut col_perm_s: Vec<usize> = (0..n).collect();
    let mut col_perm_t: Vec<usize> = (0..n).collect();
    col_perm_s.shuffle(&mut rng);
    col_perm_t.shuffle(&mut rng);

    (s_bits, t_bits, col_perm_s, col_perm_t)
}

fn hash_route(n: usize, k: usize, seed: u64) -> Vec<i32> {
    let mut routes = vec![-1i32; n * k];
    for i in 0..n {
        for ki in 0..k {
            let mut hasher = DefaultHasher::new();
            (i as u64, seed, ki as u64).hash(&mut hasher);
            routes[i * k + ki] = (hasher.finish() % n as u64) as i32;
        }
    }
    routes
}

fn bench_phase_router_vs_hash(c: &mut Criterion) {
    let k = 8;
    let density = 0.3;
    let mut group = c.benchmark_group("phase_router_vs_hash");

    for &n in &[64, 128, 256, 512, 1024, 2048, 4096] {
        let nb_words = (n + 63) / 64;
        let (s_bits, t_bits, col_perm_s, col_perm_t) =
            generate_test_data(n, nb_words, density, 12345);

        group.bench_with_input(BenchmarkId::new("phase_router", n), &n, |b, &_n| {
            b.iter(|| {
                phase_router(
                    &s_bits, &t_bits, n, nb_words, k, &col_perm_s, &col_perm_t, 42,
                )
            });
        });

        group.bench_with_input(BenchmarkId::new("hash", n), &n, |b, &_n| {
            b.iter(|| hash_route(n, k, 42));
        });
    }

    group.finish();
}

fn bench_phase_router_k_sweep(c: &mut Criterion) {
    let n = 1024;
    let nb_words = (n + 63) / 64;
    let density = 0.3;
    let (s_bits, t_bits, col_perm_s, col_perm_t) =
        generate_test_data(n, nb_words, density, 12345);

    let mut group = c.benchmark_group("phase_router_k_sweep");

    for &k in &[1, 2, 4, 8, 16] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                phase_router(
                    &s_bits, &t_bits, n, nb_words, k, &col_perm_s, &col_perm_t, 42,
                )
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_phase_router_vs_hash, bench_phase_router_k_sweep);
criterion_main!(benches);
