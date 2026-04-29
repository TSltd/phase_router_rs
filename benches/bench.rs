use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use phase_router_rs::core::{build_s_final, build_t_final, compute_offsets};
use phase_router_rs::router::{phase_router, phase_router_legacy};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

fn generate_test_data(
    n: usize,
    nb_words: usize,
    seed: u64,
) -> (Vec<u64>, Vec<u64>, Vec<usize>, Vec<usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let s_bits: Vec<u64> = (0..n * nb_words).map(|_| rng.gen()).collect();
    let t_bits: Vec<u64> = (0..n * nb_words).map(|_| rng.gen()).collect();

    let mut col_perm_s: Vec<usize> = (0..n).collect();
    let mut col_perm_t: Vec<usize> = (0..n).collect();
    col_perm_s.shuffle(&mut rng);
    col_perm_t.shuffle(&mut rng);

    (s_bits, t_bits, col_perm_s, col_perm_t)
}

fn bench_fused_vs_legacy(c: &mut Criterion) {
    let k = 8;
    let mut group = c.benchmark_group("phase_router");

    for &n in &[64, 128, 256, 512, 1024, 2048, 4096] {
        let nb_words = (n + 63) / 64;
        let (s_bits, t_bits, col_perm_s, col_perm_t) = generate_test_data(n, nb_words, 12345);

        group.bench_with_input(BenchmarkId::new("fused", n), &n, |b, &_n| {
            b.iter(|| {
                phase_router(
                    &s_bits, &t_bits, n, nb_words, k, &col_perm_s, &col_perm_t, 42,
                )
            });
        });

        group.bench_with_input(BenchmarkId::new("legacy", n), &n, |b, &_n| {
            b.iter(|| {
                phase_router_legacy(
                    &s_bits, &t_bits, n, nb_words, k, &col_perm_s, &col_perm_t, 42,
                )
            });
        });
    }

    group.finish();
}

fn bench_build_s_final(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_s_final");

    for &n in &[128, 512, 1024, 4096] {
        let nb_words = (n + 63) / 64;
        let (s_bits, _, col_perm_s, _) = generate_test_data(n, nb_words, 12345);
        let offsets = compute_offsets(&s_bits, n, nb_words);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| build_s_final(&s_bits, &offsets, &col_perm_s, n, nb_words));
        });
    }

    group.finish();
}

fn bench_build_t_final(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_t_final");

    for &n in &[128, 512, 1024, 4096] {
        let nb_words = (n + 63) / 64;
        let (_, t_bits, _, col_perm_t) = generate_test_data(n, nb_words, 12345);
        let offsets = compute_offsets(&t_bits, n, nb_words);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &_n| {
            b.iter(|| build_t_final(&t_bits, &offsets, &col_perm_t, n, nb_words));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_fused_vs_legacy, bench_build_s_final, bench_build_t_final);
criterion_main!(benches);
