use phase_router_rs::router::{phase_router, phase_router_legacy};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

fn generate_test_data(n: usize, nb_words: usize, seed: u64) -> (Vec<u64>, Vec<u64>, Vec<usize>, Vec<usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let s_bits: Vec<u64> = (0..n * nb_words).map(|_| rng.gen()).collect();
    let t_bits: Vec<u64> = (0..n * nb_words).map(|_| rng.gen()).collect();

    let mut col_perm_s: Vec<usize> = (0..n).collect();
    let mut col_perm_t: Vec<usize> = (0..n).collect();
    col_perm_s.shuffle(&mut rng);
    col_perm_t.shuffle(&mut rng);

    (s_bits, t_bits, col_perm_s, col_perm_t)
}

fn bench_fn<F>(label: &str, n: usize, k: usize, iters: usize, f: F)
where
    F: Fn(u64) -> Vec<i32>,
{
    // Warmup
    let _ = f(999);

    let mut times = Vec::with_capacity(iters);

    for iter in 0..iters {
        let start = Instant::now();
        let routes = f(iter as u64);
        let elapsed = start.elapsed();
        std::hint::black_box(&routes);
        times.push(elapsed.as_secs_f64() * 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = times[0];
    let max = times[times.len() - 1];
    let avg: f64 = times.iter().sum::<f64>() / times.len() as f64;
    let median = times[times.len() / 2];

    println!(
        "  {:<8} N={:<6} | min: {:>8.3} ms | median: {:>8.3} ms | avg: {:>8.3} ms | max: {:>8.3} ms  ({} iters)",
        label, n, min, median, avg, max, iters
    );
}

fn main() {
    let k = 8;
    let sizes = [64, 128, 256, 512, 1024, 2048, 4096];

    println!("Phase Router Benchmark — Fused vs Legacy (k={})", k);
    println!("{}", "=".repeat(105));

    for &n in &sizes {
        let nb_words = (n + 63) / 64;
        let (s_bits, t_bits, col_perm_s, col_perm_t) = generate_test_data(n, nb_words, 12345);

        let iters = if n <= 256 { 100 } else if n <= 1024 { 50 } else { 20 };

        let s = &s_bits;
        let t = &t_bits;
        let ps = &col_perm_s;
        let pt = &col_perm_t;

        bench_fn("fused", n, k, iters, |seed| {
            phase_router(s, t, n, nb_words, k, ps, pt, seed)
        });

        bench_fn("legacy", n, k, iters, |seed| {
            phase_router_legacy(s, t, n, nb_words, k, ps, pt, seed)
        });

        println!();
    }

    println!("{}", "=".repeat(105));
    println!("Done.");
}
