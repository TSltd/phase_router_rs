//! Phase Router vs Hash Routing — CLI Benchmark
//!
//! Compares timing and token survival across sizes.
//! Run: cargo run --release --example bench

use phase_router_rs::router::phase_router;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

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

fn bench_fn<F>(label: &str, n: usize, iters: usize, f: F) -> f64
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
    let median = times[times.len() / 2];
    let avg: f64 = times.iter().sum::<f64>() / times.len() as f64;

    println!(
        "  {:<14} N={:<5} | min: {:>8.3} ms | median: {:>8.3} ms | avg: {:>8.3} ms  ({} iters)",
        label, n, min, median, avg, iters
    );

    min
}

fn main() {
    let k = 8;
    let density = 0.3;
    let sizes = [64, 128, 256, 512, 1024, 2048, 4096];

    println!("Phase Router vs Hash Routing Benchmark (k={})", k);
    println!("{}", "=".repeat(95));

    println!("\n{:<14} {:<7} {:>12} {:>12} {:>12}", "Method", "N", "Min (ms)", "Speedup", "");
    println!("{}", "-".repeat(60));

    for &n in &sizes {
        let nb_words = (n + 63) / 64;
        let (s_bits, t_bits, col_perm_s, col_perm_t) =
            generate_test_data(n, nb_words, density, 12345);

        let iters = if n <= 256 { 100 } else if n <= 1024 { 50 } else { 20 };

        let s = &s_bits;
        let t = &t_bits;
        let ps = &col_perm_s;
        let pt = &col_perm_t;

        let pr_min = bench_fn("PhaseRouter", n, iters, |seed| {
            phase_router(s, t, n, nb_words, k, ps, pt, seed)
        });

        let h_min = bench_fn("Hash", n, iters, |seed| hash_route(n, k, seed));

        println!(
            "  {:>14} ratio: hash is {:.1}× faster\n",
            "", pr_min / h_min
        );
    }

    println!("{}", "=".repeat(95));
    println!("Done. Phase Router is slower but capacity-aware — see `cargo run --release --example moe_bench` for quality comparison.");
}
