//! MoE Benchmark: Phase Router vs Hash-based routing
//!
//! Compares load balance (skew) and performance across different
//! token weight distributions typical in Mixture-of-Experts inference.
//!
//! Run: cargo run --release --example moe_bench

use phase_router_rs::router::phase_router;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::time::Instant;

// --- Weight distributions ---

fn uniform_weights(n: usize, rng: &mut impl Rng) -> Vec<f64> {
    (0..n).map(|_| rng.gen_range(0.5..1.5)).collect()
}

fn zipf_weights(n: usize, s: f64, rng: &mut impl Rng) -> Vec<f64> {
    let mut w: Vec<f64> = (1..=n).map(|i| 1.0 / (i as f64).powf(s)).collect();
    w.shuffle(rng);
    w
}

fn pareto_80_20(n: usize, rng: &mut impl Rng) -> Vec<f64> {
    let heavy = (n as f64 * 0.2) as usize;
    let mut w = vec![0.25; n]; // 80% of tokens get low weight
    for i in 0..heavy {
        w[i] = 4.0; // 20% of tokens get 16× more weight
    }
    w.shuffle(rng);
    w
}

// --- Convert weights to bit-packed matrix ---
// Each row i gets ceil(w_i / total * n * density_factor) bits set

fn weights_to_bits(weights: &[f64], n: usize, nb_words: usize) -> Vec<u64> {
    let total: f64 = weights.iter().sum();
    let mut bits = vec![0u64; n * nb_words];

    for i in 0..n {
        let frac = weights[i] / total;
        let ones = ((frac * n as f64 * 0.8).round() as usize).max(1).min(n);

        for b in 0..ones {
            let w = b / 64;
            let bit = b % 64;
            bits[i * nb_words + w] |= 1u64 << bit;
        }
    }
    bits
}

// --- Hash-based routing ---

fn hash_route(n: usize, k: usize, seed: u64) -> Vec<i32> {
    let mut routes = vec![-1i32; n * k];

    for i in 0..n {
        // Hash token i to k experts
        for ki in 0..k {
            let mut hasher = DefaultHasher::new();
            (i as u64, seed, ki as u64).hash(&mut hasher);
            let h = hasher.finish();
            routes[i * k + ki] = (h % n as u64) as i32;
        }
    }
    routes
}

fn modular_hash_route(n: usize, k: usize, seed: u64) -> Vec<i32> {
    let prime = 6364136223846793005u64; // LCG multiplier
    let mut routes = vec![-1i32; n * k];

    for i in 0..n {
        for ki in 0..k {
            let h = (i as u64)
                .wrapping_mul(prime)
                .wrapping_add(seed)
                .wrapping_add(ki as u64 * 2654435761);
            routes[i * k + ki] = (h % n as u64) as i32;
        }
    }
    routes
}

// --- Metrics ---

struct LoadMetrics {
    cv: f64,        // coefficient of variation
    max_min: f64,   // max/min load ratio
    max_load: usize,
    min_load: usize,
    mean_load: f64,
}

fn compute_load_metrics(routes: &[i32], n: usize, k: usize, weights: &[f64]) -> LoadMetrics {
    let mut loads = vec![0.0f64; n];

    for i in 0..n {
        for ki in 0..k {
            let target = routes[i * k + ki];
            if target >= 0 && (target as usize) < n {
                loads[target as usize] += weights[i];
            }
        }
    }

    let mean = loads.iter().sum::<f64>() / n as f64;
    let variance = loads.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();
    let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };

    let max_load_f = loads.iter().cloned().fold(0.0f64, f64::max);
    let min_load_f = loads.iter().cloned().fold(f64::MAX, f64::min);
    let max_min = if min_load_f > 0.0 {
        max_load_f / min_load_f
    } else {
        f64::INFINITY
    };

    LoadMetrics {
        cv,
        max_min,
        max_load: max_load_f.round() as usize,
        min_load: min_load_f.round() as usize,
        mean_load: mean,
    }
}

// --- Benchmark runner ---

fn run_benchmark(
    name: &str,
    dist_name: &str,
    weights: &[f64],
    n: usize,
    k: usize,
    seed: u64,
    csv: &mut Vec<String>,
) {
    let nb_words = (n + 63) / 64;
    let iters = if n <= 512 { 20 } else { 10 };

    // --- Phase Router ---
    let s_bits = weights_to_bits(weights, n, nb_words);

    // Target weights: equal capacity
    let t_weights: Vec<f64> = vec![1.0; n];
    let t_bits = weights_to_bits(&t_weights, n, nb_words);

    let mut rng = ChaCha8Rng::seed_from_u64(seed + 1000);
    let mut col_perm_s: Vec<usize> = (0..n).collect();
    let mut col_perm_t: Vec<usize> = (0..n).collect();
    col_perm_s.shuffle(&mut rng);
    col_perm_t.shuffle(&mut rng);

    // Warmup
    let _ = phase_router(&s_bits, &t_bits, n, nb_words, k, &col_perm_s, &col_perm_t, seed);

    let start = Instant::now();
    let mut pr_routes = vec![];
    for iter in 0..iters {
        pr_routes = phase_router(
            &s_bits, &t_bits, n, nb_words, k, &col_perm_s, &col_perm_t, seed + iter as u64,
        );
    }
    let pr_time = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let pr_metrics = compute_load_metrics(&pr_routes, n, k, weights);

    // --- Hash routing ---
    let start = Instant::now();
    let mut hash_routes = vec![];
    for iter in 0..iters {
        hash_routes = hash_route(n, k, seed + iter as u64);
    }
    let hash_time = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let hash_metrics = compute_load_metrics(&hash_routes, n, k, weights);

    // --- Modular hash routing ---
    let start = Instant::now();
    let mut mod_routes = vec![];
    for iter in 0..iters {
        mod_routes = modular_hash_route(n, k, seed + iter as u64);
    }
    let mod_time = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let mod_metrics = compute_load_metrics(&mod_routes, n, k, weights);

    // Print results
    println!("  {:<14} | CV: {:.4} | max/min: {:>6.2} | time: {:>8.3} ms",
        "PhaseRouter", pr_metrics.cv, pr_metrics.max_min, pr_time);
    println!("  {:<14} | CV: {:.4} | max/min: {:>6.2} | time: {:>8.3} ms",
        "Hash", hash_metrics.cv, hash_metrics.max_min, hash_time);
    println!("  {:<14} | CV: {:.4} | max/min: {:>6.2} | time: {:>8.3} ms",
        "ModularHash", mod_metrics.cv, mod_metrics.max_min, mod_time);

    // CSV output
    csv.push(format!("{},{},{},{},{:.6},{:.4},{:.2}",
        dist_name, n, "PhaseRouter", k, pr_time, pr_metrics.cv, pr_metrics.max_min));
    csv.push(format!("{},{},{},{},{:.6},{:.4},{:.2}",
        dist_name, n, "Hash", k, hash_time, hash_metrics.cv, hash_metrics.max_min));
    csv.push(format!("{},{},{},{},{:.6},{:.4},{:.2}",
        dist_name, n, "ModularHash", k, mod_time, mod_metrics.cv, mod_metrics.max_min));
}

fn main() {
    let k = 2;
    let seed = 42u64;
    let sizes = [256, 512, 1024, 2048, 4096];

    let distributions: Vec<(&str, Box<dyn Fn(usize, &mut ChaCha8Rng) -> Vec<f64>>)> = vec![
        ("uniform", Box::new(|n, rng| uniform_weights(n, rng))),
        ("zipf_1.0", Box::new(|n, rng| zipf_weights(n, 1.0, rng))),
        ("zipf_2.0", Box::new(|n, rng| zipf_weights(n, 2.0, rng))),
        ("pareto_80_20", Box::new(|n, rng| pareto_80_20(n, rng))),
    ];

    let mut csv = vec!["distribution,n,method,k,time_ms,cv,max_min_ratio".to_string()];

    println!("MoE Benchmark: Phase Router vs Hash Routing (k={})", k);
    println!("{}", "=".repeat(80));

    for (dist_name, dist_fn) in &distributions {
        println!("\n--- Distribution: {} ---", dist_name);

        for &n in &sizes {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let weights = dist_fn(n, &mut rng);

            println!("\n  N={}", n);
            run_benchmark(dist_name, dist_name, &weights, n, k, seed, &mut csv);
        }
    }

    println!("\n{}", "=".repeat(80));

    // Write CSV
    let csv_path = "moe_results.csv";
    let mut file = std::fs::File::create(csv_path).expect("Cannot create CSV");
    for line in &csv {
        writeln!(file, "{}", line).expect("Cannot write CSV");
    }
    println!("\nResults written to {}", csv_path);
}
