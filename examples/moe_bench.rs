//! MoE Capacity-Constrained Benchmark: Phase Router vs Uniform Hash
//!
//! Four experiments demonstrating Phase Router's advantage:
//!   1. Headroom sweep   — survival vs overprovisioning ratio
//!   2. K sweep          — survival vs fan-out k
//!   3. Scale sweep      — survival vs N across capacity profiles
//!   4. Heterogeneity sweep — survival vs capacity skew
//!
//! Run: cargo run --release --example moe_bench

use phase_router_rs::router::phase_router;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::time::Instant;

// ── Bit matrix construction ──────────────────────────────────────────────

fn build_bits(ones_per_row: &[usize], n: usize, nb_words: usize) -> Vec<u64> {
    let mut bits = vec![0u64; n * nb_words];
    for i in 0..n {
        let ones = ones_per_row[i].min(n);
        for b in 0..ones {
            bits[i * nb_words + b / 64] |= 1u64 << (b % 64);
        }
    }
    bits
}

// ── Uniform hash routing baseline ────────────────────────────────────────

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

// ── Capacity enforcement ────────────────────────────────────────────────

fn enforce_capacity(routes: &mut [i32], n: usize, caps: &[usize]) -> usize {
    let mut loads = vec![0usize; n];
    let mut dropped = 0usize;
    for slot in routes.iter_mut() {
        let t = *slot;
        if t >= 0 && (t as usize) < n {
            let tu = t as usize;
            if loads[tu] < caps[tu] {
                loads[tu] += 1;
            } else {
                *slot = -1;
                dropped += 1;
            }
        }
    }
    dropped
}

// ── Metrics ─────────────────────────────────────────────────────────────

fn survival_rate(routes: &[i32], n: usize, k: usize) -> f64 {
    let assigned = routes.iter().filter(|&&t| t >= 0 && (t as usize) < n).count();
    assigned as f64 / (n * k) as f64
}

// ── Capacity scenarios ──────────────────────────────────────────────────

fn uniform_caps(n: usize, _rng: &mut ChaCha8Rng) -> Vec<f64> {
    vec![1.0; n]
}

fn mild_hetero_caps(n: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let mut c = vec![1.0f64; n];
    let high = (n as f64 * 0.2) as usize;
    for v in c.iter_mut().take(high) { *v = 3.0; }
    c.shuffle(rng);
    c
}

fn strong_hetero_caps(n: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let mut c = vec![1.0f64; n];
    let t1 = (n as f64 * 0.1) as usize;
    let t2 = (n as f64 * 0.2) as usize;
    for v in c.iter_mut().take(t1) { *v = 8.0; }
    for v in c.iter_mut().skip(t1).take(t2) { *v = 2.0; }
    c.shuffle(rng);
    c
}

fn extreme_hetero_caps(n: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let mut c = vec![1.0f64; n];
    let top = (n as f64 * 0.05).max(1.0) as usize;
    for v in c.iter_mut().take(top) { *v = 16.0; }
    c.shuffle(rng);
    c
}

// ── Helper: run one comparison ──────────────────────────────────────────

struct RunResult {
    pr_survival: f64,
    pr_time_ms: f64,
    hash_survival: f64,
    hash_time_ms: f64,
}

fn run_comparison(
    n: usize,
    k: usize,
    rel_caps: &[f64],
    headroom: f64,
    base_density: f64,
    seed: u64,
) -> RunResult {
    let nb_words = (n + 63) / 64;
    let iters = if n <= 512 { 20 } else { 5 };
    let mut rng = ChaCha8Rng::seed_from_u64(seed + 7777);

    let total_cap: f64 = rel_caps.iter().sum();
    let mean_cap = total_cap / n as f64;

    // S matrix
    let s_density = (base_density * n as f64).round() as usize;
    let s_ones: Vec<usize> = vec![s_density; n];
    let s_bits = build_bits(&s_ones, n, nb_words);

    // T matrix: density proportional to capacity
    let t_ones: Vec<usize> = rel_caps
        .iter()
        .map(|&c| {
            ((c / mean_cap) * base_density * n as f64)
                .round()
                .max(1.0)
                .min(n as f64) as usize
        })
        .collect();
    let t_bits = build_bits(&t_ones, n, nb_words);

    // Hard caps with headroom
    let hard_caps: Vec<usize> = rel_caps
        .iter()
        .map(|&c| ((c / total_cap) * (n * k) as f64 * headroom).ceil().max(1.0) as usize)
        .collect();

    // Column permutations
    let mut col_perm_s: Vec<usize> = (0..n).collect();
    let mut col_perm_t: Vec<usize> = (0..n).collect();
    col_perm_s.shuffle(&mut rng);
    col_perm_t.shuffle(&mut rng);

    // Phase Router
    let _ = phase_router(&s_bits, &t_bits, n, nb_words, k, &col_perm_s, &col_perm_t, seed);
    let start = Instant::now();
    let mut pr_routes = vec![];
    for it in 0..iters {
        pr_routes = phase_router(
            &s_bits, &t_bits, n, nb_words, k, &col_perm_s, &col_perm_t, seed + it as u64,
        );
    }
    let pr_time = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    enforce_capacity(&mut pr_routes, n, &hard_caps);
    let pr_surv = survival_rate(&pr_routes, n, k);

    // Uniform Hash
    let start = Instant::now();
    let mut h_routes = vec![];
    for it in 0..iters {
        h_routes = hash_route(n, k, seed + it as u64);
    }
    let h_time = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    enforce_capacity(&mut h_routes, n, &hard_caps);
    let h_surv = survival_rate(&h_routes, n, k);

    RunResult {
        pr_survival: pr_surv,
        pr_time_ms: pr_time,
        hash_survival: h_surv,
        hash_time_ms: h_time,
    }
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    let seed = 42u64;
    let base_density = 0.3;

    let mut csv = vec![
        "experiment,param,n,k,headroom,method,survival_rate,time_ms".to_string(),
    ];

    println!("MoE Benchmark: Phase Router vs Uniform Hash");
    println!("{}", "=".repeat(90));

    // ────────────────────────────────────────────────────────────
    // Experiment 1: Headroom sweep (the hero chart)
    // ────────────────────────────────────────────────────────────
    println!("\n━━━ Experiment 1: Headroom Sweep (N=1024, k=2, strong_hetero) ━━━\n");
    let n = 1024;
    let k = 2;
    let headrooms = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0];

    for &hr in &headrooms {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let caps = strong_hetero_caps(n, &mut rng);
        let r = run_comparison(n, k, &caps, hr, base_density, seed);
        println!(
            "  headroom={:.2}  PR: {:.4}  Hash: {:.4}  Δ={:+.4}",
            hr, r.pr_survival, r.hash_survival, r.pr_survival - r.hash_survival
        );
        csv.push(format!("headroom,{:.2},{},{},{:.2},PhaseRouter,{:.6},{:.6}",
            hr, n, k, hr, r.pr_survival, r.pr_time_ms));
        csv.push(format!("headroom,{:.2},{},{},{:.2},UniformHash,{:.6},{:.6}",
            hr, n, k, hr, r.hash_survival, r.hash_time_ms));
    }

    // ────────────────────────────────────────────────────────────
    // Experiment 2: K sweep (fan-out)
    // ────────────────────────────────────────────────────────────
    println!("\n━━━ Experiment 2: K Sweep (N=1024, headroom=1.2, strong_hetero) ━━━\n");
    let n = 1024;
    let hr = 1.2;
    let ks = [1, 2, 4, 8, 16];

    for &k in &ks {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let caps = strong_hetero_caps(n, &mut rng);
        let r = run_comparison(n, k, &caps, hr, base_density, seed);
        println!(
            "  k={:<3}  PR: {:.4}  Hash: {:.4}  Δ={:+.4}",
            k, r.pr_survival, r.hash_survival, r.pr_survival - r.hash_survival
        );
        csv.push(format!("k_sweep,{},{},{},{:.2},PhaseRouter,{:.6},{:.6}",
            k, n, k, hr, r.pr_survival, r.pr_time_ms));
        csv.push(format!("k_sweep,{},{},{},{:.2},UniformHash,{:.6},{:.6}",
            k, n, k, hr, r.hash_survival, r.hash_time_ms));
    }

    // ────────────────────────────────────────────────────────────
    // Experiment 3: Scale sweep (N)
    // ────────────────────────────────────────────────────────────
    println!("\n━━━ Experiment 3: Scale Sweep (k=2, headroom=1.2, strong_hetero) ━━━\n");
    let k = 2;
    let hr = 1.2;
    let sizes = [256, 512, 1024, 2048, 4096];

    for &n in &sizes {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let caps = strong_hetero_caps(n, &mut rng);
        let r = run_comparison(n, k, &caps, hr, base_density, seed);
        println!(
            "  N={:<5}  PR: {:.4}  Hash: {:.4}  Δ={:+.4}  PR: {:.2}ms  Hash: {:.3}ms",
            n, r.pr_survival, r.hash_survival, r.pr_survival - r.hash_survival,
            r.pr_time_ms, r.hash_time_ms
        );
        csv.push(format!("scale,{},{},{},{:.2},PhaseRouter,{:.6},{:.6}",
            n, n, k, hr, r.pr_survival, r.pr_time_ms));
        csv.push(format!("scale,{},{},{},{:.2},UniformHash,{:.6},{:.6}",
            n, n, k, hr, r.hash_survival, r.hash_time_ms));
    }

    // ────────────────────────────────────────────────────────────
    // Experiment 4: Heterogeneity sweep
    // ────────────────────────────────────────────────────────────
    println!("\n━━━ Experiment 4: Heterogeneity Sweep (N=1024, k=2, headroom=1.2) ━━━\n");
    let n = 1024;
    let k = 2;
    let hr = 1.2;
    let cap_fns: Vec<(&str, Box<dyn Fn(usize, &mut ChaCha8Rng) -> Vec<f64>>)> = vec![
        ("uniform", Box::new(uniform_caps)),
        ("mild_hetero", Box::new(mild_hetero_caps)),
        ("strong_hetero", Box::new(strong_hetero_caps)),
        ("extreme_hetero", Box::new(extreme_hetero_caps)),
    ];

    for (name, cap_fn) in &cap_fns {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let caps = cap_fn(n, &mut rng);
        let r = run_comparison(n, k, &caps, hr, base_density, seed);
        println!(
            "  {:<16}  PR: {:.4}  Hash: {:.4}  Δ={:+.4}",
            name, r.pr_survival, r.hash_survival, r.pr_survival - r.hash_survival
        );
        csv.push(format!("hetero,{},{},{},{:.2},PhaseRouter,{:.6},{:.6}",
            name, n, k, hr, r.pr_survival, r.pr_time_ms));
        csv.push(format!("hetero,{},{},{},{:.2},UniformHash,{:.6},{:.6}",
            name, n, k, hr, r.hash_survival, r.hash_time_ms));
    }

    println!("\n{}", "=".repeat(90));

    // Write CSV
    let csv_path = "moe_results.csv";
    let mut file = std::fs::File::create(csv_path).expect("Cannot create CSV");
    for line in &csv {
        writeln!(file, "{}", line).expect("Cannot write CSV");
    }
    println!("\nResults written to {}", csv_path);
}
