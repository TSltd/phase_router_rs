//! Head-to-head quality probe for five routing strategies across the four
//! canonical workload profiles.
//!
//! - `hash`                       — uniform DefaultHasher routing (capacity-blind baseline)
//! - `phase_router`               — current global-permutation kernel
//! - `phase_router_rs (additive)` — rank/select, additive rank-space transport
//! - `phase_router_rs (affine)`   — rank/select, affine rank-space transport
//! - `phase_router_rs (hybrid)`   — rank/select with global occupancy walk via offsets_s
//!
//! Reports for each (workload, method):
//!   - mean / max / std / CV / max-over-mean of per-target load
//!   - Pearson correlation between target capacity and observed load
//!   - token-survival rate at headrooms 1.0 / 1.2 / 1.5 / 2.0
//!   - wall-clock time per call
//!
//! Run:
//!   cargo run --release --features rank-select --example quality_probe

use phase_router_rs::router::phase_router;
use phase_router_rs::router_rs::{
    phase_router_rs, phase_router_rs_affine, phase_router_rs_hybrid,
};
use phase_router_rs::workloads::{Profile, Workload};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

// ── baselines ────────────────────────────────────────────────────────────

fn hash_route(n: usize, k: usize, seed: u64) -> Vec<i32> {
    let mut routes = vec![-1i32; n * k];
    for i in 0..n {
        for ki in 0..k {
            let mut h = DefaultHasher::new();
            (i as u64, seed, ki as u64).hash(&mut h);
            routes[i * k + ki] = (h.finish() % n as u64) as i32;
        }
    }
    routes
}

// ── metrics ──────────────────────────────────────────────────────────────

fn loads(routes: &[i32], n: usize) -> Vec<f64> {
    let mut l = vec![0.0f64; n];
    for &c in routes {
        if c >= 0 && (c as usize) < n {
            l[c as usize] += 1.0;
        }
    }
    l
}

fn stats(values: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let var = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let cv = if mean > 0.0 { std / mean } else { 0.0 };
    let max_over_mean = if mean > 0.0 { max / mean } else { 0.0 };
    (mean, max, std, cv, max_over_mean)
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for i in 0..x.len() {
        let a = x[i] - mx;
        let b = y[i] - my;
        num += a * b;
        dx2 += a * a;
        dy2 += b * b;
    }
    if dx2 == 0.0 || dy2 == 0.0 {
        return 0.0;
    }
    num / (dx2.sqrt() * dy2.sqrt())
}

fn survival_after_caps(routes: &[i32], n: usize, k: usize, hard_caps: &[usize]) -> f64 {
    let mut counts = vec![0usize; n];
    let mut survived = 0usize;
    for &c in routes {
        if c >= 0 && (c as usize) < n {
            let t = c as usize;
            if counts[t] < hard_caps[t] {
                counts[t] += 1;
                survived += 1;
            }
        }
    }
    survived as f64 / (n * k) as f64
}

// ── reporting helper ────────────────────────────────────────────────────

fn report_row<F: FnMut(u64) -> Vec<i32>>(
    name: &str,
    mut route_fn: F,
    n: usize,
    k: usize,
    seed: u64,
    iters: usize,
    rel_caps: &[f64],
    headrooms: &[f64],
) {
    // warm-up
    let _ = route_fn(seed);

    let mut total_time = 0.0;
    let mut routes_last = Vec::new();
    for it in 0..iters {
        let s = seed + it as u64;
        let start = Instant::now();
        routes_last = route_fn(s);
        total_time += start.elapsed().as_secs_f64();
    }
    let avg_ms = (total_time * 1000.0) / iters as f64;

    let l = loads(&routes_last, n);
    let (mean, max, std, cv, mom) = stats(&l);
    let corr = pearson(&l, rel_caps);

    let total_cap: f64 = rel_caps.iter().sum();
    let mut surv = Vec::with_capacity(headrooms.len());
    for &hr in headrooms {
        let hard_caps: Vec<usize> = rel_caps
            .iter()
            .map(|&c| ((c / total_cap) * (n * k) as f64 * hr).ceil().max(1.0) as usize)
            .collect();
        surv.push(survival_after_caps(&routes_last, n, k, &hard_caps));
    }

    println!(
        "{:<28} {:>8.2} {:>8.0} {:>8.2} {:>8.3} {:>8.2} {:>10.3}   {:>6.3}  {:>6.3}  {:>6.3}  {:>6.3}   ({:.2} ms/call)",
        name, mean, max, std, cv, mom, corr,
        surv[0], surv[1], surv[2], surv[3],
        avg_ms
    );
}

// ── per-workload section ────────────────────────────────────────────────

fn run_section(w: &Workload, k: usize, seed: u64, iters: usize, headrooms: &[f64]) {
    println!("\n━━━ workload: {} ━━━", w.label);
    println!(
        "\n{:<28} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}   {}",
        "method",
        "mean",
        "max",
        "std",
        "CV",
        "max/μ",
        "corr(L,c)",
        "survival @ headroom"
    );
    println!(
        "{:<28} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10}   {:>6}  {:>6}  {:>6}  {:>6}",
        "", "", "", "", "", "", "", "1.0×", "1.2×", "1.5×", "2.0×"
    );
    println!("{}", "-".repeat(120));

    let n = w.n;
    let nb_words = w.nb_words;

    report_row(
        "hash",
        |s| hash_route(n, k, s),
        n, k, seed, iters, &w.rel_caps, headrooms,
    );
    report_row(
        "phase_router",
        |s| {
            phase_router(
                &w.s_bits, &w.t_bits, n, nb_words, k,
                &w.col_perm_s, &w.col_perm_t, s,
            )
        },
        n, k, seed, iters, &w.rel_caps, headrooms,
    );
    report_row(
        "phase_router_rs (additive)",
        |s| {
            phase_router_rs(
                &w.s_bits, &w.t_bits, n, nb_words, k,
                &w.col_perm_s, &w.col_perm_t, s,
            )
        },
        n, k, seed, iters, &w.rel_caps, headrooms,
    );
    report_row(
        "phase_router_rs (affine)",
        |s| {
            phase_router_rs_affine(
                &w.s_bits, &w.t_bits, n, nb_words, k,
                &w.col_perm_s, &w.col_perm_t, s,
            )
        },
        n, k, seed, iters, &w.rel_caps, headrooms,
    );
    report_row(
        "phase_router_rs (hybrid)",
        |s| {
            phase_router_rs_hybrid(
                &w.s_bits, &w.t_bits, n, nb_words, k,
                &w.col_perm_s, &w.col_perm_t, s,
            )
        },
        n, k, seed, iters, &w.rel_caps, headrooms,
    );
}

// ── main ────────────────────────────────────────────────────────────────

fn main() {
    let n = 1024;
    let k = 4;
    let seed = 42u64;
    let density = 0.3;
    let base_density = 0.3;
    let iters = 5;

    println!("Quality Probe — N={n}, k={k}, density={density}, strong-hetero capacity");
    println!("Sweeping methods × workloads (rank/select kernels include the hybrid).");
    println!("{}", "=".repeat(120));

    let headrooms = [1.0, 1.2, 1.5, 2.0];

    for profile in Profile::ALL {
        let w = profile.build(n, density, base_density, seed);
        run_section(&w, k, seed, iters, &headrooms);
    }

    println!("\n{}", "=".repeat(120));
    println!(
        "\nLegend:\n  CV       = std / mean (lower is better)\n  max/μ    = peak hot-spot ratio (lower is better)\n  corr(L,c)= Pearson(load, relative_capacity), capacity-aligned routers approach 1.0\n  survival = fraction of routed tokens accepted under hard caps at the given headroom\n"
    );
}
