//! Spectral / structural diagnostic probe across kernels × workloads.
//!
//! Sweeps the four canonical workload profiles
//! (`dense_uniform`, `dense_diverse`, `sparse_diverse`, `block_local`) against
//! five routing methods (`hash`, `phase_router`, `phase_router_rs`,
//! `phase_router_rs_affine`, `phase_router_rs_hybrid`) and reports:
//!
//! - Discrepancy of per-target loads (`max_abs_dev`, `range/mean`, `CV`,
//!   `max/mean`).
//! - Low-frequency Fourier energy share of the load signal (a proxy for
//!   "low-discrepancy distributed-occupancy structure" — high share means
//!   the loads carry large-scale shape, low share means they look closer to
//!   white noise).
//! - Pairwise candidate Jaccard overlap between random pairs of source rows.
//!   Intrinsic-decode kernels collapse to ~1.0 on `dense_uniform`; the
//!   original `phase_router` and the hybrid stay well below 1.0.
//! - Pearson correlation between observed load and relative target capacity.
//!
//! Writes a tidy CSV (`spectral_probe.csv`) for downstream plotting.
//!
//! Run:
//!   cargo run --release --features rank-select --example spectral_probe

use phase_router_rs::metrics::{
    discrepancy, loads_per_target, low_freq_energy_share, pairwise_candidate_overlap,
    split_routes, support_validity,
};
use phase_router_rs::router::phase_router;
use phase_router_rs::router_rs::{
    phase_router_rs, phase_router_rs_affine, phase_router_rs_hybrid,
};
use phase_router_rs::workloads::{Profile, Workload};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::time::Instant;

// ── uniform-hash baseline ────────────────────────────────────────────────

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

// ── pearson on f64 vectors ──────────────────────────────────────────────

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut num, mut dx2, mut dy2) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let a = x[i] - mx;
        let b = y[i] - my;
        num += a * b;
        dx2 += a * a;
        dy2 += b * b;
    }
    if dx2 == 0.0 || dy2 == 0.0 {
        0.0
    } else {
        num / (dx2.sqrt() * dy2.sqrt())
    }
}

// ── one row of the report ───────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Report {
    method: &'static str,
    workload: String,
    n: usize,
    k: usize,
    mean_load: f64,
    cv: f64,
    range_over_mean: f64,
    max_over_mean: f64,
    max_abs_dev: f64,
    low_freq_share: f64,
    mean_jaccard: f64,
    max_jaccard: f64,
    corr_load_cap: f64,
    support_valid: f64,
    time_ms: f64,
}

fn analyse(
    method: &'static str,
    routes: Vec<i32>,
    w: &Workload,
    k: usize,
    elapsed_ms: f64,
    overlap_seed: u64,
) -> Report {
    let n = w.n;
    let loads = loads_per_target(&routes, n);
    let loads_f: Vec<f64> = loads.iter().map(|&l| l as f64).collect();
    let expected = (n * k) as f64 / n as f64; // = k for capacity-blind setups
    let d = discrepancy(&loads, expected);

    // Fourier low-frequency band: first 1/16 of the spectrum.
    let band = ((n / 16).max(1)).min(n / 2);
    let lf = low_freq_energy_share(&loads_f, band);

    let routes_per_row = split_routes(&routes, n, k);
    // Sample a few hundred random pairs (independent of n; cost dominated by
    // small per-row Vec sort+merge).
    let overlap = pairwise_candidate_overlap(&routes_per_row, 256, overlap_seed);
    let corr = pearson(&loads_f, &w.rel_caps);
    let valid = if method == "hash" {
        // hash routes are not constrained to source support; reporting 1.0
        // would be misleading. Mark as -1.0 so CSV consumers can ignore it.
        -1.0
    } else {
        support_validity(&routes_per_row, &w.s_bits, w.nb_words)
    };

    Report {
        method,
        workload: w.label.clone(),
        n,
        k,
        mean_load: loads_f.iter().sum::<f64>() / n as f64,
        cv: d.cv,
        range_over_mean: d.range_over_mean,
        max_over_mean: d.max_over_mean,
        max_abs_dev: d.max_abs_dev,
        low_freq_share: lf,
        mean_jaccard: overlap.mean_jaccard,
        max_jaccard: overlap.max_jaccard,
        corr_load_cap: corr,
        support_valid: valid,
        time_ms: elapsed_ms,
    }
}

fn run_method<F>(
    method: &'static str,
    mut f: F,
    w: &Workload,
    k: usize,
    seed: u64,
    iters: usize,
    overlap_seed: u64,
) -> Report
where
    F: FnMut(u64) -> Vec<i32>,
{
    // warm-up
    let _ = f(seed);
    let mut last = Vec::new();
    let start = Instant::now();
    for it in 0..iters {
        last = f(seed.wrapping_add(it as u64));
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    analyse(method, last, w, k, elapsed_ms, overlap_seed)
}

fn print_header(label: &str) {
    println!("\n━━━ {label} ━━━");
    println!(
        "{:<26} {:<26} {:>6} {:>7} {:>7} {:>7} {:>9} {:>7} {:>7} {:>9} {:>7} {:>7}",
        "method",
        "workload",
        "mean",
        "CV",
        "rng/μ",
        "max/μ",
        "max|Δ|",
        "lf%",
        "J̄",
        "corr(L,c)",
        "valid",
        "ms"
    );
    println!("{}", "-".repeat(140));
}

fn print_row(r: &Report) {
    println!(
        "{:<26} {:<26} {:>6.2} {:>7.3} {:>7.3} {:>7.3} {:>9.2} {:>7.3} {:>7.3} {:>9.3} {:>7.3} {:>7.2}",
        r.method,
        r.workload,
        r.mean_load,
        r.cv,
        r.range_over_mean,
        r.max_over_mean,
        r.max_abs_dev,
        r.low_freq_share,
        r.mean_jaccard,
        r.corr_load_cap,
        r.support_valid,
        r.time_ms,
    );
}

// ── main ────────────────────────────────────────────────────────────────

fn main() {
    let n = 1024;
    let k = 4;
    let density = 0.3;
    let base_density = 0.3;
    let seed = 42u64;
    let iters = 5;
    let overlap_seed = 0xBEEF_F00D;

    println!(
        "Spectral / structural probe — N={n}, k={k}, density={density}, base_density={base_density}, seed={seed}"
    );
    println!("Methods: hash | phase_router | rs_additive | rs_affine | rs_hybrid");
    println!("Workloads: dense_uniform | dense_diverse | sparse_diverse | block_local");
    println!("{}", "=".repeat(140));

    let mut all_reports: Vec<Report> = Vec::new();

    for profile in Profile::ALL {
        let w = profile.build(n, density, base_density, seed);
        print_header(&w.label);

        let r_hash = run_method(
            "hash",
            |s| hash_route(n, k, s),
            &w,
            k,
            seed,
            iters,
            overlap_seed,
        );
        print_row(&r_hash);

        let r_pr = run_method(
            "phase_router",
            |s| {
                phase_router(
                    &w.s_bits,
                    &w.t_bits,
                    w.n,
                    w.nb_words,
                    k,
                    &w.col_perm_s,
                    &w.col_perm_t,
                    s,
                )
            },
            &w,
            k,
            seed,
            iters,
            overlap_seed,
        );
        print_row(&r_pr);

        let r_add = run_method(
            "rs_additive",
            |s| {
                phase_router_rs(
                    &w.s_bits,
                    &w.t_bits,
                    w.n,
                    w.nb_words,
                    k,
                    &w.col_perm_s,
                    &w.col_perm_t,
                    s,
                )
            },
            &w,
            k,
            seed,
            iters,
            overlap_seed,
        );
        print_row(&r_add);

        let r_aff = run_method(
            "rs_affine",
            |s| {
                phase_router_rs_affine(
                    &w.s_bits,
                    &w.t_bits,
                    w.n,
                    w.nb_words,
                    k,
                    &w.col_perm_s,
                    &w.col_perm_t,
                    s,
                )
            },
            &w,
            k,
            seed,
            iters,
            overlap_seed,
        );
        print_row(&r_aff);

        let r_hyb = run_method(
            "rs_hybrid",
            |s| {
                phase_router_rs_hybrid(
                    &w.s_bits,
                    &w.t_bits,
                    w.n,
                    w.nb_words,
                    k,
                    &w.col_perm_s,
                    &w.col_perm_t,
                    s,
                )
            },
            &w,
            k,
            seed,
            iters,
            overlap_seed,
        );
        print_row(&r_hyb);

        all_reports.extend([r_hash, r_pr, r_add, r_aff, r_hyb]);
    }

    println!("\n{}", "=".repeat(140));
    println!(
        "Legend:\n\
         mean   = mean per-target load (≈ k)\n\
         CV     = std/mean of per-target load\n\
         rng/μ  = (max − min)/mean\n\
         max/μ  = peak hot-spot ratio\n\
         max|Δ| = max absolute deviation from expected uniform load\n\
         lf%    = fraction of non-DC spectral energy in the bottom 1/16 of bins\n\
         J̄      = mean Jaccard overlap of candidate sets between random row pairs\n\
         corr(L,c) = Pearson(load, relative_capacity); capacity-aligned ≈ 1.0\n\
         valid  = fraction of routed columns inside source support (-1 = N/A for hash)"
    );

    // ── CSV output ───────────────────────────────────────────────────────
    let csv_path = "spectral_probe.csv";
    let mut file = std::fs::File::create(csv_path).expect("Cannot create CSV");
    writeln!(
        file,
        "method,workload,n,k,mean_load,cv,range_over_mean,max_over_mean,max_abs_dev,\
         low_freq_share,mean_jaccard,max_jaccard,corr_load_cap,support_valid,time_ms"
    )
    .unwrap();
    for r in &all_reports {
        writeln!(
            file,
            "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            r.method,
            r.workload,
            r.n,
            r.k,
            r.mean_load,
            r.cv,
            r.range_over_mean,
            r.max_over_mean,
            r.max_abs_dev,
            r.low_freq_share,
            r.mean_jaccard,
            r.max_jaccard,
            r.corr_load_cap,
            r.support_valid,
            r.time_ms,
        )
        .unwrap();
    }
    println!("\nResults written to {csv_path}");
}
