//! MoE benchmark — rank/select kernel head-to-head with the original phase
//! router and uniform hash, sweeping the four canonical workload profiles.
//!
//! Methods (5):
//!   - `hash`      uniform DefaultHasher
//!   - `pr`        original `phase_router`
//!   - `rs_add`    additive rank-space transport
//!   - `rs_aff`    affine   rank-space transport
//!   - `rs_hyb`    hybrid: occupancy walk via `offsets_s` + intrinsic decode
//!
//! Workloads (4): dense_uniform, dense_diverse, sparse_diverse, block_local
//!
//! Run:
//!   cargo run --release --features rank-select --example moe_bench_rs

use phase_router_rs::metrics::{discrepancy, loads_per_target, low_freq_energy_share};
use phase_router_rs::router::phase_router;
use phase_router_rs::router_rs::{
    phase_router_rs, phase_router_rs_affine, phase_router_rs_hybrid,
};
use phase_router_rs::workloads::{Profile, Workload};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::time::Instant;

// ── Uniform hash routing baseline ────────────────────────────────────────

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

fn survival_rate(routes: &[i32], n: usize, k: usize) -> f64 {
    let assigned = routes
        .iter()
        .filter(|&&t| t >= 0 && (t as usize) < n)
        .count();
    assigned as f64 / (n * k) as f64
}

// ── one comparison row ──────────────────────────────────────────────────

#[derive(Default, Clone, Copy)]
struct MethodResult {
    survival: f64,
    cv: f64,
    max_over_mean: f64,
    low_freq_share: f64,
    time_ms: f64,
}

struct Comparison {
    hash: MethodResult,
    pr: MethodResult,
    rs_add: MethodResult,
    rs_aff: MethodResult,
    rs_hyb: MethodResult,
}

fn time_and_score<F: FnMut(u64) -> Vec<i32>>(
    mut f: F,
    seed: u64,
    iters: usize,
    n: usize,
    k: usize,
    hard_caps: &[usize],
) -> MethodResult {
    let _ = f(seed); // warm-up
    let mut last = Vec::new();
    let start = Instant::now();
    for it in 0..iters {
        last = f(seed + it as u64);
    }
    let time_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Pre-cap structural metrics on the raw routes (so capacity enforcement
    // doesn't distort discrepancy / spectral signals).
    let raw_loads = loads_per_target(&last, n);
    let d = discrepancy(&raw_loads, k as f64);
    let raw_loads_f: Vec<f64> = raw_loads.iter().map(|&x| x as f64).collect();
    let band = ((n / 16).max(1)).min(n / 2);
    let lf = low_freq_energy_share(&raw_loads_f, band);

    enforce_capacity(&mut last, n, hard_caps);
    MethodResult {
        survival: survival_rate(&last, n, k),
        cv: d.cv,
        max_over_mean: d.max_over_mean,
        low_freq_share: lf,
        time_ms,
    }
}

fn run_comparison(w: &Workload, k: usize, headroom: f64, seed: u64) -> Comparison {
    let n = w.n;
    let nb_words = w.nb_words;
    let iters = if n <= 512 { 10 } else { 3 };

    let total_cap: f64 = w.rel_caps.iter().sum();
    let hard_caps: Vec<usize> = w
        .rel_caps
        .iter()
        .map(|&c| ((c / total_cap) * (n * k) as f64 * headroom).ceil().max(1.0) as usize)
        .collect();

    let hash = time_and_score(
        |s| hash_route(n, k, s),
        seed,
        iters,
        n,
        k,
        &hard_caps,
    );
    let pr = time_and_score(
        |s| {
            phase_router(
                &w.s_bits,
                &w.t_bits,
                n,
                nb_words,
                k,
                &w.col_perm_s,
                &w.col_perm_t,
                s,
            )
        },
        seed,
        iters,
        n,
        k,
        &hard_caps,
    );
    let rs_add = time_and_score(
        |s| {
            phase_router_rs(
                &w.s_bits,
                &w.t_bits,
                n,
                nb_words,
                k,
                &w.col_perm_s,
                &w.col_perm_t,
                s,
            )
        },
        seed,
        iters,
        n,
        k,
        &hard_caps,
    );
    let rs_aff = time_and_score(
        |s| {
            phase_router_rs_affine(
                &w.s_bits,
                &w.t_bits,
                n,
                nb_words,
                k,
                &w.col_perm_s,
                &w.col_perm_t,
                s,
            )
        },
        seed,
        iters,
        n,
        k,
        &hard_caps,
    );
    let rs_hyb = time_and_score(
        |s| {
            phase_router_rs_hybrid(
                &w.s_bits,
                &w.t_bits,
                n,
                nb_words,
                k,
                &w.col_perm_s,
                &w.col_perm_t,
                s,
            )
        },
        seed,
        iters,
        n,
        k,
        &hard_caps,
    );

    Comparison {
        hash,
        pr,
        rs_add,
        rs_aff,
        rs_hyb,
    }
}

fn print_header(label: &str) {
    println!("\n━━━ {label} ━━━\n");
    println!(
        "  {:<14}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "param", "hash", "pr", "rs_add", "rs_aff", "rs_hyb"
    );
}

fn print_row(label: &str, r: &Comparison) {
    println!(
        "  {:<14}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
        label,
        r.hash.survival,
        r.pr.survival,
        r.rs_add.survival,
        r.rs_aff.survival,
        r.rs_hyb.survival
    );
}

fn append_csv(
    csv: &mut Vec<String>,
    workload: &str,
    experiment: &str,
    param: &str,
    n: usize,
    k: usize,
    headroom: f64,
    r: &Comparison,
) {
    for (m, v) in [
        ("hash", r.hash),
        ("PhaseRouter", r.pr),
        ("PhaseRouterRsAdd", r.rs_add),
        ("PhaseRouterRsAff", r.rs_aff),
        ("PhaseRouterRsHyb", r.rs_hyb),
    ] {
        csv.push(format!(
            "{},{},{},{},{},{:.2},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            workload,
            experiment,
            param,
            n,
            k,
            headroom,
            m,
            v.survival,
            v.cv,
            v.max_over_mean,
            v.low_freq_share,
            v.time_ms,
        ));
    }
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    let seed = 42u64;
    let density = 0.3;
    let base_density = 0.3;

    let mut csv = vec![
        "workload,experiment,param,n,k,headroom,method,survival_rate,cv,max_over_mean,low_freq_share,time_ms"
            .to_string(),
    ];

    println!("MoE Benchmark (rank/select edition, workload sweep)");
    println!(
        "Methods: hash | phase_router (pr) | rs_additive (rs_add) | rs_affine (rs_aff) | rs_hybrid (rs_hyb)"
    );
    println!("Workloads: dense_uniform | dense_diverse | sparse_diverse | block_local");
    println!("{}", "=".repeat(110));

    for profile in Profile::ALL {
        println!(
            "\n############ workload: {} ############",
            profile.name()
        );

        // ────────────────────────────────────────────────────────────
        // Experiment 1: Headroom sweep (N=1024, k=2)
        // ────────────────────────────────────────────────────────────
        let n = 1024;
        let k = 2;
        let headrooms = [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0];
        let w = profile.build(n, density, base_density, seed);
        print_header(&format!(
            "[{}] Experiment 1: Headroom Sweep (N={n}, k={k}) — survival rate",
            w.label
        ));
        for &hr in &headrooms {
            let r = run_comparison(&w, k, hr, seed);
            print_row(&format!("hr={hr:.2}"), &r);
            append_csv(
                &mut csv,
                profile.name(),
                "headroom",
                &format!("{hr:.2}"),
                n,
                k,
                hr,
                &r,
            );
        }

        // ────────────────────────────────────────────────────────────
        // Experiment 2: K sweep
        // ────────────────────────────────────────────────────────────
        let n = 1024;
        let hr = 1.2;
        let ks = [1, 2, 4, 8, 16];
        let w = profile.build(n, density, base_density, seed);
        print_header(&format!(
            "[{}] Experiment 2: K Sweep (N={n}, headroom={hr}) — survival rate",
            w.label
        ));
        for &k in &ks {
            let r = run_comparison(&w, k, hr, seed);
            print_row(&format!("k={k}"), &r);
            append_csv(
                &mut csv,
                profile.name(),
                "k_sweep",
                &format!("{k}"),
                n,
                k,
                hr,
                &r,
            );
        }

        // ────────────────────────────────────────────────────────────
        // Experiment 3: Scale sweep
        // ────────────────────────────────────────────────────────────
        let k = 2;
        let hr = 1.2;
        let sizes = [256, 512, 1024, 2048, 4096];
        print_header(&format!(
            "[{}] Experiment 3: Scale Sweep (k={k}, headroom={hr}) — survival rate",
            profile.name()
        ));
        for &n in &sizes {
            let w = profile.build(n, density, base_density, seed);
            let r = run_comparison(&w, k, hr, seed);
            print_row(&format!("N={n}"), &r);
            append_csv(
                &mut csv,
                profile.name(),
                "scale",
                &format!("{n}"),
                n,
                k,
                hr,
                &r,
            );
        }

        // Timing snapshot at the largest scale.
        let n_big = 4096;
        let k = 2;
        let hr = 1.2;
        let w = profile.build(n_big, density, base_density, seed);
        let r = run_comparison(&w, k, hr, seed);
        println!(
            "\n━━━ [{}] Timing snapshot at N={n_big} (k={k}, hr={hr}) ━━━\n",
            profile.name()
        );
        println!("  hash    : {:>7.3} ms", r.hash.time_ms);
        println!("  pr      : {:>7.3} ms", r.pr.time_ms);
        println!("  rs_add  : {:>7.3} ms", r.rs_add.time_ms);
        println!("  rs_aff  : {:>7.3} ms", r.rs_aff.time_ms);
        println!("  rs_hyb  : {:>7.3} ms", r.rs_hyb.time_ms);
    }

    println!("\n{}", "=".repeat(110));

    let csv_path = "moe_results_rs.csv";
    let mut file = std::fs::File::create(csv_path).expect("Cannot create CSV");
    for line in &csv {
        writeln!(file, "{line}").expect("Cannot write CSV");
    }
    println!("\nResults written to {csv_path}");
}
