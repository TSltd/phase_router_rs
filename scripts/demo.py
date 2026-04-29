#!/usr/bin/env python3
"""
Phase Router Demo — Compare vs Hash Routing, Show Load Skew Reduction

Run:  python scripts/demo.py
Requires:  maturin develop --release  (to build the Rust bindings)

Demonstrates:
  1. Phase Router vs hash routing under heterogeneous expert capacities
  2. Load distribution comparison (before capacity enforcement)
  3. Token survival after capacity enforcement
  4. Load skew metrics (CV, max/mean ratio)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from phase_router import (
    route,
    hash_route,
    enforce_capacity,
    compute_loads,
    load_stats,
    survival_rate,
    strong_hetero_capacities,
    make_hard_caps,
)


def bar(value, max_val, width=40):
    """Simple text bar chart."""
    filled = int(round(value / max_val * width)) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def print_header(title):
    print(f"\n{'━' * 70}")
    print(f"  {title}")
    print(f"{'━' * 70}")


def demo_comparison():
    """Main demo: Phase Router vs Hash under heterogeneous capacities."""

    n = 512
    k = 4
    seed = 42
    headroom = 1.2

    print_header("Phase Router Demo — MoE Expert Routing")
    print(f"  Experts: {n}  |  Fan-out: k={k}  |  Headroom: {headroom}×")
    print(f"  Capacity profile: strong heterogeneous (10% @ 8×, 20% @ 2×, rest @ 1×)")
    print(f"  Total route slots: {n * k}")

    # ── Setup ────────────────────────────────────────────────────────

    caps = strong_hetero_capacities(n, seed=seed)
    hard_caps = make_hard_caps(caps, n, k, headroom)

    # ── Route ────────────────────────────────────────────────────────

    pr_routes = route(target_capacities=caps, n=n, k=k, seed=seed)
    h_routes = hash_route(n, k, seed=seed)

    # ── Load analysis BEFORE capacity enforcement ────────────────────

    print_header("1. Load Distribution (BEFORE capacity enforcement)")

    pr_loads = compute_loads(pr_routes, n)
    h_loads = compute_loads(h_routes, n)

    pr_stats = load_stats(pr_loads)
    h_stats = load_stats(h_loads)

    print(f"\n  {'Metric':<20} {'Phase Router':>14} {'Hash':>14}")
    print(f"  {'─' * 48}")
    print(f"  {'Mean load':<20} {pr_stats['mean']:>14.2f} {h_stats['mean']:>14.2f}")
    print(f"  {'Std dev':<20} {pr_stats['std']:>14.2f} {h_stats['std']:>14.2f}")
    print(f"  {'Max load':<20} {pr_stats['max']:>14d} {h_stats['max']:>14d}")
    print(f"  {'Min load':<20} {pr_stats['min']:>14d} {h_stats['min']:>14d}")
    print(f"  {'CV (std/mean)':<20} {pr_stats['cv']:>14.3f} {h_stats['cv']:>14.3f}")
    print(f"  {'Max/Mean ratio':<20} {pr_stats['max_over_mean']:>14.2f} {h_stats['max_over_mean']:>14.2f}")

    # Show correlation between capacity and load
    corr_pr = np.corrcoef(caps, pr_loads)[0, 1]
    corr_h = np.corrcoef(caps, h_loads)[0, 1]

    print(f"\n  {'Load–Capacity corr':<20} {corr_pr:>14.3f} {corr_h:>14.3f}")
    print(f"  (Phase Router targets corr ≈ 1.0 — load tracks capacity)")

    # ── Top 10 experts by capacity: show load alignment ──────────────

    print_header("2. Load Alignment — Top 10 Experts by Capacity")

    sorted_idx = np.argsort(caps)[::-1][:10]
    max_load = max(pr_loads.max(), h_loads.max())

    print(f"\n  {'Expert':>6} {'Cap':>6} {'PR Load':>8} {'Hash Load':>10}  PR Load Bar")
    print(f"  {'─' * 70}")
    for idx in sorted_idx:
        print(
            f"  {idx:>6d} {caps[idx]:>6.1f} {pr_loads[idx]:>8d} {h_loads[idx]:>10d}  "
            f"{bar(pr_loads[idx], max_load, 30)}"
        )

    # ── Bottom 10 experts (lowest capacity) ──────────────────────────

    print(f"\n  ... Bottom 10 (lowest capacity):")
    bottom_idx = np.argsort(caps)[:10]
    for idx in bottom_idx:
        print(
            f"  {idx:>6d} {caps[idx]:>6.1f} {pr_loads[idx]:>8d} {h_loads[idx]:>10d}  "
            f"{bar(pr_loads[idx], max_load, 30)}"
        )

    # ── Capacity enforcement ─────────────────────────────────────────

    print_header("3. Token Survival (AFTER capacity enforcement)")

    pr_enforced = enforce_capacity(pr_routes, hard_caps)
    h_enforced = enforce_capacity(h_routes, hard_caps)

    pr_surv = survival_rate(pr_enforced, n)
    h_surv = survival_rate(h_enforced, n)

    pr_dropped = np.sum(pr_enforced.ravel() == -1)
    h_dropped = np.sum(h_enforced.ravel() == -1)

    total_slots = n * k

    print(f"\n  {'Metric':<25} {'Phase Router':>14} {'Hash':>14}")
    print(f"  {'─' * 53}")
    print(f"  {'Survival rate':<25} {pr_surv:>13.1%} {h_surv:>13.1%}")
    print(f"  {'Tokens assigned':<25} {total_slots - pr_dropped:>14d} {total_slots - h_dropped:>14d}")
    print(f"  {'Tokens dropped':<25} {pr_dropped:>14d} {h_dropped:>14d}")
    print(f"  {'Advantage':<25} {'+' + f'{(pr_surv - h_surv):.1%}':>14}")

    # ── Visual: survival bars ────────────────────────────────────────

    print(f"\n  Phase Router: {bar(pr_surv, 1.0, 50)} {pr_surv:.1%}")
    print(f"  Hash Routing: {bar(h_surv, 1.0, 50)} {h_surv:.1%}")

    # ── Load skew after enforcement ──────────────────────────────────

    print_header("4. Load Skew After Enforcement")

    pr_post = compute_loads(pr_enforced, n)
    h_post = compute_loads(h_enforced, n)

    pr_post_stats = load_stats(pr_post)
    h_post_stats = load_stats(h_post)

    # Compute overload: how many experts hit their cap
    pr_at_cap = np.sum(pr_post >= hard_caps)
    h_at_cap = np.sum(h_post >= hard_caps)

    print(f"\n  {'Metric':<25} {'Phase Router':>14} {'Hash':>14}")
    print(f"  {'─' * 53}")
    print(f"  {'CV (std/mean)':<25} {pr_post_stats['cv']:>14.3f} {h_post_stats['cv']:>14.3f}")
    print(f"  {'Max/Mean':<25} {pr_post_stats['max_over_mean']:>14.2f} {h_post_stats['max_over_mean']:>14.2f}")
    print(f"  {'Experts at cap limit':<25} {pr_at_cap:>14d} {h_at_cap:>14d}")
    print(f"  {'Utilisation (load/cap)':<25} "
          f"{np.mean(pr_post / hard_caps):>13.1%} "
          f"{np.mean(h_post / hard_caps):>13.1%}")

    # ── Plot: Capacity vs Load ───────────────────────────────────────

    print_header("5. Generating Capacity vs Load Plot")

    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

   # Experts, k
    n_ex = 512
    k_ex = 4

    # Construct capacities with multiple experts per level
    levels = np.arange(1, 9)
    caps_ex = np.repeat(levels, n_ex // len(levels)).astype(np.float64)
    n_ex = len(caps_ex)

    # Run once (no need for 20 seeds anymore)
    pr_r = route(target_capacities=caps_ex, n=n_ex, k=k_ex, seed=42)
    h_r = hash_route(n_ex, k_ex, seed=42)

    pr_loads = compute_loads(pr_r, n_ex)
    h_loads = compute_loads(h_r, n_ex)

    # Aggregate by capacity
    pr_avg = [pr_loads[caps_ex == c].mean() for c in levels]
    h_avg = [h_loads[caps_ex == c].mean() for c in levels]

    # Theoretical expectation
    total_mass = n_ex * k_ex
    expected = total_mass * levels / caps_ex.sum()

    ax.plot(levels, pr_avg, "o-", label="Phase Router", linewidth=2)
    ax.plot(levels, h_avg, "s-", label="Hash Routing", linewidth=2)
    ax.plot(levels, expected, "--", label="Expected (∝ capacity)", linewidth=2)

    ax.set_xticks(caps_ex)
    ax.set_xlabel("Capacity", fontsize=14)
    ax.set_ylabel("Load", fontsize=14)
    ax.set_title("Capacity vs Load", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = "plots/capacity_vs_load.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved → {plot_path}")

    # ── Plot 2: Load/Capacity ratio vs Capacity ─────────────────────

    print_header("6. Generating Load/Capacity Ratio Plot")

    fig2, ax2 = plt.subplots(figsize=(7, 5))

    pr_ratio = np.array(pr_avg) / levels
    h_ratio = np.array(h_avg) / levels

    ax2.plot(levels, pr_ratio, "o-", label="Phase Router", linewidth=2)
    ax2.plot(levels, h_ratio, "s-", label="Hash Routing", linewidth=2)

    ax2.set_xticks(levels)
    ax2.set_xlabel("Capacity", fontsize=14)
    ax2.set_ylabel("Load / Capacity", fontsize=14)
    ax2.set_title("Load / Capacity vs Capacity", fontsize=15)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    plot_path2 = "plots/load_capacity_ratio.png"
    fig2.savefig(plot_path2, dpi=150)
    plt.close(fig2)
    print(f"  Saved → {plot_path2}")

    # ── Insight ──────────────────────────────────────────────────────

    print_header("Key Insight")
    print(f"""
  Phase Router distributes load proportional to expert capacity:
    E[load_j] ∝ capacity_j   (by construction of cyclic phase embedding)

  Hash routing distributes uniformly regardless of capacity:
    E[load_j] = k             (ignores capacity entirely)

  Result: Phase Router drops {pr_dropped} tokens vs {h_dropped} for hash.
  That's {h_dropped - pr_dropped} fewer dropped tokens — a {(pr_surv - h_surv):.1%} survival improvement.

  In production, each dropped token = wasted compute or degraded model quality.
  Phase Router eliminates this waste with zero coordination overhead.
""")


if __name__ == "__main__":
    demo_comparison()
