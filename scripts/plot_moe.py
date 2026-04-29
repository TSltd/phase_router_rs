#!/usr/bin/env python3
"""
Plot MoE benchmark results: Phase Router vs Uniform Hash.

Four charts matching the four experiments in moe_bench.rs:
  1. Survival vs Headroom (hero chart)
  2. Survival vs K
  3. Survival vs N
  4. Survival vs Heterogeneity

Usage: python scripts/plot_moe.py
"""

import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['font.family'] = 'sans-serif'

PR_COLOR = '#2196F3'
HASH_COLOR = '#F44336'


def load_csv(path="moe_results.csv"):
    data = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['survival_rate'] = float(row['survival_rate'])
            row['time_ms'] = float(row['time_ms'])
            data.append(row)
    return data


def get_exp(data, experiment, method=None):
    out = [r for r in data if r['experiment'] == experiment]
    if method:
        out = [r for r in out if r['method'] == method]
    return out


def plot_headroom_sweep(data, output_dir="plots"):
    """Figure 1 (HERO): Token survival vs capacity headroom."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, color, marker, label in [
        ('PhaseRouter', PR_COLOR, 'o', 'Phase Router'),
        ('UniformHash', HASH_COLOR, 's', 'Uniform Hash'),
    ]:
        rows = get_exp(data, 'headroom', method)
        rows.sort(key=lambda r: float(r['headroom']))
        xs = [float(r['headroom']) for r in rows]
        ys = [r['survival_rate'] * 100 for r in rows]
        ax.plot(xs, ys, marker=marker, color=color, label=label,
                linewidth=2.5, markersize=8)

    ax.set_xlabel('Capacity Headroom (1.0 = exact fit)', fontsize=12)
    ax.set_ylabel('Token Survival Rate (%)', fontsize=12)
    ax.set_title('Phase Router Needs Less Overprovisioning\n(N=1024, k=2, strong heterogeneous capacity)',
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(None, 102)
    ax.axhline(y=100, color='gray', linestyle=':', alpha=0.5)

    # Annotate the gap
    pr_rows = get_exp(data, 'headroom', 'PhaseRouter')
    h_rows = get_exp(data, 'headroom', 'UniformHash')
    if pr_rows and h_rows:
        pr_rows.sort(key=lambda r: float(r['headroom']))
        h_rows.sort(key=lambda r: float(r['headroom']))
        # Find midpoint for annotation
        mid = len(pr_rows) // 2
        pr_y = pr_rows[mid]['survival_rate'] * 100
        h_y = h_rows[mid]['survival_rate'] * 100
        hr_x = float(pr_rows[mid]['headroom'])
        if pr_y > h_y + 2:
            ax.annotate(f'  +{pr_y - h_y:.1f}%',
                        xy=(hr_x, (pr_y + h_y) / 2),
                        fontsize=10, color='#333', fontweight='bold')

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/moe_headroom.png', bbox_inches='tight')
    fig.savefig(f'{output_dir}/moe_headroom.svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  → {output_dir}/moe_headroom.png")


def plot_k_sweep(data, output_dir="plots"):
    """Figure 2: Token survival vs fan-out k."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for method, color, marker, label in [
        ('PhaseRouter', PR_COLOR, 'o', 'Phase Router'),
        ('UniformHash', HASH_COLOR, 's', 'Uniform Hash'),
    ]:
        rows = get_exp(data, 'k_sweep', method)
        rows.sort(key=lambda r: int(r['k']))
        xs = [int(r['k']) for r in rows]
        ys = [r['survival_rate'] * 100 for r in rows]
        ax.plot(xs, ys, marker=marker, color=color, label=label,
                linewidth=2.5, markersize=8)

    ax.set_xlabel('Fan-out k (experts per token)', fontsize=12)
    ax.set_ylabel('Token Survival Rate (%)', fontsize=12)
    ax.set_title('Phase Router Advantage Grows with Fan-out\n(N=1024, headroom=1.2×, strong hetero)',
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/moe_k_sweep.png', bbox_inches='tight')
    fig.savefig(f'{output_dir}/moe_k_sweep.svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  → {output_dir}/moe_k_sweep.png")


def plot_scale_sweep(data, output_dir="plots"):
    """Figure 3: Token survival vs N."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for method, color, marker, label in [
        ('PhaseRouter', PR_COLOR, 'o', 'Phase Router'),
        ('UniformHash', HASH_COLOR, 's', 'Uniform Hash'),
    ]:
        rows = get_exp(data, 'scale', method)
        rows.sort(key=lambda r: int(r['n']))
        xs = [int(r['n']) for r in rows]
        ys = [r['survival_rate'] * 100 for r in rows]
        ax.plot(xs, ys, marker=marker, color=color, label=label,
                linewidth=2.5, markersize=8)

    ax.set_xlabel('N (tokens = experts)', fontsize=12)
    ax.set_ylabel('Token Survival Rate (%)', fontsize=12)
    ax.set_title('Phase Router Advantage Across Scale\n(k=2, headroom=1.2×, strong hetero)',
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks([256, 512, 1024, 2048, 4096])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/moe_scale.png', bbox_inches='tight')
    fig.savefig(f'{output_dir}/moe_scale.svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  → {output_dir}/moe_scale.png")


def plot_hetero_sweep(data, output_dir="plots"):
    """Figure 4: Token survival vs heterogeneity level — grouped bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    scenarios = ['uniform', 'mild_hetero', 'strong_hetero', 'extreme_hetero']
    scenario_labels = ['Uniform', 'Mild\n(20% @ 3×)', 'Strong\n(10%@8× 20%@2×)', 'Extreme\n(5% @ 16×)']

    x = np.arange(len(scenarios))
    width = 0.35

    for i, (method, color, label) in enumerate([
        ('PhaseRouter', PR_COLOR, 'Phase Router'),
        ('UniformHash', HASH_COLOR, 'Uniform Hash'),
    ]):
        rows = get_exp(data, 'hetero', method)
        vals = []
        for scen in scenarios:
            row = next((r for r in rows if r['param'] == scen), None)
            vals.append(row['survival_rate'] * 100 if row else 0)

        bars = ax.bar(x + i * width - width / 2, vals, width, label=label,
                      color=color, edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Token Survival Rate (%)', fontsize=12)
    ax.set_title('Phase Router Advantage Grows with Capacity Heterogeneity\n(N=1024, k=2, headroom=1.2×)',
                 fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 108)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/moe_hetero.png', bbox_inches='tight')
    fig.savefig(f'{output_dir}/moe_hetero.svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  → {output_dir}/moe_hetero.png")


def plot_combined(data, output_dir="plots"):
    """Figure 5: 2×2 combined summary."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods_cfg = [
        ('PhaseRouter', PR_COLOR, 'o', 'Phase Router'),
        ('UniformHash', HASH_COLOR, 's', 'Uniform Hash'),
    ]

    # (0,0) Headroom
    ax = axes[0][0]
    for method, color, marker, label in methods_cfg:
        rows = sorted(get_exp(data, 'headroom', method), key=lambda r: float(r['headroom']))
        ax.plot([float(r['headroom']) for r in rows],
                [r['survival_rate'] * 100 for r in rows],
                marker=marker, color=color, label=label, linewidth=2, markersize=6)
    ax.set_xlabel('Capacity Headroom')
    ax.set_ylabel('Survival (%)')
    ax.set_title('a) Headroom Sweep', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(100, color='gray', ls=':', alpha=0.4)

    # (0,1) K sweep
    ax = axes[0][1]
    for method, color, marker, label in methods_cfg:
        rows = sorted(get_exp(data, 'k_sweep', method), key=lambda r: int(r['k']))
        ax.plot([int(r['k']) for r in rows],
                [r['survival_rate'] * 100 for r in rows],
                marker=marker, color=color, label=label, linewidth=2, markersize=6)
    ax.set_xlabel('Fan-out k')
    ax.set_ylabel('Survival (%)')
    ax.set_title('b) K Sweep', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # (1,0) Scale sweep
    ax = axes[1][0]
    for method, color, marker, label in methods_cfg:
        rows = sorted(get_exp(data, 'scale', method), key=lambda r: int(r['n']))
        ax.plot([int(r['n']) for r in rows],
                [r['survival_rate'] * 100 for r in rows],
                marker=marker, color=color, label=label, linewidth=2, markersize=6)
    ax.set_xlabel('N')
    ax.set_ylabel('Survival (%)')
    ax.set_title('c) Scale Sweep', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks([256, 512, 1024, 2048, 4096])
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # (1,1) Hetero bar
    ax = axes[1][1]
    scenarios = ['uniform', 'mild_hetero', 'strong_hetero', 'extreme_hetero']
    labels = ['Uniform', 'Mild', 'Strong', 'Extreme']
    x = np.arange(len(scenarios))
    w = 0.35
    for i, (method, color, _, lbl) in enumerate(methods_cfg):
        rows = get_exp(data, 'hetero', method)
        vals = []
        for s in scenarios:
            row = next((r for r in rows if r['param'] == s), None)
            vals.append(row['survival_rate'] * 100 if row else 0)
        ax.bar(x + i * w - w / 2, vals, w, label=lbl, color=color, edgecolor='white')
    ax.set_ylabel('Survival (%)')
    ax.set_title('d) Heterogeneity Sweep', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 108)

    fig.suptitle('Phase Router vs Hash Routing: Capacity-Constrained MoE',
                 fontweight='bold', fontsize=14, y=1.01)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/moe_combined.png', bbox_inches='tight')
    fig.savefig(f'{output_dir}/moe_combined.svg', bbox_inches='tight')
    plt.close(fig)
    print(f"  → {output_dir}/moe_combined.png")


if __name__ == '__main__':
    data = load_csv()

    print("Generating plots...")
    plot_headroom_sweep(data)
    plot_k_sweep(data)
    plot_scale_sweep(data)
    plot_hetero_sweep(data)
    plot_combined(data)

    print("\nAll plots saved to plots/ directory.")
