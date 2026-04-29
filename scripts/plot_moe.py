#!/usr/bin/env python3
"""
Plot MoE benchmark results: Phase Router vs Hash routing.

Reads moe_results.csv and produces comparison charts.

Usage: python scripts/plot_moe.py
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 10

def load_csv(path="moe_results.csv"):
    data = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['n'] = int(row['n'])
            row['k'] = int(row['k'])
            row['time_ms'] = float(row['time_ms'])
            row['cv'] = float(row['cv'])
            row['max_min_ratio'] = float(row['max_min_ratio'])
            data.append(row)
    return data

def plot_cv_vs_n(data, output_dir="plots"):
    """Figure 1: Load CV vs N, faceted by distribution."""
    distributions = sorted(set(r['distribution'] for r in data))
    methods = ['PhaseRouter', 'Hash', 'ModularHash']
    colors = {'PhaseRouter': '#2196F3', 'Hash': '#F44336', 'ModularHash': '#FF9800'}
    markers = {'PhaseRouter': 'o', 'Hash': 's', 'ModularHash': '^'}

    fig, axes = plt.subplots(1, len(distributions), figsize=(4 * len(distributions), 4),
                             sharey=True, squeeze=False)

    for idx, dist in enumerate(distributions):
        ax = axes[0][idx]
        for method in methods:
            subset = [r for r in data if r['distribution'] == dist and r['method'] == method]
            subset.sort(key=lambda r: r['n'])
            ns = [r['n'] for r in subset]
            cvs = [r['cv'] for r in subset]
            ax.plot(ns, cvs, marker=markers[method], color=colors[method],
                    label=method, linewidth=2, markersize=6)

        ax.set_title(dist.replace('_', ' '), fontweight='bold')
        ax.set_xlabel('N (tokens/experts)')
        ax.set_xscale('log', base=2)
        ax.set_xticks([256, 512, 1024, 2048, 4096])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, alpha=0.3)

    axes[0][0].set_ylabel('Load CV (σ/μ) — lower is better')
    axes[0][-1].legend(loc='upper right', fontsize=8)

    fig.suptitle('Expert Load Skew: Phase Router vs Hash Routing (k=2)', fontweight='bold', y=1.02)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/moe_cv_vs_n.png', bbox_inches='tight')
    fig.savefig(f'{output_dir}/moe_cv_vs_n.svg', bbox_inches='tight')
    print(f"Saved {output_dir}/moe_cv_vs_n.png")

def plot_max_min_bar(data, output_dir="plots"):
    """Figure 2: Max/min load ratio bar chart for skewed distributions at N=1024."""
    target_n = 1024
    skewed_dists = ['zipf_1.0', 'zipf_2.0', 'pareto_80_20']
    methods = ['PhaseRouter', 'Hash', 'ModularHash']
    colors = {'PhaseRouter': '#2196F3', 'Hash': '#F44336', 'ModularHash': '#FF9800'}

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(skewed_dists))
    width = 0.25

    for i, method in enumerate(methods):
        ratios = []
        for dist in skewed_dists:
            row = next((r for r in data
                       if r['distribution'] == dist
                       and r['n'] == target_n
                       and r['method'] == method), None)
            ratios.append(row['max_min_ratio'] if row else 0)

        bars = ax.bar(x + i * width, ratios, width, label=method, color=colors[method],
                      edgecolor='white', linewidth=0.5)

        # Value labels
        for bar, val in zip(bars, ratios):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Max/Min Expert Load Ratio — lower is better')
    ax.set_title(f'Worst-Case Load Imbalance (N={target_n}, k=2)', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.replace('_', ' ') for d in skewed_dists])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/moe_max_min_bar.png', bbox_inches='tight')
    fig.savefig(f'{output_dir}/moe_max_min_bar.svg', bbox_inches='tight')
    print(f"Saved {output_dir}/moe_max_min_bar.png")

def plot_time_vs_cv(data, output_dir="plots"):
    """Figure 3: Speed vs Balance tradeoff scatter (all N, all distributions)."""
    methods = ['PhaseRouter', 'Hash', 'ModularHash']
    colors = {'PhaseRouter': '#2196F3', 'Hash': '#F44336', 'ModularHash': '#FF9800'}
    markers = {'PhaseRouter': 'o', 'Hash': 's', 'ModularHash': '^'}

    fig, ax = plt.subplots(figsize=(8, 6))

    for method in methods:
        subset = [r for r in data if r['method'] == method]
        times = [r['time_ms'] for r in subset]
        cvs = [r['cv'] for r in subset]
        sizes = [r['n'] / 50 for r in subset]  # scale marker by N

        ax.scatter(times, cvs, c=colors[method], marker=markers[method],
                   s=sizes, alpha=0.7, label=method, edgecolors='white', linewidth=0.5)

    ax.set_xlabel('Time (ms) — log scale')
    ax.set_ylabel('Load CV (σ/μ) — lower is better')
    ax.set_xscale('log')
    ax.set_title('Speed vs Balance Tradeoff (all N, all distributions, k=2)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate the ideal region
    ax.annotate('← better', xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=9, color='gray', style='italic')

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/moe_speed_vs_balance.png', bbox_inches='tight')
    fig.savefig(f'{output_dir}/moe_speed_vs_balance.svg', bbox_inches='tight')
    print(f"Saved {output_dir}/moe_speed_vs_balance.png")

def plot_cv_summary(data, output_dir="plots"):
    """Figure 4: Combined CV comparison — grouped bar chart at N=1024."""
    target_n = 1024
    distributions = ['uniform', 'zipf_1.0', 'zipf_2.0', 'pareto_80_20']
    methods = ['PhaseRouter', 'Hash', 'ModularHash']
    colors = {'PhaseRouter': '#2196F3', 'Hash': '#F44336', 'ModularHash': '#FF9800'}

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(distributions))
    width = 0.25

    for i, method in enumerate(methods):
        cvs = []
        for dist in distributions:
            row = next((r for r in data
                       if r['distribution'] == dist
                       and r['n'] == target_n
                       and r['method'] == method), None)
            cvs.append(row['cv'] if row else 0)

        bars = ax.bar(x + i * width, cvs, width, label=method, color=colors[method],
                      edgecolor='white', linewidth=0.5)

        for bar, val in zip(bars, cvs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    ax.set_ylabel('Load CV (σ/μ) — lower is better')
    ax.set_title(f'Expert Load Balance by Distribution (N={target_n}, k=2)', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([d.replace('_', ' ') for d in distributions])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(f'{output_dir}/moe_cv_summary.png', bbox_inches='tight')
    fig.savefig(f'{output_dir}/moe_cv_summary.svg', bbox_inches='tight')
    print(f"Saved {output_dir}/moe_cv_summary.png")

if __name__ == '__main__':
    data = load_csv()

    plot_cv_vs_n(data)
    plot_max_min_bar(data)
    plot_time_vs_cv(data)
    plot_cv_summary(data)

    print("\nAll plots generated in plots/ directory.")
