"""Compare two training runs (one per router) and write a summary report.

Usage:
    python train/compare.py runs/topk runs/phase -o reports/tiny.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(run_dir: Path) -> dict:
    train = []
    val = []
    final = None
    with open(run_dir / "metrics.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if row["phase"] == "train":
                train.append(row)
            elif row["phase"] == "val":
                val.append(row)
            elif row["phase"] == "val_final":
                final = row
    return {"train": train, "val": val, "final": final}


def plot_curve(out_path: Path, title: str, ylabel: str, series: list[tuple[str, list[int], list[float]]]):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, xs, ys in series:
        ax.plot(xs, ys, label=label, linewidth=2)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("topk_dir", type=str)
    p.add_argument("phase_dir", type=str)
    p.add_argument("-o", "--out", type=str, default="reports/comparison.md")
    args = p.parse_args()

    topk = load_metrics(Path(args.topk_dir))
    phase = load_metrics(Path(args.phase_dir))

    out_path = Path(args.out)
    plots_dir = out_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ---- curves ---------------------------------------------------------
    def series(metrics, key):
        xs = [r["step"] for r in metrics["train"] if key in r]
        ys = [r[key]    for r in metrics["train"] if key in r]
        return xs, ys

    def two_series(key):
        xs_t, ys_t = series(topk, key)
        xs_p, ys_p = series(phase, key)
        return [("top-k", xs_t, ys_t), ("phase", xs_p, ys_p)]

    plot_curve(plots_dir / "train_ce.png",
               "Train cross-entropy", "ce",
               two_series("ce"))
    plot_curve(plots_dir / "load_cv.png",
               "Expert load CV (lower = better balance)", "CV",
               two_series("load_cv"))
    plot_curve(plots_dir / "dropped.png",
               "Dropped-token fraction (lower = better)", "fraction",
               two_series("drop"))
    plot_curve(plots_dir / "contig_frac.png",
               "Adjacent-row expert overlap (higher = better locality)",
               "fraction",
               two_series("contig_frac"))
    plot_curve(plots_dir / "perm_entropy.png",
               "Routing entropy (normalised, 1.0 = uniform)",
               "H / log(E)",
               two_series("perm_entropy"))
    plot_curve(plots_dir / "route_time_ms.png",
               "Per-forward routing wall time (sum across MoE blocks)",
               "ms",
               two_series("route_time_ms"))

    # ---- summary --------------------------------------------------------
    def final_or_last(metrics, key):
        if metrics["final"] and key in metrics["final"]:
            return metrics["final"][key]
        if metrics["val"]:
            return metrics["val"][-1].get(key)
        return None

    def last_train(metrics, key):
        for r in reversed(metrics["train"]):
            if key in r:
                return r[key]
        return None

    rows = []
    rows.append(("final val CE",
                 final_or_last(topk, "val_ce"),
                 final_or_last(phase, "val_ce"),
                 "lower"))
    rows.append(("final val load CV",
                 final_or_last(topk, "val_cv"),
                 final_or_last(phase, "val_cv"),
                 "lower"))
    rows.append(("final val drop frac",
                 final_or_last(topk, "val_drop"),
                 final_or_last(phase, "val_drop"),
                 "lower"))
    rows.append(("final val contig_frac",
                 final_or_last(topk, "val_contig"),
                 final_or_last(phase, "val_contig"),
                 "higher"))
    rows.append(("final val perm_entropy",
                 final_or_last(topk, "val_pe"),
                 final_or_last(phase, "val_pe"),
                 "lower"))
    rows.append(("final val route_time_ms",
                 final_or_last(topk, "val_rt_ms"),
                 final_or_last(phase, "val_rt_ms"),
                 "lower"))

    rows.append(("end train tok/s",
                 last_train(topk, "tok_per_s"),
                 last_train(phase, "tok_per_s"),
                 "higher"))
    rows.append(("wall time (s)",
                 last_train(topk, "elapsed_s"),
                 last_train(phase, "elapsed_s"),
                 "lower"))

    # Derived: routing overhead percentage = route_time_ms / step_time_ms
    def routing_overhead(metrics):
        rt = last_train(metrics, "route_time_ms")
        st = last_train(metrics, "step_time_ms")
        if rt is None or st is None or st <= 0:
            return None
        return 100.0 * rt / st

    rows.append(("routing overhead %",
                 routing_overhead(topk),
                 routing_overhead(phase),
                 "lower"))

    lines = [
        f"# Comparison: top-k vs phase router",
        "",
        f"- top-k dir:  `{args.topk_dir}`",
        f"- phase dir:  `{args.phase_dir}`",
        "",
        "| metric | top-k | phase | winner |",
        "| --- | --- | --- | --- |",
    ]
    for name, a, b, direction in rows:
        if a is None or b is None:
            winner = "—"
            a_s = "—" if a is None else f"{a:.4f}"
            b_s = "—" if b is None else f"{b:.4f}"
        else:
            if direction == "higher":
                winner = "top-k" if a > b else "phase"
            else:  # lower
                winner = "top-k" if a < b else "phase"
            a_s = f"{a:.4f}"
            b_s = f"{b:.4f}"
        lines.append(f"| {name} | {a_s} | {b_s} | {winner} |")

    lines += [
        "",
        "## Metric definitions",
        "",
        "- **load CV**: std/mean of per-expert assignment counts. Lower = more balanced.",
        "- **drop frac**: fraction of (token, slot) assignments dropped by capacity. Lower is better.",
        "- **contig_frac**: fraction of adjacent token rows that share ≥1 chosen expert.",
        "  Higher = better routing locality (good proxy for MegaBlocks-style block-sparse dispatch).",
        "- **perm_entropy**: Shannon entropy of chosen experts, normalised by log(n_experts).",
        "  1.0 = uniform over experts; lower = more concentrated.",
        "- **route_time_ms**: total wall-time spent inside the router (summed across MoE blocks)",
        "  per forward pass. Includes capacity enforcement.",
        "- **routing overhead %**: 100 × route_time_ms / step_time_ms. Sanity-checks that the",
        "  router is not dominating step time.",
        "",
        "## Plots",
        "",
        "![Train CE](plots/train_ce.png)",
        "![Load CV](plots/load_cv.png)",
        "![Dropped](plots/dropped.png)",
        "![Contig frac](plots/contig_frac.png)",
        "![Perm entropy](plots/perm_entropy.png)",
        "![Route time](plots/route_time_ms.png)",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
