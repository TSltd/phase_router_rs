"""Aggregate a capacity-factor sweep into one summary table + figure.

Expected layout of `sweep_dir`:

    <sweep_dir>/
      topk_cf1.00/metrics.jsonl
      topk_cf1.25/metrics.jsonl
      topk_cf1.50/metrics.jsonl
      topk_cf2.00/metrics.jsonl
      phase_cf1.00/metrics.jsonl
      ...
      topk_cf1.00_noaux/metrics.jsonl
      topk_cf1.25_noaux/metrics.jsonl

Each subdir's name is parsed for (router, cf, ablation). Produces:

    <out>                       — markdown summary
    <out>.parent/sweep_plots/   — one PNG per metric, x-axis = cf,
                                  one series per (router, ablation).

Usage:
    python train/compare_sweep.py runs/cf_sweep -o runs/cf_sweep/sweep.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUBDIR_RE = re.compile(
    r"^(?P<router>topk|phase|balanced)_cf(?P<cf>\d+\.\d+)(?P<ablation>_noaux)?$"
)


def parse_subdir(name: str) -> tuple[str, float, str] | None:
    m = SUBDIR_RE.match(name)
    if not m:
        return None
    router = m.group("router")
    cf = float(m.group("cf"))
    ablation = m.group("ablation") or ""
    series = f"{router}{ablation}"   # e.g. "topk", "topk_noaux", "phase"
    return series, cf, name


def load_metrics(run_dir: Path) -> dict:
    """Read final val + last train row from one run."""
    final = None
    last_train = None
    last_val = None
    fp = run_dir / "metrics.jsonl"
    if not fp.exists():
        return {}
    with open(fp) as f:
        for line in f:
            row = json.loads(line)
            ph = row.get("phase")
            if ph == "train":
                last_train = row
            elif ph == "val":
                last_val = row
            elif ph == "val_final":
                final = row
    out: dict = {}
    if final:
        out.update({k: v for k, v in final.items() if k.startswith("val_")})
    elif last_val:
        out.update({k: v for k, v in last_val.items() if k.startswith("val_")})
    if last_train:
        out["end_tok_per_s"] = last_train.get("tok_per_s")
        out["wall_time_s"]   = last_train.get("elapsed_s")
        out["route_time_ms"] = last_train.get("route_time_ms")
        out["step_time_ms"]  = last_train.get("step_time_ms")
    return out


def collect(sweep_dir: Path) -> list[dict]:
    rows = []
    for sub in sorted(sweep_dir.iterdir()):
        if not sub.is_dir():
            continue
        parsed = parse_subdir(sub.name)
        if parsed is None:
            continue
        series, cf, name = parsed
        m = load_metrics(sub)
        if not m:
            print(f"  skip (no metrics.jsonl): {sub}")
            continue
        rows.append({"series": series, "cf": cf, "name": name, **m})
    return rows


# Plot keys: (metric_key, ylabel, title, lower_is_better)
PLOT_METRICS = [
    ("val_ce",      "val cross-entropy",      "Final val CE vs capacity factor",       True),
    ("val_cv",      "load CV (std/mean)",     "Load imbalance vs capacity factor",     True),
    ("val_drop",    "drop fraction",          "Dropped-token rate vs capacity factor", True),
    ("val_contig",  "contig_frac",            "Routing locality vs capacity factor",   False),
    ("end_tok_per_s","tokens / sec",          "Throughput vs capacity factor",         False),
    ("route_time_ms","route_time_ms",         "Routing wall-time vs capacity factor",  True),
]


SERIES_STYLE = {
    "topk":        ("top-k (aux=0.01)", "tab:blue",   "-",  "o"),
    "topk_noaux":  ("top-k (aux=0)",    "tab:cyan",   "--", "x"),
    "phase":       ("phase",            "tab:orange", "-",  "s"),
    "balanced":    ("balanced",         "tab:green",  "-",  "D"),
}


def plot_sweep(rows: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    series_names = sorted({r["series"] for r in rows})
    for key, ylabel, title, _ in PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(7, 4))
        for s in series_names:
            xs, ys = [], []
            for r in sorted([r for r in rows if r["series"] == s], key=lambda r: r["cf"]):
                v = r.get(key)
                if v is None:
                    continue
                xs.append(r["cf"]); ys.append(v)
            if not xs:
                continue
            label, color, ls, mk = SERIES_STYLE.get(s, (s, None, "-", "o"))
            ax.plot(xs, ys, label=label, color=color, linestyle=ls, marker=mk, linewidth=2)
        ax.set_xlabel("capacity factor")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{key}.png", dpi=130)
        plt.close(fig)


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.3f}"
        return f"{v:.4f}"
    return str(v)


def write_markdown(rows: list[dict], out_path: Path, sweep_dir: Path,
                   plots_dirname: str):
    rows_sorted = sorted(rows, key=lambda r: (r["series"], r["cf"]))

    lines = [
        f"# Capacity-factor sweep: `{sweep_dir.name}`",
        "",
        f"- sweep dir: `{sweep_dir}`",
        f"- {len(rows)} runs",
        "",
        "## Per-run summary",
        "",
        "| run | cf | val_ce | val_cv | val_drop | val_contig | val_pe | val_rt_ms | tok/s | wall s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows_sorted:
        lines.append(
            "| {name} | {cf:.2f} | {ce} | {cv} | {dr} | {cg} | {pe} | {rt} | {tps} | {ws} |".format(
                name=r["name"],
                cf=r["cf"],
                ce=fmt(r.get("val_ce")),
                cv=fmt(r.get("val_cv")),
                dr=fmt(r.get("val_drop")),
                cg=fmt(r.get("val_contig")),
                pe=fmt(r.get("val_pe")),
                rt=fmt(r.get("val_rt_ms")),
                tps=fmt(r.get("end_tok_per_s")),
                ws=fmt(r.get("wall_time_s")),
            )
        )

    # Head-to-head pivot at each cf (excluding _noaux ablations).
    lines += [
        "",
        "## Head-to-head: top-k vs phase at each capacity factor",
        "",
        "| cf | metric | top-k | phase | winner |",
        "| --- | --- | --- | --- | --- |",
    ]
    cfs = sorted({r["cf"] for r in rows if r["series"] in ("topk", "phase")})
    HH = [
        ("val_ce",     "lower"),
        ("val_cv",     "lower"),
        ("val_drop",   "lower"),
        ("val_contig", "higher"),
        ("end_tok_per_s", "higher"),
    ]
    for cf in cfs:
        topk = next((r for r in rows if r["series"] == "topk"  and r["cf"] == cf), None)
        ph   = next((r for r in rows if r["series"] == "phase" and r["cf"] == cf), None)
        if topk is None or ph is None:
            continue
        for key, direction in HH:
            a, b = topk.get(key), ph.get(key)
            if a is None or b is None:
                winner = "—"
            else:
                if direction == "lower":
                    winner = "top-k" if a < b else "phase"
                else:
                    winner = "top-k" if a > b else "phase"
            lines.append(f"| {cf:.2f} | {key} | {fmt(a)} | {fmt(b)} | {winner} |")

    # Aux-loss ablation (if present)
    if any(r["series"] == "topk_noaux" for r in rows):
        lines += [
            "",
            "## Top-k aux-loss ablation",
            "",
            "_How much does top-k's load balance depend on the auxiliary loss?_",
            "",
            "| cf | aux=0.01 val_cv | aux=0 val_cv | aux=0.01 val_ce | aux=0 val_ce |",
            "| --- | --- | --- | --- | --- |",
        ]
        for r in [r for r in rows if r["series"] == "topk_noaux"]:
            cf = r["cf"]
            base = next((b for b in rows if b["series"] == "topk" and b["cf"] == cf), None)
            if base is None:
                continue
            lines.append(
                f"| {cf:.2f} | {fmt(base.get('val_cv'))} | {fmt(r.get('val_cv'))} | "
                f"{fmt(base.get('val_ce'))} | {fmt(r.get('val_ce'))} |"
            )

    lines += [
        "",
        "## Plots",
        "",
    ]
    for key, _, title, _ in PLOT_METRICS:
        lines.append(f"![{title}]({plots_dirname}/{key}.png)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sweep_dir", type=str)
    p.add_argument("-o", "--out", type=str, default=None,
                   help="markdown output path (default: <sweep_dir>/sweep.md)")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_path = Path(args.out) if args.out else sweep_dir / "sweep.md"
    plots_dir = out_path.parent / "sweep_plots"

    rows = collect(sweep_dir)
    if not rows:
        print(f"no matching runs under {sweep_dir}")
        return
    print(f"found {len(rows)} runs")

    plot_sweep(rows, plots_dir)
    write_markdown(rows, out_path, sweep_dir, plots_dir.name)
    print(f"wrote {out_path}")
    print(f"plots in {plots_dir}")


if __name__ == "__main__":
    main()
