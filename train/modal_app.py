"""Modal app — runs the MoE comparison on GPUs.

Usage from your laptop:

    modal setup                                   # one-time
    modal volume create pr-runs                   # one-time
    modal run train/modal_app.py::smoke           # quick 50-step sanity check
    modal run train/modal_app.py::full --router topk  --config tiny
    modal run train/modal_app.py::full --router phase --config tiny
    modal volume get pr-runs / ./runs --force     # pull artefacts back

Everything else (image build, GPU provisioning, log streaming) is
handled by Modal.
"""
from __future__ import annotations

import os
import subprocess

import modal

# ── Image ───────────────────────────────────────────────────────────────
#
# CUDA base + Python + Rust toolchain + maturin, then `maturin build` the
# wheel and `pip install` it into the system site-packages. We mount the
# local repo at runtime via `add_local_dir` so iteration is fast — only
# the wheel build runs on Modal's cache layer.

CUDA_IMAGE = "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"
REMOTE_REPO = "/root/phase_router_rs"

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .apt_install("git", "curl", "build-essential", "pkg-config")
    .run_commands(
        # Rust toolchain
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        " | sh -s -- -y --default-toolchain stable --profile minimal",
    )
    .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
    .pip_install(
        "maturin>=1.5,<2.0",
        "torch==2.4.0",
        "transformers==4.44.2",
        "datasets==2.21.0",
        "numpy<2",
        "pyyaml",
        "tqdm",
        "matplotlib",
    )
    # Stage 1: copy the Rust sources into the image as a build-time
    # input (`copy=True` so subsequent build steps can see them) and
    # build + install the wheel ONCE per image rebuild.
    .add_local_dir(
        local_path=".",
        remote_path="/tmp/phase_router_rs_src",
        copy=True,
        ignore=[
            "target/**", ".venv*/**", "runs/**", "reports/**",
            "**/__pycache__/**", "*.pyc",
        ],
    )
    .run_commands(
        # Build + install the Python extension into the system env.
        "cd /tmp/phase_router_rs_src && "
        "maturin build --release --interpreter python3.11 && "
        "pip install target/wheels/phase_router_rs-*.whl",
    )
    # Stage 2: re-attach the working tree at the canonical path. Because
    # this is the LAST build step (no run_commands after it), Modal will
    # re-sync it at container startup rather than rebuilding the image
    # when train/*.py changes — fast iteration.
    .add_local_dir(
        local_path=".",
        remote_path=REMOTE_REPO,
        ignore=[
            "target/**", ".venv*/**", "runs/**", "reports/**",
            "**/__pycache__/**", "*.pyc",
        ],
    )
)

app = modal.App("phase-router-moe", image=image)

# Persistent storage for run artefacts.
volume = modal.Volume.from_name("pr-runs", create_if_missing=True)

# Persistent HF cache so we don't re-tokenise / re-download each run.
hf_cache = modal.Volume.from_name("pr-hf-cache", create_if_missing=True)

VOLUME_MOUNTS = {
    "/root/runs": volume,
    "/root/.cache/huggingface": hf_cache,
}

GPU_DEFAULT = "A10G"


# ── Remote functions ────────────────────────────────────────────────────


@app.function(
    gpu=GPU_DEFAULT,
    timeout=4 * 3600,
    volumes=VOLUME_MOUNTS,
)
def train_one(router: str, config: str = "tiny", max_steps: int | None = None,
              out_subdir: str | None = None,
              capacity_factor: float | None = None,
              aux_loss_alpha: float | None = None,
              seq_len: int | None = None):
    """Run one training job (top-k or phase) and return its run dir."""
    os.chdir(REMOTE_REPO)

    out_subdir = out_subdir or f"{config}/{router}"
    out_dir = f"/root/runs/{out_subdir}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "python", f"train/train_{ 'baseline' if router == 'topk' else 'phase' }.py",
        "--config", f"train/configs/{config}.yaml",
        "--out_dir", out_dir,
    ]
    if max_steps is not None:
        cmd += ["--max_steps", str(max_steps)]
    if capacity_factor is not None:
        cmd += ["--capacity_factor", str(capacity_factor)]
    if aux_loss_alpha is not None:
        cmd += ["--aux_loss_alpha", str(aux_loss_alpha)]
    if seq_len is not None:
        cmd += ["--seq_len", str(seq_len)]

    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd)
    # commit the volume so the next function sees the files
    volume.commit()
    return out_dir


@app.function(
    timeout=1800,
    volumes=VOLUME_MOUNTS,
)
def compare(topk_dir: str, phase_dir: str, out: str):
    """Run `train/compare.py` over two run dirs and write a markdown report."""
    os.chdir(REMOTE_REPO)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [
        "python", "train/compare.py", topk_dir, phase_dir, "-o", out,
    ]
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd)
    volume.commit()
    return out


@app.function(
    timeout=1800,
    volumes=VOLUME_MOUNTS,
)
def compare_by_config(config: str):
    """Convenience: compare runs under /root/runs/<config>/{topk,phase}."""
    os.chdir(REMOTE_REPO)
    topk_dir = f"/root/runs/{config}/topk"
    phase_dir = f"/root/runs/{config}/phase"
    out = f"/root/runs/{config}/comparison.md"
    cmd = ["python", "train/compare.py", topk_dir, phase_dir, "-o", out]
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd)
    volume.commit()
    return out


@app.function(
    timeout=1800,
    volumes=VOLUME_MOUNTS,
)
def compare_sweep(sweep_dir: str):
    """Aggregate all subdirs under <sweep_dir> into one summary table + plot."""
    os.chdir(REMOTE_REPO)
    out = f"{sweep_dir}/sweep.md"
    cmd = ["python", "train/compare_sweep.py", sweep_dir, "-o", out]
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd)
    volume.commit()
    return out


@app.function(
    # No GPU here — this is just the orchestrator. The actual GPU work
    # happens inside train_one.remote(...) calls below, which each get
    # their own GPU container.
    timeout=8 * 3600,
    volumes=VOLUME_MOUNTS,
)
def orchestrate(config: str = "tiny", max_steps: int | None = None):
    """Detach-safe orchestrator: runs topk → phase → compare on Modal.

    This is itself a remote @app.function, so when the local entrypoint
    `both` does `orchestrate.spawn(config)` and exits, the orchestrator
    keeps living on Modal and its blocking .remote() calls below survive
    the local disconnect.
    """
    print(f"[orchestrate] config={config} max_steps={max_steps}")
    print("[orchestrate] === top-k ===")
    train_one.remote("topk", config=config, max_steps=max_steps,
                     out_subdir=f"{config}/topk")
    print("[orchestrate] === phase ===")
    train_one.remote("phase", config=config, max_steps=max_steps,
                     out_subdir=f"{config}/phase")
    print("[orchestrate] === compare ===")
    compare_by_config.remote(config)
    print(f"[orchestrate] done — artefacts in /root/runs/{config}")
    return f"/root/runs/{config}"


@app.function(
    timeout=12 * 3600,
    volumes=VOLUME_MOUNTS,
)
def orchestrate_sweep(config: str = "tiny", max_steps: int | None = None,
                      sweep_name: str = "cf_sweep"):
    """Capacity-factor sweep + aux-loss ablation.

    Runs every (router, cf) ∈ {topk, phase} × {1.0, 1.25, 1.5, 2.0} = 8 jobs,
    plus two top-k aux-loss ablations at cf ∈ {1.0, 1.25} with aux=0.

    Layout on volume:
        /root/runs/<sweep_name>/topk_cf1.00/
        /root/runs/<sweep_name>/topk_cf1.25/
        ...
        /root/runs/<sweep_name>/phase_cf2.00/
        /root/runs/<sweep_name>/topk_cf1.00_noaux/
        /root/runs/<sweep_name>/topk_cf1.25_noaux/

    Each subdir has metrics.jsonl + model.pt. The aggregator
    `compare_sweep` produces one summary table + figure across all of them.

    On A10G with `tiny.yaml` (2000 steps), each run is ~8 min, so the
    full sweep is ~80 min wall-time × 1 A10G = ~$1.50.
    """
    sweep_root = f"/root/runs/{sweep_name}"
    print(f"[sweep] root={sweep_root} config={config} max_steps={max_steps}")

    capacity_factors = [1.0, 1.25, 1.5, 2.0]

    # Main 2 × 4 grid.
    for cf in capacity_factors:
        for router in ("topk", "phase"):
            tag = f"{router}_cf{cf:.2f}"
            print(f"[sweep] === {tag} ===")
            train_one.remote(
                router,
                config=config,
                max_steps=max_steps,
                out_subdir=f"{sweep_name}/{tag}",
                capacity_factor=cf,
            )

    # Top-k aux-loss ablations: how much does balance degrade without
    # the auxiliary loss? Quantifies "phase gets balance for free".
    for cf in (1.0, 1.25):
        tag = f"topk_cf{cf:.2f}_noaux"
        print(f"[sweep] === {tag} ===")
        train_one.remote(
            "topk",
            config=config,
            max_steps=max_steps,
            out_subdir=f"{sweep_name}/{tag}",
            capacity_factor=cf,
            aux_loss_alpha=0.0,
        )

    print("[sweep] === aggregate ===")
    compare_sweep.remote(sweep_root)
    print(f"[sweep] done — artefacts in {sweep_root}")
    return sweep_root



# ── Local entrypoints ───────────────────────────────────────────────────


@app.local_entrypoint()
def smoke(config: str = "tiny", max_steps: int = 50):
    """50-step smoke run for both routers, then compare. ~5 min on A10G."""
    print("=== smoke: top-k ===")
    train_one.remote("topk", config=config, max_steps=max_steps,
                     out_subdir=f"smoke_{config}/topk")
    print("=== smoke: phase ===")
    train_one.remote("phase", config=config, max_steps=max_steps,
                     out_subdir=f"smoke_{config}/phase")
    # quick comparison (uses smoke_<config> subdir)
    print("=== smoke: compare ===")
    compare.remote(
        f"/root/runs/smoke_{config}/topk",
        f"/root/runs/smoke_{config}/phase",
        f"/root/runs/smoke_{config}/comparison.md",
    )
    print(f"done. fetch with:  modal volume get pr-runs /smoke_{config} ./runs/smoke_{config}")


@app.local_entrypoint()
def full(router: str = "topk", config: str = "tiny"):
    """One full training run. Call twice (router=topk then router=phase)."""
    assert router in ("topk", "phase"), router
    train_one.remote(router, config=config)


@app.local_entrypoint()
def both(config: str = "tiny", max_steps: int | None = None):
    """Run both routers sequentially, then compare.

    Recommended invocation (survives local disconnect):

        modal run --detach train/modal_app.py::both --config tiny

    This fires `orchestrate.spawn(config)` and returns. The orchestrator
    lives entirely on Modal and its three blocking `.remote()` calls
    (topk → phase → compare) will complete regardless of your CLI state.
    """
    call = orchestrate.spawn(config, max_steps)
    print(f"orchestrator spawned: call_id={call.object_id}")
    print(f"  watch logs:  modal app logs phase-router-moe")
    print(f"  fetch when done:  modal volume get pr-runs /{config} ./runs/{config} --force")


@app.local_entrypoint()
def both_blocking(config: str = "tiny", max_steps: int | None = None):
    """Same as `both` but blocks the local CLI until everything completes.

    Useful when you DON'T pass --detach and want to stream logs live.
    """
    out = orchestrate.remote(config, max_steps)
    print(f"orchestrator finished -> {out}")
    print(f"  fetch:  modal volume get pr-runs /{config} ./runs/{config} --force")


@app.local_entrypoint()
def sweep(config: str = "tiny", max_steps: int | None = None,
          sweep_name: str = "cf_sweep"):
    """Capacity-factor sweep (4 cf × 2 routers) + 2 aux-loss ablations.

    Recommended invocation:

        modal run --detach train/modal_app.py::sweep --config tiny

    Total wall-time on A10G with default `tiny.yaml` (2000 steps): ~80 min,
    cost: ~$1.50. Writes a single aggregated `sweep.md` at the end.
    """
    call = orchestrate_sweep.spawn(config, max_steps, sweep_name)
    print(f"sweep orchestrator spawned: call_id={call.object_id}")
    print(f"  watch logs:  modal app logs phase-router-moe")
    print(f"  fetch when done:  modal volume get pr-runs /{sweep_name} ./runs/{sweep_name} --force")

