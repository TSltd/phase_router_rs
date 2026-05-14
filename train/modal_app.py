"""Modal app — runs the MoE comparison on GPUs.

Usage from your laptop:

    modal setup                                                 # one-time
    modal volume create pr-runs                                 # one-time
    modal run          train/modal_app.py::smoke                # quick 50-step sanity check (foreground)
    modal run --detach train/modal_app.py::full  --router topk  --config tiny
    modal run --detach train/modal_app.py::full  --router phase --config tiny
    modal run --detach train/modal_app.py::both  --config tiny  # both routers + compare
    modal run --detach train/modal_app.py::sweep --config tiny  # cf sweep + aux ablation
    modal volume get   pr-runs / ./runs --force                 # pull artefacts back

⚠  **`--detach` is REQUIRED for every entrypoint that uses `.spawn(...)`
    internally** (`both`, `sweep`). Without it, the local CLI exits the
    moment the entrypoint returns and Modal tears down the app along
    with the spawned orchestrator — `https://modal.com/apps` will show
    "Live Apps: 0" within seconds and no work will actually run.
    `smoke` and `full` use `.remote(...)` (blocking) and run fine
    without `--detach`, though `--detach` is still recommended for
    `full` because each run is ~2 hr on A10G.

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
    .env({
        "PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin",
        # Without this, Python's stdout is line-buffered only when
        # attached to a TTY. Modal log streams are NOT a TTY, so
        # `print(...)` calls from `_loop.py` get held in a 4 KB buffer
        # and never reach the log viewer until the buffer fills or the
        # process exits. The `transformers` library issues
        # `warnings.warn(...)` which goes to stderr (unbuffered) and
        # appears immediately, giving the false impression that the
        # script has hung right after tokeniser init.
        #
        # PYTHONUNBUFFERED=1 forces stdout/stderr to be unbuffered
        # everywhere — `print(...)` lines now appear in real time.
        "PYTHONUNBUFFERED": "1",
    })
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


@app.function(
    timeout=24 * 3600,
    volumes=VOLUME_MOUNTS,
)
def orchestrate_stress_sweep(config: str = "stress",
                             max_steps: int | None = None,
                             sweep_name: str = "stress_sweep"):
    """32-expert stress sweep — 4 cf × 2 routers = 8 runs, no aux ablation.

    The aux-loss ablation was already characterised in cf_sweep_v2
    (see `dev/findings_cf_sweep_v2.md`): without aux loss, top-k's CV
    explodes from ~0.10 to ~1.0 at 8 experts. We don't need to re-prove
    that at 32 experts; what we DO need is the head-to-head behaviour
    when balance actually matters.

    Layout on volume:
        /root/runs/<sweep_name>/topk_cf1.00/
        /root/runs/<sweep_name>/topk_cf1.25/
        ...
        /root/runs/<sweep_name>/phase_cf2.00/

    Budget (rough): with `stress.yaml` defaults (8000 steps, batch 16,
    seq 512, d_model 512, 8 layers, 32 experts), each run is ~50–80 min
    on A10G ⇒ ~8 × ~60 min = ~8 GPU-hours = **~$8–10**.
    """
    sweep_root = f"/root/runs/{sweep_name}"
    print(f"[stress] root={sweep_root} config={config} max_steps={max_steps}")

    capacity_factors = [1.0, 1.25, 1.5, 2.0]
    for cf in capacity_factors:
        for router in ("topk", "phase"):
            tag = f"{router}_cf{cf:.2f}"
            print(f"[stress] === {tag} ===")
            train_one.remote(
                router,
                config=config,
                max_steps=max_steps,
                out_subdir=f"{sweep_name}/{tag}",
                capacity_factor=cf,
            )

    print("[stress] === aggregate ===")
    compare_sweep.remote(sweep_root)
    print(f"[stress] done — artefacts in {sweep_root}")
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

    ⚠  **`--detach` is REQUIRED.** This entrypoint uses `.spawn(...)`,
        which only survives if Modal is told to keep the app alive
        after the local CLI exits. Without `--detach`:
          • the orchestrator is killed within seconds of launch,
          • `https://modal.com/apps` shows "Live Apps: 0",
          • no GPU work runs and no artefacts are written.

    Correct invocation:

        modal run --detach train/modal_app.py::both --config tiny

    This fires `orchestrate.spawn(config)` and returns. The orchestrator
    lives entirely on Modal and its three blocking `.remote()` calls
    (topk → phase → compare) will complete regardless of your CLI state.
    Use `both_blocking` instead if you want to stream logs live without
    `--detach`.
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

    ⚠  **`--detach` is REQUIRED.** Same caveat as `both`: this entrypoint
        uses `.spawn(...)` and Modal will tear the app (and the spawned
        orchestrator) down the moment the local CLI exits unless
        `--detach` is set. If you launch without `--detach` you'll see
        "Live Apps: 0" on https://modal.com/apps within seconds and no
        GPU work will run.

    Correct invocation:

        modal run --detach train/modal_app.py::sweep --config tiny

    Total wall-time on A10G with default `tiny.yaml` (2000 steps): ~80 min,
    cost: ~$1.50. Writes a single aggregated `sweep.md` at the end.
    """

    call = orchestrate_sweep.spawn(config, max_steps, sweep_name)
    print(f"sweep orchestrator spawned: call_id={call.object_id}")
    print(f"  watch logs:  modal app logs phase-router-moe")
    print(f"  fetch when done:  modal volume get pr-runs /{sweep_name} ./runs/{sweep_name} --force")


@app.local_entrypoint()
def stress_sweep(config: str = "stress", max_steps: int | None = None,
                 sweep_name: str = "stress_sweep"):
    """32-expert stress sweep (4 cf × 2 routers, no noaux ablation).

    ⚠  **`--detach` is REQUIRED.** Same caveat as `both` / `sweep`: this
        entrypoint uses `.spawn(...)` and Modal will tear the app down
        the moment the local CLI exits unless `--detach` is set. If you
        launch without `--detach` you'll see "Live Apps: 0" on
        https://modal.com/apps within seconds and no GPU work will run.

    Correct invocation:

        modal run --detach train/modal_app.py::stress_sweep

    Companion to `cf_sweep_v2` — answers "does the v2 conclusion
    (top-k +5 % CE, phase wins everything else) hold at 32 experts?".
    See `dev/stress_sweep_plan.md` for the experimental rationale.

    Default budget: ~8 GPU-hours on A10G ≈ **$8–10** (8 runs × ~60 min
    each on `stress.yaml` with its 8000-step default).

    After it finishes:

        SKIP_MODEL=1 SWEEP=stress_sweep bash scripts/pull_sweep.sh
        python train/compare_sweep.py runs/stress_sweep -o reports/stress_sweep.md
    """
    call = orchestrate_stress_sweep.spawn(config, max_steps, sweep_name)
    print(f"stress sweep orchestrator spawned: call_id={call.object_id}")
    print(f"  watch logs:  modal app logs phase-router-moe")
    print(f"  fetch when done (metrics only, fast):")
    print(f"    SKIP_MODEL=1 SWEEP={sweep_name} bash scripts/pull_sweep.sh")


