# MoE comparison harness — Phase Router vs Top-k gating

This folder contains a small, self-contained training stack for
running side-by-side comparisons of two MoE routers on a tiny GPT-style
language model.

| Run | Router                                 | Entry point         |
| --- | -------------------------------------- | ------------------- |
| A   | Top-k softmax gating (Switch / GShard) | `train_baseline.py` |
| B   | Phase Router (`phase_router_rs`)       | `train_phase.py`    |

Both entry points share `_loop.py`, so the only thing that differs
between runs is the router.

## Layout

```
train/
├── _loop.py             # shared train loop
├── compare.py           # post-hoc analysis (writes md + plots)
├── configs/
│   ├── tiny_cpu.yaml    # only for local CPU smoke tests (don't compare!)
│   ├── tiny.yaml        # first real comparison run
│   └── small.yaml       # bigger run for A10G / A100
├── data.py              # streaming TinyStories tokens
├── modal_app.py         # Modal GPU launcher
├── moe_model.py         # GPT-style decoder w/ MoE FFN
├── routers.py           # TopKRouter, PhaseRouter
├── train_baseline.py    # entrypoint A
└── train_phase.py       # entrypoint B
```

## Quick start (local CPU smoke)

```bash
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install transformers datasets pyyaml matplotlib numpy tqdm
maturin develop --release   # one-time build of the Rust extension

python train/train_baseline.py --config train/configs/tiny_cpu.yaml \
       --max_steps 3 --device cpu --out_dir runs/smoke_topk
python train/train_phase.py    --config train/configs/tiny_cpu.yaml \
       --max_steps 3 --device cpu --out_dir runs/smoke_phase
python train/compare.py runs/smoke_topk runs/smoke_phase -o reports/smoke.md
```

Expected runtime: ~30s/step on a laptop CPU. The smoke test exists
only to prove the wiring works; it's far too short to learn anything.

## Real comparison on Modal

```bash
pip install modal && modal setup       # one-time
modal volume create pr-runs            # one-time

# Sanity check on a real GPU (~5 min, ~$0.10)
modal run train/modal_app.py::smoke

# Full comparison on tiny.yaml (~2 hr per router on A10G)
modal run --detach train/modal_app.py::both --config tiny

# Watch live
modal app logs phase-router-moe

# When done, pull artefacts back
modal volume get pr-runs /tiny ./runs/tiny --force
python train/compare.py runs/tiny/topk runs/tiny/phase -o reports/tiny.md
```

## What each metric means

- **CE loss**: next-token cross-entropy. Lower = better LM.
- **load_cv**: coefficient of variation of per-expert assignment
  counts. Lower = more balanced load.
- **dropped_frac**: fraction of (token, slot) pairs where the router
  was forced to assign `-1` because of capacity overflow. Lower =
  fewer wasted compute slots.
- **tok/s**: training throughput, excluding step 0 warmup.

## Fairness notes

- Both runs use identical seed, identical batch order, identical
  model dims, identical optimiser.
- Baseline uses Switch's `aux_loss` (alpha = `cfg.aux_loss_alpha`).
- Phase Router ignores the aux loss because the kernel handles balance
  structurally. If you want a strict apples-to-apples, you can either
  zero the aux loss for baseline (`aux_loss_alpha: 0.0`) or add an
  equivalent regulariser to the phase run.
- Phase Router routes on CPU (the kernel is CPU-only and releases the
  GIL). For the tiny config this is negligible (<10ms / layer); for
  much larger N it becomes a real cost — see `dev/modal_training_plan.md`
  §13 for mitigations.
