# Findings — Modal `tiny` run (TinyStories, 2000 steps, A10G)

## Run identity

|                         | top-k baseline                       | phase router                                        |
| ----------------------- | ------------------------------------ | --------------------------------------------------- |
| Config                  | `train/configs/tiny.yaml`            | same                                                |
| Steps                   | 2000                                 | 2000                                                |
| Routing                 | Switch top-k + aux loss (α from cfg) | `phase_router_rs.phase_router_auto` w/ EMA capacity |
| Hardware                | Modal A10G                           | Modal A10G                                          |
| Wall time               | 445.9 s                              | 563.0 s                                             |
| End-of-train throughput | 18,371 tok/s                         | 14,551 tok/s                                        |

(Despite the noisy "NVIDIA Driver was not detected" line from the base image's
entrypoint, the throughput is firmly GPU-grade — that warning is a known
false-alarm from `nvidia/cuda:*-devel` images on Modal.)

## Final validation metrics

| metric          | top-k  | phase      | winner |
| --------------- | ------ | ---------- | ------ |
| val CE          | 4.9146 | 5.1240     | top-k  |
| val load CV     | 0.1483 | **0.7152** | top-k  |
| val drop frac   | 0.0023 | **0.0010** | phase  |
| end train tok/s | 18,371 | 14,551     | top-k  |
| wall time (s)   | 445.9  | 563.0      | top-k  |

Plots: `runs/tiny/tiny/plots/{train_ce,load_cv,dropped}.png`.

## Interpretation

1. **Both routers train.** CE drops from ~175 (sum-not-mean reporting at init)
   to ~5 nats. Phase-router pays a small accuracy cost in this config
   (Δ val CE ≈ 0.21 nats).

2. **Phase drops fewer tokens (0.10 % vs 0.23 %).** This matches the
   theoretical claim: bipartite-incidence routing guarantees that _no_
   expert exceeds its capacity, so the only drops come from the
   first-`k`-unique post-processing in `routers.PhaseRouter._select`
   (when the kernel returns fewer than `k` distinct experts for a token).

3. **Phase has _worse_ load balance (CV 0.72 vs 0.15).** This is the
   non-trivial finding. Cause: our integration uses
   `EMA(gate softmax) × base_density × width` as the per-expert capacity
   when building `t_bits`. High-gate experts therefore get _more_
   capacity → _more_ tokens → higher CV. The kernel is honouring the
   capacities it's given; the _policy_ feeding the kernel isn't doing the
   balancing.

   Compare: top-k Switch uses an explicit auxiliary load-balance loss
   (`aux_loss_alpha`) that pushes the gate towards uniform routing. We
   gave phase no such pressure.

## What this means for the paper

This is not a refutation of the kernel — it's a confirmation that the
**policy layer** (how `t_bits` is set per step) matters as much as the
kernel itself. Three follow-ups, in order of cheapness:

1. **Uniform-capacity phase router**: set every expert's capacity to
   `cf × N / E` (Switch-style hard cap), independent of gate. Predicts
   CV ≪ 0.15 with the same drop guarantee. This is the cleanest
   apples-to-apples comparison to Switch and is probably what the paper
   intended.

2. **Balance-corrected EMA**: keep gate-driven capacities but normalise
   so that `sum(cap) = N · k` exactly, and clip the per-expert capacity
   above so dominant experts can't hog more than e.g. `2 × mean`.

3. **Aux-loss for phase**: add the same Switch aux loss on the gate
   _and_ keep phase's deterministic routing on top. Decouples
   "what the gate wants" from "where tokens actually go".

I'd run (1) next — single-line change to `train/routers.py`, ~10 min of
GPU.

## Modal infrastructure notes (for future runs)

- The orchestrator refactor (`@app.function orchestrate` +
  `local_entrypoint` calling `.spawn`) is now in `train/modal_app.py`.
  Run with `modal run --detach train/modal_app.py::both --config tiny`
  and the orchestrator lives on Modal — local CLI disconnect is now
  safe.
- `train/_loop.py` now logs `cuda_available`, `cuda_device`, `torch`
  version on startup so future runs make GPU presence unambiguous.
- `modal volume get pr-runs /tiny ./runs/tiny --force` puts artefacts
  at `./runs/tiny/tiny/…` (double-nested). Either fetch with
  `modal volume get pr-runs /tiny/tiny ./runs/tiny --force` or accept
  the nesting and update paths in any post-processing scripts.
