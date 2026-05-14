# Stress sweep plan — 32-expert balance regime

**Date:** 2026-05-14
**Goal:** push n_experts from 8 → 32 to test whether `cf_sweep_v2`'s
verdict (`top-k +5 % CE`, `phase wins on everything else`) still holds
when load balance is harder.
**Sequel to:** `dev/findings_cf_sweep_v2.md`
**Budget target:** ~$3–5 of Modal credit.

## Scope — what changed since the first draft

First-draft `stress.yaml` was sized at `small.yaml` scale (`d_model: 512`,
`n_layers: 8`, `n_experts: 32`, `max_steps: 8000`). Smoke run on A10G
showed **571 M params at fp32** and **>13 s per step** (dominated by
Python-side per-expert MLP launches — 256 expert linears per step,
each individually dispatched). That projects to ~30 h per run × 8 runs
≈ **$200+** of GPU — an order of magnitude over budget.

The bottleneck isn't physics, it's the Python loop's per-expert launch
overhead at high `n_experts`. To get a clean, apples-to-apples answer
on the n_experts knob without paying that overhead 4× over, the
rescoped `stress.yaml` is now `tiny.yaml` with **only `n_experts`
changed**, 8 → 32:

| field        | tiny.yaml | stress.yaml | small.yaml (rejected) |
| ------------ | --------- | ----------- | --------------------- |
| `d_model`    | 256       | 256         | 512                   |
| `n_layers`   | 4         | 4           | 8                     |
| `d_ff`       | 1024      | 1024        | 2048                  |
| `n_experts`  | **8**     | **32**      | 32                    |
| `batch_size` | 8         | 8           | 16                    |
| `seq_len`    | 256       | 256         | 512                   |
| `max_steps`  | 2000      | 2000        | 8000                  |
| params       | ~37 M     | ~87 M       | 571 M                 |
| budget/run   | ~$0.20    | ~$0.30–0.50 | ~$25 (rejected)       |
| total budget | ~$1.50    | **~$3–5**   | ~$200                 |

This means the v2 cf_sweep and this stress sweep share an identical
experimental scaffold; the **only** independent variable across the
two sweeps is `n_experts ∈ {8, 32}`. That's exactly what we want for a
diff.

## Hypothesis grid

| outcome at 32 experts              | what it means for the paper                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| CE gap **closes** (phase ≤ top-k)  | Phase becomes a quality competitor as balance matters — clean win.                                                             |
| CE gap **widens** (phase >> top-k) | Phase is a balance-only primitive, not a CE replacement. Sensible paper framing: "use phase as a constraint, top-k as a head". |
| CE gap **constant** at ~5 %        | Phase's structural cost is independent of E. Paper claim is: "balance for free at every scale, at fixed CE tax".               |

Any of these is publishable; the question is which paragraph to write.

## Expected directional moves

Compared to `cf_sweep_v2` (n_experts: 8):

- **top-k CV** should _rise_ — with 32 experts and `aux_loss_alpha=0.01`
  fixed, the aux loss has 4× more dimensions to balance. Switch papers
  show CV ≈ 0.15–0.30 at 32 experts.
- **top-k drop @ cf=1.0** should rise from 8.5 % toward ~15–25 %.
- **phase CV** should stay ~0.04 — structural and independent of E
  by construction (proven empirically in `cf_sweep_v2`).
- **phase drop** should stay 0 % — bipartite construction never overflows.
- **phase locality (`contig_frac`)** is harder to predict; with 32
  bands instead of 8, adjacent rows hit the _same_ expert less often,
  so contig may _fall_. Worth measuring.
- **CE**: open question. `TinyStories` has weak expert specialisation
  — most signal sits in a few experts and the rest underfit. Top-k can
  chase that pattern; phase forces uniform load and thus may _lose
  more_ at higher E.

## Experimental design

Grid:

| router | cf   | aux_loss_alpha | runs |
| ------ | ---- | -------------- | ---- |
| top-k  | 1.00 | 0.01           | 1    |
| top-k  | 1.25 | 0.01           | 1    |
| top-k  | 1.50 | 0.01           | 1    |
| top-k  | 2.00 | 0.01           | 1    |
| phase  | 1.00 | n/a            | 1    |
| phase  | 1.25 | n/a            | 1    |
| phase  | 1.50 | n/a            | 1    |
| phase  | 2.00 | n/a            | 1    |

**Total: 8 runs.** No noaux ablation — already characterised in v2 and
will produce the same qualitative result (without aux loss, top-k's
CV explodes; phase doesn't care). Skipping it saves ~$0.50.

Config: `train/configs/stress.yaml` (tiny scale, `n_experts: 32`,
2000 steps).

Per-run wall time estimate (A10G, fp32):

- v2 (`tiny.yaml`, 8 experts) ran at ~17 500 tok/s ⇒ ~4 min/run on 2000 steps.
- With 4× more experts, the per-step expert-loop cost grows by ~4×.
  Other costs (attention, embeddings, opt.step) stay constant, so the
  realised step time is ~2–3× v2's, not 4×. Estimated ~10–20 min/run.
- 8 runs × ~15 min ≈ **2 GPU-hours ≈ $2–3**.

If a smoke test shows >30 min/run, drop `max_steps` to 1000 (v2 trends
plateau well before 2000 anyway — see `dev/findings_cf_sweep_v2.md`).

## Launch checklist

1. **Smoke first** (cheap sanity check, single run, 200 steps):

   ```bash
   modal run train/modal_app.py::smoke --config stress --max-steps 200
   ```

   ⚠ Modal converts Python kwargs `foo_bar` → CLI `--foo-bar`. Use
   `--max-steps`, **not** `--max_steps`. (The log line you'll see
   inside the container — `>>> python train/train_phase.py … --max_steps 200`
   — is the _inner_ argparse-based CLI, which keeps underscores. Both
   are correct in their own context.)

   ⚠ The Modal image embeds **`PYTHONUNBUFFERED=1`** (see the
   `image.env(...)` block in `train/modal_app.py`). Without that,
   Python's `print(...)` calls go through a 4 KB block buffer when the
   destination is not a TTY (which Modal log streams aren't), and
   `_loop.py`'s progress lines never appear until the buffer fills or
   the process exits. The hint that you've hit this regression is
   that the very first log line in the container is the transformers
   tokenizer warning (which goes to stderr, unbuffered) and then …
   nothing, for 10+ minutes. If you see that, rebuild the image.

   What the smoke verifies:
   - the image rebuilds with the new config,
   - `n_experts=32` doesn't OOM on A10G (it won't at this scale —
     model is ~87 M params),
   - metrics.jsonl is written correctly for both routers,
   - step time is in the expected ~0.5 s range (4× v2's 0.23 s).

   Should cost <$0.10.

2. **Launch the sweep** (8 runs, detached):

   ```bash
   modal run --detach train/modal_app.py::stress_sweep
   ```

   `--detach` is **mandatory** — the entrypoint uses `.spawn(...)` and
   Modal will tear the app down on local-CLI exit otherwise.

3. **Watch progress:**

   ```bash
   modal app logs phase-router-moe
   ```

4. **Pull artefacts (metrics only, fast):**

   ```bash
   SKIP_MODEL=1 SWEEP=stress_sweep bash scripts/pull_sweep.sh
   ```

5. **Aggregate locally:**

   ```bash
   python train/compare_sweep.py runs/stress_sweep -o reports/stress_sweep.md
   ```

6. **Write findings** to `dev/findings_stress_sweep.md`, diffing
   against `cf_sweep_v2`.

## What to report

For each of the four CE-gap outcomes above, the writeup template is:

> _At 32 experts on `stress.yaml` × 2000 steps, phase routing produced
> {CV} load balance and {drop %} drop rate vs top-k's {CV} / {drop %},
> at a {Δ} CE {gap | parity | inversion} compared to cf_sweep_v2's
> 5 % gap at 8 experts. {Take-home interpretation}._

If outcome is "gap widens": frame as "phase is a balance primitive,
combine with top-k for quality" — pitch a brief follow-up showing
`phase ⊕ top-k` ensemble matches both metrics. (~1 extra GPU-day.)

If outcome is "gap closes": this is the paper's headline. Direct
write-up against Switch / GShard baselines.

If outcome is "gap unchanged": the structural-balance narrative
already in v2 generalises, paper writes itself. Good outcome.

## Fallbacks if the sweep stalls or OOMs

- **OOM on A10G:** unlikely at this scale, but if it happens drop
  `batch_size 8 → 4`, raise `grad_accum 2 → 4`. Token throughput
  unchanged.
- **Per-run runtime > 30 min:** drop `max_steps 2000 → 1000`. We
  already know from v2 that 1000-step trends predict final val_ce
  within 1 %.
- **One run crashes:** Modal will retry the @app.function once. If
  that fails too, re-launch only the missing tag with
  `modal run --detach train/modal_app.py::full --router <r> --config stress`
  and `--capacity_factor <cf>` (need to add the cf override to `full`
  if we want true one-off resumes; the current sweep entrypoint
  re-runs everything, which costs ~$0.50 to skip a failure).

## Historical note: why the first stress.yaml was rejected

Smoke run on A10G with the small-scale draft (commit `<see git log>`):

- Image build: ~2 min (one-off).
- Top-k smoke, fp32:
  - **571,401,728 params** — fits on A10G with room to spare, AdamW
    state ≈ 4.6 GB.
  - Step 1: 144 ms route time, throughput 2 724 tok/s, dt ≈ 12 s
    (includes cudnn warmup + first-pass JIT).
  - Step 50: **did not appear** after 11+ wall-clock minutes — i.e.
    steady-state ≥ 13 s per step. 8000 steps × 13 s ≈ **29 h per run**.
- Diagnosis: 32 experts × 8 layers × 4 micro-batches × ~250 µs per
  per-expert linear launch ≈ 250 ms just on Python/CUDA launch
  overhead, before any actual math. At this expert count and layer
  count, the python `for expert in self.experts: ...` loop in
  `train/moe_model.py` saturates the GPU's launch throughput rather
  than its compute throughput. Phase already won by 4.5× in v2 on the
  routing portion; here it would get crushed by the expert-loop
  bottleneck on **both** sides.
- Decision: kill the small-scale stress, rescope `stress.yaml` to
  `tiny + n_experts=32`. The expert-loop bottleneck still exists at
  tiny scale but it's ~4× smaller (4 layers vs 8) and absolute step
  time is ~0.5 s instead of 13 s, putting an 8-run sweep at $3–5
  rather than $200+. The science (does the gap close at 4× experts?)
  is unchanged.
- Follow-up: a true "small-scale 32-expert" sweep is worth doing
  **after** we replace the python expert loop with a fused / grouped
  matmul kernel (or torch's `torch._grouped_mm` if it lands). That's
  out of scope for this paper but on the roadmap.
