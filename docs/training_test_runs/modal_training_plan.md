# Modal MoE Training Plan: Phase Router vs Top-k Gating

A step-by-step playbook for renting an Nvidia GPU on
[Modal](https://modal.com), building `phase_router_rs` on the remote
host, plugging it into a small Mixture-of-Experts (MoE) language-model
training loop based on Hugging Face `transformers`, and running two
controlled comparison runs:

| Run | Gating mechanism                                            |
| --- | ----------------------------------------------------------- |
| A   | **Baseline** — softmax top-k gating (Switch / GShard style) |
| B   | **Phase Router** — `phase_router_rs.phase_router_auto`      |

This document is written for someone who has never used Modal before.
Every step is explicit. Read it once end-to-end, then come back to the
sections marked **"do this now"** as you go.

---

## 0. What we are actually comparing

The Rust crate produces, per token, up to `k` expert indices that are
**capacity-aware by construction** — the expected load on expert `j`
is proportional to that expert's encoded capacity. Standard top-k
gating instead picks the `k` experts with highest gate logits, which
is capacity-blind and relies on auxiliary load-balancing losses.

Both gates still use a learned linear layer to produce gate logits.
The logits are used for:

1. **Routing weight** — the scalar multiplier on each expert's output.
2. **For baseline**: also for selecting which experts.
3. **For Phase Router**: only as weights; expert _selection_ comes
   from the Rust kernel, fed bit-packed source demand and target
   capacities derived from the logits.

This way both runs are gradient-compatible (the gate matrix still
receives gradient through the routing weights) and the only thing that
changes between runs is **how `(token → expert)` is decided**.

Comparison metrics:

- Train / val cross-entropy
- Step throughput (tokens / s)
- Dropped-token rate per layer
- Expert load CV (`std / mean`) and `max / mean`
- Total wall time

---

## 1. Prerequisites on your laptop

**Do this now:**

```bash
# (a) make sure your repo is committed and pushed
cd /home/dan/Desktop/Paper/phase_router_rs
git status
git push origin main         # or whatever branch you're on

# (b) tooling on your laptop
python3 -m venv .venv-modal
source .venv-modal/bin/activate
pip install --upgrade pip
pip install modal             # the Modal CLI + SDK
```

**Sign up + authenticate:**

```bash
modal setup
```

That opens a browser window. Log in (GitHub SSO is easiest). The CLI
will write a token to `~/.modal.toml`. You're now ready to launch
remote containers.

Modal gives every new account ~$30 of free credit, which is plenty
for the runs below (each is ~$2-$10 depending on GPU choice).

---

## 2. GPU and budget choice

For our scale (small GPT-2-sized MoE, ~50M active params, 100M-1B
training tokens) any of the following work:

| GPU      | $/hr (Modal) | Suggested for                            |
| -------- | ------------ | ---------------------------------------- |
| T4       | ~$0.59       | smoke tests only                         |
| L4       | ~$0.80       | smoke tests, fp16/bf16 ok                |
| A10G     | ~$1.10       | **recommended** — fits 8 experts × d=512 |
| A100 40G | ~$2.10       | if you want larger model / faster runs   |
| H100     | ~$3.95+      | overkill for this experiment             |

Per run we want:

- ~30 min smoke test on A10G → $0.55
- ~2-3 hr "real" run × 2 (baseline + phase) on A10G → ~$6 total

Set a spend cap in the Modal dashboard (Settings → Usage limits).

---

## 3. Project layout for the Modal training work

We won't pollute the existing repo. Everything Modal-specific lives
under `train/`:

```
phase_router_rs/
├── train/
│   ├── modal_app.py            # Modal entry-point
│   ├── Dockerfile.notes        # what goes into the image (reference)
│   ├── moe_model.py            # tiny GPT-MoE model definition
│   ├── routers.py              # TopKRouter, PhaseRouter (gateway to Rust)
│   ├── train_baseline.py       # run A
│   ├── train_phase.py          # run B
│   ├── data.py                 # streaming dataset (tinystories or wikitext-103)
│   ├── compare.py              # post-hoc analysis on the two run dirs
│   └── configs/
│       ├── tiny.yaml           # 6L 512d 8 experts, ~50M params
│       └── small.yaml          # 12L 768d 8 experts, ~150M params
```

We will create those files in stage 6.

---

## 4. Modal mental model (1 minute)

Modal is "Python functions, but the body runs on a GPU container we
spin up for you." The model is:

```python
import modal

# 1. Define an image (effectively a Dockerfile in Python)
image = (modal.Image.debian_slim()
         .apt_install("git", "curl", "build-essential")
         .pip_install("torch==2.4.0", "transformers", ...)
         .run_commands("curl ... | sh -s -- -y")      # install rustup
        )

# 2. Define an app
app = modal.App("phase-router-moe")

# 3. Define a function with a GPU + image attached
@app.function(image=image, gpu="A10G", timeout=4*3600,
              volumes={"/root/runs": modal.Volume.from_name("pr-runs")})
def train_one(config_path: str, router: str):
    # this body executes on the GPU container
    import subprocess
    subprocess.run(["python", "train/train_baseline.py", "--config", config_path])

# 4. Local entry point — what `modal run modal_app.py` calls
@app.local_entrypoint()
def main():
    train_one.remote("train/configs/tiny.yaml", "topk")
    train_one.remote("train/configs/tiny.yaml", "phase")
```

Three things to remember:

- **`@app.function(...)`** turns a Python function into a remote job.
- **`.remote(...)`** runs it on Modal; **`.local(...)`** runs locally.
- **Volumes** are how you persist files (checkpoints, logs) across
  runs — the container itself is ephemeral.

Reference: <https://modal.com/docs/guide>

---

## 5. The Modal image we need

The image must contain:

- CUDA 12.x base
- Python ≥ 3.10
- Rust toolchain (for `cargo build` of `phase_router_rs`)
- `maturin` (to build the PyO3 wheel)
- `torch`, `transformers`, `accelerate`, `datasets`, `wandb` (optional)
- A clone of this repo

We can either (a) `git clone` inside the image build, or (b) mount the
local repo at runtime. Option (b) is faster to iterate on but uploads
your working tree each launch. Option (a) is reproducible. **We'll do
(a)** and bake a pinned commit into the image, plus support (b) for
hot-iteration via `modal.Mount`.

```python
# train/modal_app.py (skeleton — full version in stage 6)
import modal

REPO_URL = "https://github.com/TSltd/phase_router_rs.git"
COMMIT   = "HEAD"   # pin to a hash once stable

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
                              add_python="3.11")
    .apt_install("git", "curl", "build-essential", "pkg-config")
    .run_commands(
        # Rust toolchain
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
        "  | sh -s -- -y --default-toolchain stable",
        'echo \'source $HOME/.cargo/env\' >> /root/.bashrc',
    )
    .env({"PATH": "/root/.cargo/bin:${PATH}"})
    .pip_install(
        "maturin>=1.5",
        "torch==2.4.0",
        "transformers==4.44.2",
        "accelerate==0.34.2",
        "datasets==2.21.0",
        "numpy",
        "pyyaml",
        "wandb",        # optional — comment out if you don't want it
        "tqdm",
    )
    .run_commands(
        f"git clone {REPO_URL} /root/phase_router_rs",
        f"cd /root/phase_router_rs && git checkout {COMMIT}",
        # Build the Rust extension into the global Python env
        "cd /root/phase_router_rs && maturin build --release "
        "  && pip install target/wheels/phase_router_rs-*.whl",
    )
)
```

A few notes:

- `add_python="3.11"` from `modal.Image.from_registry` gives us a
  Python interpreter inside the CUDA image.
- We install the wheel into the system site-packages so any process
  in the container can `import phase_router_rs`.
- Pin `torch==2.4.0` (or whatever current stable matches the CUDA 12.4
  base). `transformers` 4.44+ has Mixtral / OLMoE blocks we'll borrow.

---

## 6. The training stack — what each file does

### 6.1 `train/moe_model.py`

A tiny GPT-style decoder where the FFN of every other block is
replaced with an **MoE block**. The block is identical between
baseline and phase runs except for the router. Shape:

```python
class MoEBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, k, router):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                          nn.Linear(d_ff, d_model))
            for _ in range(n_experts)
        ])
        self.router = router          # injected: TopKRouter or PhaseRouter
        self.k = k

    def forward(self, x):                     # x: (B, T, d_model)
        B, T, D = x.shape
        flat = x.reshape(B*T, D)
        logits = self.gate(flat)              # (N, n_experts)
        idx, weights, aux = self.router(logits, self.k)
        # idx: (N, k) int64;  weights: (N, k) float
        # aux: dict with load stats, drop counts
        out = torch.zeros_like(flat)
        for e_id, expert in enumerate(self.experts):
            mask = (idx == e_id).any(dim=-1)
            if not mask.any():
                continue
            tok = flat[mask]
            w_e = (weights * (idx == e_id)).sum(dim=-1, keepdim=True)[mask]
            out[mask] += w_e * expert(tok)
        return out.reshape(B, T, D), aux
```

The block returns `aux` so we can log dropped-token and load-CV
metrics per layer per step.

### 6.2 `train/routers.py`

Two router modules with the same signature.

```python
class TopKRouter(nn.Module):
    def forward(self, logits, k):
        # logits: (N, n_experts)
        gate = F.softmax(logits, dim=-1)
        weights, idx = gate.topk(k, dim=-1)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-9)
        # capacity enforcement (Switch-style)
        n_experts = logits.size(-1)
        capacity = int(self.capacity_factor * logits.size(0) * k / n_experts)
        idx, weights, dropped = enforce_capacity(idx, weights, n_experts, capacity)
        return idx, weights, {"dropped": dropped, "load_cv": load_cv(idx, n_experts)}


class PhaseRouter(nn.Module):
    """Capacity-aware selection via phase_router_rs.

    Pipeline per forward call:
      1. project logits → per-expert relative capacity (softmax + EMA)
      2. derive per-token demand bit-vector (top-K_demand experts of each token)
      3. build bit-packed (s_bits, t_bits)
      4. call phase_router_rs.phase_router_auto → (N, k) idx
      5. read routing weights = softmax(logits) gathered at idx, normalised
    """
    def __init__(self, n_experts, capacity_factor=1.25, demand_top=8,
                 seed=0, ema=0.99):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.demand_top = demand_top
        self.seed = seed
        self.register_buffer("cap_ema",
                             torch.ones(n_experts) / n_experts)
        self.ema = ema

    @torch.no_grad()
    def _build_bits(self, logits, n_pad):
        # ... see stage 7 for details
        ...

    def forward(self, logits, k):
        # routing weights still come from the differentiable softmax
        gate = F.softmax(logits, dim=-1)
        # update EMA of "where would top-k go" — used as expert capacity
        with torch.no_grad():
            avg_demand = gate.mean(dim=0)
            self.cap_ema.mul_(self.ema).add_(avg_demand, alpha=1 - self.ema)
        idx = self._call_kernel(gate, k)             # (N, k) int64 on CPU
        idx = idx.to(logits.device)
        weights = gate.gather(-1, idx.clamp_min(0))  # (N, k)
        weights = weights * (idx >= 0)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-9)
        return idx, weights, {"dropped": (idx == -1).float().mean().item(),
                              "load_cv": load_cv(idx, self.n_experts)}
```

The `_build_bits` helper is the crux of the integration — see
section 7.

### 6.3 `train/data.py`

Stream tokens from a small dataset so we don't spend Modal time on
data download. Recommended: **TinyStories** (~3 GB raw) or
**wikitext-103-raw-v1** (~500 MB raw). Both stream cleanly via
`datasets.load_dataset(..., streaming=True)`.

Tokenise on the fly with `transformers.GPT2TokenizerFast`.

### 6.4 `train/configs/tiny.yaml`

```yaml
d_model: 512
n_layers: 6
n_heads: 8
d_ff: 2048
n_experts: 8
k: 2
seq_len: 512
batch_size: 16
grad_accum: 4
lr: 3.0e-4
max_steps: 5000
warmup: 200
capacity_factor: 1.25
log_every: 50
val_every: 500
dataset: roneneldan/TinyStories # or wikitext-103-raw-v1
```

### 6.5 `train/train_baseline.py`, `train/train_phase.py`

Both files import the same `train_loop(cfg, router_cls)` from a
shared `train/_loop.py`. Only the last line differs:

```python
# train_baseline.py
from _loop import run; run("topk")

# train_phase.py
from _loop import run; run("phase")
```

That guarantees apples-to-apples: same optimiser, same data order,
same seed, same model dims.

### 6.6 `train/compare.py`

After both runs finish, this script reads `runs/topk/*.jsonl` and
`runs/phase/*.jsonl`, produces a comparison CSV and three plots:

- val loss vs steps
- expert-load CV vs steps (lower = better balance)
- dropped-token rate vs steps

---

## 7. The Phase-Router-↔-PyTorch glue (the only non-trivial bit)

The Rust kernel wants:

- `s_bits` — flat `np.uint64` array, `n * nb_words`, the source-demand
  bit matrix. Row `i` has `popcount` proportional to token `i`'s
  _demand_ (we'll use a constant since every token has equal
  demand for `k` experts — see below).
- `t_bits` — flat `np.uint64` array, capacity-proportional bit pattern
  per expert.
- `n` — must equal **both** rows and columns of the bit matrices.

Here `n = number_of_tokens_in_microbatch`, and we have
`n_experts << n`. We embed `n_experts` columns into a length-`n`
phase ring by tiling, which the kernel handles naturally. Concretely:

```python
def build_phase_inputs(logits, n_experts, cap_ema,
                       base_density=0.3, capacity_factor=1.25):
    """Construct bit-packed (s_bits, t_bits) for one micro-batch."""
    N = logits.shape[0]                    # number of tokens
    nb_words = (N + 63) // 64

    # --- source side ---------------------------------------------------
    # every token has the same demand → uniform ones-per-row
    s_ones = max(1, int(round(base_density * N)))
    s_bits = np.zeros(N * nb_words, dtype=np.uint64)
    for b in range(s_ones):
        s_bits[b // 64::nb_words] |= np.uint64(1) << np.uint64(b % 64)
    # (vectorised version omitted for clarity)

    # --- target side ---------------------------------------------------
    # capacities are EMA-smoothed expert demand, scaled to N.
    rel = cap_ema.detach().cpu().numpy()
    rel = rel / rel.mean()
    # Tile experts across N "columns" so each expert owns a band.
    width = N // n_experts
    t_ones = np.zeros(N, dtype=np.int64)
    for e in range(n_experts):
        ones_e = max(1, int(round(rel[e] * base_density * width)))
        # assign all rows in expert e's band the same ones-count
        t_ones[e*width:(e+1)*width] = ones_e
    t_bits = build_bits_vectorised(t_ones, N, nb_words)

    return s_bits, t_bits
```

Then in `PhaseRouter._call_kernel`:

```python
import phase_router_rs
routes = phase_router_rs.phase_router_auto(
    s_bits, t_bits, N, k=k, seed=self.seed
)
# routes: (N, k) int32 — column indices in [0, N)
# map column index → expert id
expert_id = (routes // width).clip(max=n_experts-1)
expert_id[routes < 0] = -1
return torch.from_numpy(expert_id.astype(np.int64))
```

That's the entire bridge: build two `uint64` arrays once per layer
per step, call the kernel (releases the GIL → rayon parallel), divide
to recover expert ids.

**Performance budget:** For `N = batch * seq_len = 16 * 512 = 8192`
and 6 layers, that's ~10ms/step on A10G CPU for kernel work — small
compared to a forward/backward pass (~80ms).

---

## 8. Concrete execution plan

### Stage A — smoke test locally (30 min)

Goal: make sure the integration works _on CPU_ before paying for GPU.

**Do this now:**

```bash
cd /home/dan/Desktop/Paper/phase_router_rs
source .venv-modal/bin/activate
pip install maturin torch transformers datasets numpy pyyaml tqdm
maturin develop --release           # builds the wheel into your venv
```

Create `train/` (I'll generate the files for you on request). Then:

```bash
python train/train_phase.py --config train/configs/tiny.yaml \
       --max_steps 20 --device cpu --no_wandb
```

If that completes 20 steps without error, you're ready for Modal.

### Stage B — first Modal launch (5 min)

```bash
cd /home/dan/Desktop/Paper/phase_router_rs
modal run train/modal_app.py::smoke
```

This calls a `smoke` entrypoint we'll define that runs 50 steps on
an A10G with both routers. Expected wall time: ~5 min, cost: ~$0.10.

You should see logs streaming live in your terminal. The container
disappears when done; checkpoints/logs are kept in the named volume.

### Stage C — full comparison run (~2 hr per router on A10G)

```bash
modal run train/modal_app.py::full --config tiny --router topk
modal run train/modal_app.py::full --config tiny --router phase
```

Or, fire both off concurrently in the background:

```bash
modal run --detach train/modal_app.py::full --config tiny --router topk
modal run --detach train/modal_app.py::full --config tiny --router phase
```

`--detach` returns immediately. Track progress at
<https://modal.com/apps> or via `modal app logs phase-router-moe`.

### Stage D — pull results locally

The container wrote everything to a Modal Volume. Mount it locally:

```bash
modal volume get pr-runs / ./runs --force
python train/compare.py runs/topk runs/phase -o reports/tiny.md
```

`reports/tiny.md` will summarise:

- final val loss (lower = better)
- mean load CV per layer (lower = better)
- mean dropped-token rate (lower = better)
- step throughput (tokens / s)
- $ cost from Modal usage

---

## 9. Sanity checks before declaring victory

Before claiming Phase Router wins or loses, verify:

- [ ] **Same seed, same data order** in both runs. (Print first 5
      token ids from step 0 to confirm.)
- [ ] **Same gate weights at init**. (Save `model.state_dict()` after
      construction and `torch.equal`.)
- [ ] **Both runs use the same capacity factor.** Phase Router
      effectively has _built-in_ capacity awareness, so for a fair
      fight we let the baseline use **the same** `capacity_factor`.
- [ ] **Load-balancing auxiliary loss is reported separately.** For
      baseline we add the standard `aux_loss` term (alpha=0.01); for
      Phase Router we do _not_ add it (the kernel handles balance
      structurally). Make sure both reach reasonable balance — if
      Phase Router's load CV is _much_ worse than baseline's with
      aux_loss, the EMA `cap_ema` may be too slow.
- [ ] **Throughput**: report tokens/sec excluding compile/warmup.
- [ ] **Cost**: log Modal billing per run.

If any of these fail, fix before scaling up to `small.yaml`.

---

## 10. What scaling up looks like

Once the tiny config is clean, the same code runs on `small.yaml`
(12 layers, 768 d_model, 32 experts, 50k steps) on an A100 40G or
two A10Gs. Expected cost: ~$15-$30 per router.

For _much_ larger experiments (>1B params), the integration we're
building is _not_ the right tool — you'd want to swap in `megablocks`
or `vllm`'s MoE kernel and replace just the _assignment_ step with
the Rust kernel. That's a follow-up not covered here.

---

## 11. Files I will create on request

Tell me to proceed and I will create:

1. `train/modal_app.py` — Modal image + functions (~120 LOC)
2. `train/moe_model.py` — tiny GPT-MoE (~150 LOC)
3. `train/routers.py` — `TopKRouter`, `PhaseRouter` (~180 LOC)
4. `train/_loop.py` — shared train loop (~200 LOC)
5. `train/train_baseline.py` — 3 LOC entrypoint
6. `train/train_phase.py` — 3 LOC entrypoint
7. `train/data.py` — streaming dataset (~80 LOC)
8. `train/compare.py` — post-hoc analysis (~150 LOC)
9. `train/configs/tiny.yaml` — config (~25 lines)
10. `train/configs/small.yaml` — config (~25 lines)

We'll go file-by-file so you can review each before moving on.

---

## 12. Quick reference — Modal commands you'll use

| What                      | Command                                       |
| ------------------------- | --------------------------------------------- |
| log in once               | `modal setup`                                 |
| create persistent volume  | `modal volume create pr-runs`                 |
| run an entrypoint         | `modal run train/modal_app.py::main`          |
| run detached (background) | `modal run --detach train/modal_app.py::main` |
| stream live logs          | `modal app logs phase-router-moe`             |
| list running containers   | `modal container list`                        |
| copy volume → local       | `modal volume get pr-runs / ./runs`           |
| copy local → volume       | `modal volume put pr-runs ./local /remote`    |
| stop everything           | `modal app stop phase-router-moe`             |
| see GPU + $ usage         | dashboard at <https://modal.com/apps>         |

---

## 13. Risk register

| Risk                                     | Mitigation                                                                                |
| ---------------------------------------- | ----------------------------------------------------------------------------------------- |
| Modal image build flakes on Rust install | pre-build wheel locally and copy in via `add_local_file`                                  |
| CUDA / torch version mismatch            | pin `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` + `torch==2.4.0`                         |
| Kernel call dominates step time          | move `build_bits` to a Rust helper, or call once per N steps                              |
| Routing is non-differentiable            | gradient flows via softmax weights — selection is detached on purpose                     |
| EMA capacity collapses to one expert     | clamp `cap_ema` to `[0.5 / n_experts, 4 / n_experts]`                                     |
| Phase Router worse than baseline         | **that is a valid result** — log it, then ablate (EMA speed, demand_top, capacity_factor) |

---

## 14. Next action

When you're ready, say **"proceed with stage A"** and I will:

1. Create all the files in section 11.
2. Verify it runs locally for 20 steps on CPU.
3. Walk you through `modal setup` and the first `modal run` together.

Estimated time from "go" to first GPU train step: **45 minutes**.

---

## 15. Reviewer feedback — addressed vs deferred

After the first Modal `tiny` run (see `dev/findings_modal_tiny.md`) we
received a sharp third-party review. This section records which points
have been **addressed for the next run** and which are **deferred**.

### 15.1 Addressed before re-running

| Point                                                       | What changed                                                                                                                                                                                                                                        | Where                                      |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| _"You don't know the breakdown of step time."_              | Added `route_time_ms` per-forward (summed across MoE blocks) and `step_time_ms` per training step. `compare.py` derives **routing overhead %** = `route_time_ms / step_time_ms`.                                                                    | `train/routers.py`, `train/_loop.py`       |
| _"CE alone is a weak win — show locality."_                 | Added two structural metrics: **`contig_frac`** (fraction of adjacent token rows sharing ≥1 chosen expert — proxy for MegaBlocks-style block-sparse dispatch efficiency) and **`perm_entropy`** (normalised entropy of chosen-expert distribution). | `train/routers.py`, `train/moe_model.py`   |
| _"Phase router's CV explosion makes the whole run void."_   | Identified as EMA-capacity policy bug; added `capacity_mode = "uniform"` (Switch-style hard cap per expert, independent of gate). All configs default to `uniform`.                                                                                 | `train/routers.py`, `train/configs/*.yaml` |
| _"Reviewer can't reproduce — no per-step instrumentation."_ | Train rows in `metrics.jsonl` now include `contig_frac`, `perm_entropy`, `route_time_ms`, `step_time_ms` alongside `ce`, `load_cv`, `drop`, `tok_per_s`. Val rows include `val_contig`, `val_pe`, `val_rt_ms`.                                      | `train/_loop.py`, `train/compare.py`       |

### 15.2 Deferred (after the next run, gated on results)

| Point                                                                                                                 | Why deferred                                                                                                                                                                                                                            | Trigger to revisit                                                                                                                            |
| --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| MegaBlocks-style permute / group dispatch instead of per-slot mask loop                                               | The current `MoEFFN.forward` uses a `(k × n_experts)` Python mask loop. Replacing it with a single permute → batched GEMM → unpermute pass is ~1-2 days of work and would obscure the router comparison (both routers benefit equally). | If `routing overhead %` ends up < 5% but step throughput is still embarrassingly low compared to MegaBlocks-style dispatch published numbers. |
| Routing-only latency micro-benchmark (kernel call alone)                                                              | Already partially covered by `route_time_ms`. A dedicated micro-benchmark sweeping `N`, `n_experts`, `k` adds noise to the training story and lives more naturally in `examples/moe_bench_rs.rs`.                                       | If phase wins on locality, we add the kernel-only sweep to back up the cost claim.                                                            |
| Capacity-factor / `n_experts` sweep                                                                                   | A 2D grid over (cf ∈ {1.0, 1.25, 1.5, 2.0}) × (E ∈ {4, 8, 16, 32}) is 16 runs × 2 routers ≈ 32 Modal jobs ≈ ~$40-$80. Worth doing once one config is clean.                                                                             | After this run confirms the qualitative direction (phase better on locality / cost).                                                          |
| Paper framing rewrite ("capacity-aware routing for block-sparse dispatch", not "MoE balancer that beats top-k on CE") | Premature — we don't have the locality numbers yet. After the next run we'll know whether `contig_frac` actually differs meaningfully between phase and top-k; that empirical result drives the framing, not the other way around.      | Once we have at least one set of locality numbers from the `tiny` run.                                                                        |
| Bigger model / longer training (`small.yaml`, 50k steps, A100)                                                        | We don't yet have a clean, reproducible win at the tiny scale. Spending real money before that is wasted.                                                                                                                               | After `tiny` + sweep confirm a direction.                                                                                                     |
| Differentiable surrogate for selection (to retain gradient through `idx`)                                             | Reviewer's most ambitious suggestion. Significant research effort. The current "weights through softmax + detached selection" pattern is standard (Switch, GShard, Expert Choice all do the same).                                      | Only if everything else lands and CE gap to top-k is still > 1-2%.                                                                            |

### 15.3 Re-run command

```bash
modal run --detach train/modal_app.py::both --config tiny
```

Wait for the run to complete, then:

```bash
modal volume get pr-runs / ./runs --force
python train/compare.py runs/tiny/topk runs/tiny/phase -o reports/tiny_v2.md
```

The new `reports/tiny_v2.md` will have the extra three rows
(`contig_frac`, `perm_entropy`, `route_time_ms`) plus the
`routing overhead %` derived row and three additional plots.
