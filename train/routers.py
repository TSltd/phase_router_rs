"""Router modules: TopKRouter (baseline) and PhaseRouter (Rust kernel).

Both routers share the signature

    idx, weights, aux = router(logits, k)

where
    logits:  (N, n_experts) — un-normalised gate logits
    idx:     (N, k) int64   — expert id per slot, or -1 if dropped
    weights: (N, k) float   — routing weight per slot (sums to ≤1 per token)
    aux:     dict with logged metrics + optional aux_loss tensor

Differences:
    * TopKRouter: chooses experts by argmax of softmax(logits).
      Adds a Switch-style load-balancing aux loss.
    * PhaseRouter: chooses experts via `phase_router_rs.phase_router_auto`
      using EMA-smoothed expert capacities. Selection is non-diff but
      gradients flow through the gate via `weights = softmax(logits)[..., idx]`.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── shared helpers ─────────────────────────────────────────────────────


def load_cv(idx: torch.Tensor, n_experts: int) -> float:
    """Coefficient of variation of per-expert assignment counts."""
    valid = idx[idx >= 0]
    if valid.numel() == 0:
        return 0.0
    counts = torch.bincount(valid.flatten().long(), minlength=n_experts).float()
    mean = counts.mean()
    if mean.item() == 0.0:
        return 0.0
    return (counts.std(unbiased=False) / mean).item()


def contig_frac(idx: torch.Tensor) -> float:
    """Fraction of adjacent token rows that share at least one expert.

    Proxy for cache locality of the downstream dispatch: contiguous
    same-expert tokens map to one batched GEMM block in MegaBlocks-style
    permute-group-execute dispatchers.

    `idx` is (N, k) int64 with -1 for dropped slots.
    """
    if idx.numel() == 0 or idx.size(0) < 2:
        return 0.0
    a = idx[:-1]    # (N-1, k)
    b = idx[1:]     # (N-1, k)
    # For each adjacent pair, do any (ai, bj) match and both ≥ 0?
    a_exp = a.unsqueeze(-1)        # (N-1, k, 1)
    b_exp = b.unsqueeze(-2)        # (N-1, 1, k)
    matches = (a_exp == b_exp) & (a_exp >= 0)
    share = matches.any(dim=(-1, -2)).float()
    return share.mean().item()


def perm_entropy(idx: torch.Tensor, n_experts: int) -> float:
    """Shannon entropy (nats) of the chosen-expert distribution.

    Lower entropy ⇒ routing is more concentrated / predictable. Reported
    in *nats* and normalised by log(n_experts) so 1.0 = uniform, 0.0 =
    a single expert chosen for every slot. Phase router with deterministic
    capacity bands is expected to be slightly *lower* than top-k.
    """
    valid = idx[idx >= 0]
    if valid.numel() == 0:
        return 0.0
    counts = torch.bincount(valid.flatten().long(), minlength=n_experts).float()
    total = counts.sum().clamp_min(1.0)
    p = counts / total
    # Drop zeros to avoid 0*log0
    nz = p[p > 0]
    H = -(nz * nz.log()).sum().item()
    return H / max(1e-9, math.log(n_experts))


def _switch_capacity(n_tokens: int, n_experts: int, k: int, factor: float) -> int:
    """Per-expert hard cap, identical to Switch / GShard."""
    return max(1, math.ceil(factor * n_tokens * k / n_experts))


def _enforce_capacity_switch(
    idx: torch.Tensor,        # (N, k) int64, values in [0, n_experts)
    weights: torch.Tensor,    # (N, k) float, gate weights at idx (differentiable)
    n_experts: int,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Drop tokens that overflow per-expert capacity (Switch policy).

    For each (token, slot), set idx → -1 and zero the weight if that
    expert's running load is already at capacity. Slots are processed
    in priority order (slot 0 first), matching Switch's behaviour of
    keeping the highest-weighted assignment.

    Gradients flow through `weights` because we only build a boolean
    keep-mask in numpy and then apply it back to the live tensor.
    """
    N, k = idx.shape
    # Decide keep/drop on CPU — purely from indices, no gradient needed.
    idx_cpu = idx.detach().cpu().numpy()
    keep = np.ones((N, k), dtype=bool)
    loads = np.zeros(n_experts, dtype=np.int64)
    dropped = 0
    for slot in range(k):
        for tok in range(N):
            e = idx_cpu[tok, slot]
            if e < 0:
                continue
            if loads[e] < capacity:
                loads[e] += 1
            else:
                keep[tok, slot] = False
                dropped += 1

    keep_t = torch.from_numpy(keep).to(idx.device)
    # idx: dropped slots → -1
    out_idx = torch.where(keep_t, idx, torch.full_like(idx, -1))
    # weights: zero dropped slots, then renormalise (differentiable).
    out_w = weights * keep_t.to(weights.dtype)
    s = out_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    out_w = out_w / s
    return out_idx, out_w, dropped


# ── Top-K (baseline) ───────────────────────────────────────────────────


class TopKRouter(nn.Module):
    """Switch / GShard top-k gating with load-balancing aux loss."""

    def __init__(self, n_experts: int, capacity_factor: float = 1.25,
                 aux_loss_alpha: float = 0.01):
        super().__init__()
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.aux_loss_alpha = aux_loss_alpha

    def forward(self, logits: torch.Tensor, k: int):
        N, E = logits.shape
        t_route = time.perf_counter()
        gate = F.softmax(logits, dim=-1)                     # (N, E)
        weights, idx = gate.topk(k, dim=-1)                  # (N, k)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

        capacity = _switch_capacity(N, E, k, self.capacity_factor)
        idx, weights, dropped = _enforce_capacity_switch(
            idx, weights, E, capacity
        )
        route_time_ms = (time.perf_counter() - t_route) * 1000.0

        # Switch aux loss: encourages uniform load.
        # importance = mean gate probability per expert
        # fraction   = mean number of times each expert was chosen (top-1)
        with torch.no_grad():
            one_hot = F.one_hot(idx.clamp_min(0), num_classes=E).float()
            mask = (idx >= 0).unsqueeze(-1).float()
            fraction = (one_hot * mask).sum(dim=(0, 1)) / (mask.sum() + 1e-9)
        importance = gate.mean(dim=0)
        aux_loss = self.aux_loss_alpha * E * (fraction.detach() * importance).sum()

        aux = {
            "dropped_frac": dropped / max(1, N * k),
            "load_cv": load_cv(idx, E),
            "contig_frac": contig_frac(idx),
            "perm_entropy": perm_entropy(idx, E),
            "route_time_ms": route_time_ms,
            "aux_loss": aux_loss,
            "capacity": capacity,
        }
        return idx, weights, aux


# ── Phase Router (Rust kernel) ─────────────────────────────────────────


def _build_t_bits(t_ones: np.ndarray, n: int, nb_words: int) -> np.ndarray:
    """Vectorised left-aligned bit packing for target rows.

    Row i gets `t_ones[i]` consecutive low bits set.
    """
    out = np.zeros(n * nb_words, dtype=np.uint64)
    # Iterate words rather than bits — n is small (≤ a few thousand).
    for i in range(n):
        ones = int(min(t_ones[i], n))
        full_words = ones // 64
        tail = ones - full_words * 64
        base = i * nb_words
        if full_words > 0:
            out[base:base + full_words] = np.uint64(0xFFFFFFFFFFFFFFFF)
        if tail > 0:
            out[base + full_words] = (np.uint64(1) << np.uint64(tail)) - np.uint64(1)
    return out


def _build_s_bits_uniform(s_ones: int, n: int, nb_words: int) -> np.ndarray:
    """Every source row gets the same left-aligned popcount = s_ones."""
    out = np.zeros(n * nb_words, dtype=np.uint64)
    ones = int(min(s_ones, n))
    full_words = ones // 64
    tail = ones - full_words * 64
    proto = np.zeros(nb_words, dtype=np.uint64)
    if full_words > 0:
        proto[:full_words] = np.uint64(0xFFFFFFFFFFFFFFFF)
    if tail > 0:
        proto[full_words] = (np.uint64(1) << np.uint64(tail)) - np.uint64(1)
    out[:] = np.tile(proto, n)
    return out


class PhaseRouter(nn.Module):
    """Capacity-aware MoE router backed by `phase_router_rs`.

    Selection (which experts) comes from the deterministic Rust kernel.
    Routing weights still flow gradient through the gate via
    `softmax(logits).gather(-1, idx)`.

    Internally we map kernel columns ∈ [0, N) to experts by tiling:
    expert e owns columns [e*width, (e+1)*width) where width = N // n_experts.
    Per-expert capacity is encoded as the ones-count in each row of the
    target bit matrix, taken from an EMA of observed gate demand.
    """

    def __init__(
        self,
        n_experts: int,
        capacity_factor: float = 1.25,
        seed: int = 0,
        ema: float = 0.99,
        base_density: float = 0.3,
        oversample: int = 4,
        capacity_mode: str = "uniform",   # "uniform" | "ema"
    ):
        super().__init__()
        if capacity_mode not in ("uniform", "ema"):
            raise ValueError(
                f"capacity_mode must be 'uniform' or 'ema', got {capacity_mode!r}"
            )
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.seed = seed
        self.ema = ema
        self.base_density = base_density
        self.oversample = oversample
        self.capacity_mode = capacity_mode
        # Per-expert EMA of average gate probability (~ relative demand).
        # Only consulted when capacity_mode == "ema".
        self.register_buffer(
            "cap_ema", torch.full((n_experts,), 1.0 / n_experts)
        )
        self._step = 0

    @torch.no_grad()
    def _select(self, gate: torch.Tensor, k: int) -> torch.Tensor:
        """Return (N, k) int64 expert ids (or -1) via the Rust kernel.

        Fast path (`capacity_mode == "uniform"`): one call into
        `phase_router_rs.phase_router_uniform_dispatch`, which fuses
        bit-packing, kernel invocation, expert-band mapping, and per-row
        unique-pick into a single GIL-released Rust function.

        Slow path (`capacity_mode == "ema"`): builds non-uniform `t_bits`
        in NumPy and calls `phase_router_auto` (kept for ablations).
        """
        import phase_router_rs

        N, E = gate.shape

        # ── fast path: uniform capacity (the production setting) ──────
        if self.capacity_mode == "uniform":
            routes = phase_router_rs.phase_router_uniform_dispatch(
                int(N),
                int(E),
                int(k),
                float(self.base_density),
                int(self.seed + self._step),
                int(self.oversample),
            )
            # routes: (N, k) int32 in [0, E) with -1 padding
            return torch.from_numpy(routes.astype(np.int64))

        # ── slow path: EMA-shaped capacities (ablation only) ──────────
        # Pad N to a multiple of n_experts so banding is clean.
        width = max(1, N // E)
        N_pad = width * E
        if N_pad < N:
            N_pad = (N // E + 1) * E
            width = N_pad // E

        nb_words = (N_pad + 63) // 64

        t_ones = np.zeros(N_pad, dtype=np.int64)
        rel = self.cap_ema.detach().cpu().numpy().astype(np.float64)
        rel = rel / max(rel.mean(), 1e-12)
        for e in range(E):
            ones_e = max(
                1,
                int(round(rel[e] * self.base_density * width)),
            )
            t_ones[e * width:(e + 1) * width] = min(ones_e, N_pad)
        t_bits = _build_t_bits(t_ones, N_pad, nb_words)

        s_ones_pertok = max(1, int(round(self.base_density * N_pad)))
        s_bits = _build_s_bits_uniform(s_ones_pertok, N_pad, nb_words)

        k_kernel = max(k * self.oversample, k + 1)
        k_kernel = min(k_kernel, N_pad)
        routes = phase_router_rs.phase_router_auto(
            s_bits, t_bits, N_pad, k_kernel, self.seed + self._step
        )

        routes = routes[:N]
        expert_ids = routes.astype(np.int64) // width
        expert_ids[routes < 0] = -1
        expert_ids[expert_ids >= E] = -1

        out = np.full((N, k), -1, dtype=np.int64)
        for r in range(N):
            seen: set[int] = set()
            j = 0
            for c in expert_ids[r]:
                if c < 0 or c in seen:
                    continue
                seen.add(int(c))
                out[r, j] = int(c)
                j += 1
                if j == k:
                    break
        return torch.from_numpy(out)


    def forward(self, logits: torch.Tensor, k: int):
        N, E = logits.shape
        t_route = time.perf_counter()
        gate = F.softmax(logits, dim=-1)             # (N, E)

        # Update EMA of expert demand on every forward.
        with torch.no_grad():
            avg = gate.mean(dim=0)
            self.cap_ema.mul_(self.ema).add_(avg, alpha=1 - self.ema)
            # Clamp so a collapsing EMA can't zero out an expert.
            lo = 0.5 / E
            hi = 4.0 / E
            self.cap_ema.clamp_(min=lo, max=hi)

        # Selection on CPU (kernel is CPU; tensors are small).
        idx = self._select(gate.detach(), k).to(logits.device)
        route_time_ms = (time.perf_counter() - t_route) * 1000.0

        # Weights = softmax(logits) gathered at idx, renormalised.
        # idx == -1 means dropped → zero weight.
        gather_idx = idx.clamp_min(0)
        gathered = gate.gather(-1, gather_idx)
        weights = gathered * (idx >= 0).float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

        # Capacity stat (informational — kernel already balances by construction)
        capacity = _switch_capacity(N, E, k, self.capacity_factor)
        dropped = (idx == -1).float().sum().item()
        aux = {
            "dropped_frac": dropped / max(1, N * k),
            "load_cv": load_cv(idx, E),
            "contig_frac": contig_frac(idx),
            "perm_entropy": perm_entropy(idx, E),
            "route_time_ms": route_time_ms,
            "aux_loss": torch.tensor(0.0, device=logits.device),
            "capacity": capacity,
        }
        self._step += 1
        return idx, weights, aux


# ── Balanced Router (constrained-optimisation ensemble) ────────────────


class BalancedRouter(nn.Module):
    """Priority-ordered occupancy-aware admission router.

    Solves *maximise affinity subject to per-expert occupancy constraints*
    as a greedy assignment:

        1. Compute top-(k + overflow) candidates per token by gate score.
        2. Process tokens in descending max-affinity order.
        3. Each token claims its highest-scoring candidate whose quota is
           not yet full. Falls through to the 2nd / 3rd / ... candidate
           on overflow rather than being dropped — this is the *soft*
           admission process the design constraints require.
        4. A slot is dropped (idx = -1) only when every candidate in the
           token's (k + overflow)-deep list is at quota; in practice
           this is < 0.5 % at any sensible (cf, overflow).

    Reduces to top-k as cf → ∞ (quota becomes non-binding) and
    approximates phase-uniform as cf → 1 (quota becomes tight). One
    knob, same shape as Switch / GShard.

    See `dev/ensemble_probe_plan.md` for the design rationale, the
    pre-registered hypothesis grid, and the success criteria.
    """

    def __init__(
        self,
        n_experts: int,
        capacity_factor: float = 1.25,
        overflow: int = 2,
        aux_loss_alpha: float = 0.0,   # 0 → no aux; quota is the constraint
    ):
        super().__init__()
        if overflow < 0:
            raise ValueError(f"overflow must be ≥ 0, got {overflow}")
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor
        self.overflow = overflow
        self.aux_loss_alpha = aux_loss_alpha

    @torch.no_grad()
    def _select(
        self, gate: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (idx (N,k) int64, w_at_idx (N,k) float) on CPU.

        Soft admission: descending priority, fall-through on quota
        overflow. Determines selection only; differentiable weight
        re-binding happens in `forward`.
        """
        N, E = gate.shape
        k_cand = min(E, k + self.overflow)

        # Candidate scores per token, descending. (N, k_cand)
        top_w, top_idx = gate.topk(k_cand, dim=-1)

        # Priority: most-confident tokens first.
        priority = torch.argsort(-top_w[:, 0])

        # Quota: same formula as Switch top-k. cf=1 ⇒ tight, cf=∞ ⇒ loose.
        quota = max(1, math.ceil(self.capacity_factor * N * k / E))

        top_idx_np = top_idx.cpu().numpy().astype(np.int64)
        priority_np = priority.cpu().numpy().astype(np.int64)

        loads = np.zeros(E, dtype=np.int64)
        out_idx = np.full((N, k), -1, dtype=np.int64)

        for tok in priority_np:
            slot = 0
            row = top_idx_np[tok]
            for c in row:
                if loads[c] < quota:
                    out_idx[tok, slot] = c
                    loads[c] += 1
                    slot += 1
                    if slot == k:
                        break

        idx_t = torch.from_numpy(out_idx)
        return idx_t

    def forward(self, logits: torch.Tensor, k: int):
        N, E = logits.shape
        t_route = time.perf_counter()
        gate = F.softmax(logits, dim=-1)              # (N, E)

        idx = self._select(gate.detach().cpu(), k).to(logits.device)
        route_time_ms = (time.perf_counter() - t_route) * 1000.0

        # Differentiable weight binding: gather softmax weights at the
        # selected experts and renormalise per row. Dropped slots get
        # zero weight (and idx = -1 keeps them out of the dispatch).
        gather_idx = idx.clamp_min(0)
        gathered = gate.gather(-1, gather_idx)
        weights = gathered * (idx >= 0).float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)

        # Optional aux loss — disabled by default. The quota IS the
        # constraint; an aux term would double-count. Exposed so we can
        # ablate "balanced + aux" if the first sweep underperforms.
        if self.aux_loss_alpha > 0:
            with torch.no_grad():
                one_hot = F.one_hot(idx.clamp_min(0), num_classes=E).float()
                mask = (idx >= 0).unsqueeze(-1).float()
                fraction = (one_hot * mask).sum(dim=(0, 1)) / (mask.sum() + 1e-9)
            importance = gate.mean(dim=0)
            aux_loss = self.aux_loss_alpha * E * (fraction.detach() * importance).sum()
        else:
            aux_loss = torch.tensor(0.0, device=logits.device)

        capacity = _switch_capacity(N, E, k, self.capacity_factor)
        dropped = (idx == -1).float().sum().item()
        aux = {
            "dropped_frac": dropped / max(1, N * k),
            "load_cv": load_cv(idx, E),
            "contig_frac": contig_frac(idx),
            "perm_entropy": perm_entropy(idx, E),
            "route_time_ms": route_time_ms,
            "aux_loss": aux_loss,
            "capacity": capacity,
        }
        return idx, weights, aux
