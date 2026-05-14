"""Tiny GPT-style decoder with MoE feed-forward blocks.

Architecture (deliberately simple — no flash-attention, no fancy norms)
so that the router is the only moving part between runs:

    Embed → [ MHSA + MoE-FFN ] * n_layers → LayerNorm → LM head

Every block uses MoE in the FFN. The router (TopKRouter or PhaseRouter)
is injected by the training loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    n_experts: int
    k: int
    seq_len: int


class MHSA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, Dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-1, -2)) / (self.d_head ** 0.5)
        attn = attn.masked_fill(self.mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class MoEFFN(nn.Module):
    """Mixture-of-Experts feed-forward.

    Each expert is a 2-layer MLP (d_model → d_ff → d_model). The router
    chooses up to `k` experts per token and produces routing weights.
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int, k: int, router: nn.Module):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k
        self.router = router
        # We keep experts as a single batched MLP for speed:
        # W1: (E, D, F), W2: (E, F, D)
        self.W1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.W2 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        nn.init.kaiming_uniform_(self.W1, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.W2, a=5 ** 0.5)
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        flat = x.reshape(B * T, D)                     # (N, D)
        logits = self.gate(flat)                       # (N, E)
        idx, weights, aux = self.router(logits, self.k)
        # idx: (N, k) int64 with -1 for dropped slots
        # weights: (N, k) float

        N = flat.size(0)
        out = torch.zeros_like(flat)
        for slot in range(self.k):
            slot_idx = idx[:, slot]                    # (N,)
            slot_w = weights[:, slot].unsqueeze(-1)    # (N, 1)
            for e in range(self.n_experts):
                mask = slot_idx == e
                if not mask.any():
                    continue
                tok = flat[mask]                       # (n_e, D)
                h = tok @ self.W1[e]                   # (n_e, F)
                h = F.gelu(h)
                y = h @ self.W2[e]                     # (n_e, D)
                out[mask] = out[mask] + slot_w[mask] * y
        return out.reshape(B, T, D), aux


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, router: nn.Module):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MHSA(cfg.d_model, cfg.n_heads, cfg.seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = MoEFFN(cfg.d_model, cfg.d_ff, cfg.n_experts, cfg.k, router)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        y, aux = self.ff(self.ln2(x))
        x = x + y
        return x, aux


class MoELM(nn.Module):
    def __init__(self, cfg: ModelConfig, router_factory):
        """router_factory: zero-arg callable producing a fresh router per block."""
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg, router_factory()) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight tying
        self.head.weight = self.tok_emb.weight
        self.register_buffer(
            "pos_ids", torch.arange(cfg.seq_len).unsqueeze(0), persistent=False
        )

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = x.shape
        h = self.tok_emb(x) + self.pos_emb(self.pos_ids[:, :T])
        aux_total: dict[str, Any] = {
            "aux_loss": torch.tensor(0.0, device=x.device),
            "load_cv": 0.0,
            "dropped_frac": 0.0,
            "contig_frac": 0.0,
            "perm_entropy": 0.0,
            "route_time_ms": 0.0,
        }
        for block in self.blocks:
            h, aux = block(h)
            aux_total["aux_loss"] = aux_total["aux_loss"] + aux["aux_loss"]
            aux_total["load_cv"] += aux["load_cv"]
            aux_total["dropped_frac"] += aux["dropped_frac"]
            aux_total["contig_frac"] += aux.get("contig_frac", 0.0)
            aux_total["perm_entropy"] += aux.get("perm_entropy", 0.0)
            aux_total["route_time_ms"] += aux.get("route_time_ms", 0.0)
        nb = len(self.blocks)
        aux_total["load_cv"] /= nb
        aux_total["dropped_frac"] /= nb
        aux_total["contig_frac"] /= nb
        aux_total["perm_entropy"] /= nb
        # route_time_ms is kept as a SUM across layers — that's the
        # total routing cost per forward, which is what we want to
        # compare against total step time.

        h = self.ln_f(h)
        logits = self.head(h)
        if targets is None:
            return logits, aux_total
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
        total = loss + aux_total["aux_loss"]
        return total, {"ce_loss": loss.detach().item(),
                       "aux_loss": aux_total["aux_loss"].detach().item(),
                       "load_cv": aux_total["load_cv"],
                       "dropped_frac": aux_total["dropped_frac"],
                       "contig_frac": aux_total["contig_frac"],
                       "perm_entropy": aux_total["perm_entropy"],
                       "route_time_ms": aux_total["route_time_ms"]}


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
