"""Shared training loop used by train_baseline.py and train_phase.py.

Both entry points call `run("topk")` or `run("phase")`. Everything else
about the run — data, model, optimiser, seed — is identical, so any
difference in metrics is attributable to the router.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# Make sibling files importable when run as `python train/train_phase.py`.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data import get_tokenizer, windowed_batches  # noqa: E402
from moe_model import MoELM, ModelConfig, count_params  # noqa: E402
from routers import BalancedRouter, PhaseRouter, TopKRouter  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def make_router_factory(name: str, cfg: dict):
    if name == "topk":
        return lambda: TopKRouter(
            n_experts=cfg["n_experts"],
            capacity_factor=cfg["capacity_factor"],
            aux_loss_alpha=cfg["aux_loss_alpha"],
        )
    elif name == "phase":
        return lambda: PhaseRouter(
            n_experts=cfg["n_experts"],
            capacity_factor=cfg["capacity_factor"],
            seed=cfg["seed"],
            capacity_mode=cfg.get("phase_capacity_mode", "uniform"),
        )
    elif name == "balanced":
        # Constrained-optimisation ensemble: top-k affinity scores +
        # phase-style per-expert quotas with soft fall-through.
        # See `dev/ensemble_probe_plan.md`.
        return lambda: BalancedRouter(
            n_experts=cfg["n_experts"],
            capacity_factor=cfg["capacity_factor"],
            overflow=cfg.get("balanced_overflow", 2),
            aux_loss_alpha=cfg.get("balanced_aux_loss_alpha", 0.0),
        )
    raise ValueError(f"unknown router: {name}")


def make_model(cfg: dict, router_factory) -> MoELM:
    mc = ModelConfig(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        n_experts=cfg["n_experts"],
        k=cfg["k"],
        seq_len=cfg["seq_len"],
    )
    return MoELM(mc, router_factory)


def cosine_lr(step: int, warmup: int, max_steps: int, base: float) -> float:
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model: MoELM, loader_iter, steps: int, device: str) -> dict:
    model.eval()
    ce_sum = 0.0
    n = 0
    cv_sum = 0.0
    drop_sum = 0.0
    contig_sum = 0.0
    pe_sum = 0.0
    rt_sum = 0.0
    for _ in range(steps):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        batch = batch.to(device, non_blocking=True)
        x, y = batch[:, :-1], batch[:, 1:]
        loss, info = model(x, y)
        ce_sum += info["ce_loss"]
        cv_sum += info["load_cv"]
        drop_sum += info["dropped_frac"]
        contig_sum += info.get("contig_frac", 0.0)
        pe_sum += info.get("perm_entropy", 0.0)
        rt_sum += info.get("route_time_ms", 0.0)
        n += 1
    model.train()
    if n == 0:
        return {
            "val_ce": float("nan"), "val_cv": float("nan"), "val_drop": float("nan"),
            "val_contig": float("nan"), "val_pe": float("nan"), "val_rt_ms": float("nan"),
        }
    return {
        "val_ce": ce_sum / n,
        "val_cv": cv_sum / n,
        "val_drop": drop_sum / n,
        "val_contig": contig_sum / n,
        "val_pe": pe_sum / n,
        "val_rt_ms": rt_sum / n,
    }


def run(router_name: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="where to write metrics.jsonl + ckpt (default: runs/<router>)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="override config.max_steps (handy for smoke tests)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda|cpu (auto-detect)")
    parser.add_argument("--capacity_factor", type=float, default=None,
                        help="override config.capacity_factor (sweep)")
    parser.add_argument("--aux_loss_alpha", type=float, default=None,
                        help="override config.aux_loss_alpha (top-k aux-loss ablation)")
    parser.add_argument("--seq_len", type=int, default=None,
                        help="override config.seq_len (locality scan)")
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.capacity_factor is not None:
        cfg["capacity_factor"] = args.capacity_factor
    if args.aux_loss_alpha is not None:
        cfg["aux_loss_alpha"] = args.aux_loss_alpha
    if args.seq_len is not None:
        cfg["seq_len"] = args.seq_len

    out_dir = Path(args.out_dir or f"runs/{router_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_f = open(metrics_path, "w")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cuda_avail = torch.cuda.is_available()
    cuda_name = torch.cuda.get_device_name(0) if cuda_avail else ""
    print(f"[{router_name}] device={device}  cuda_available={cuda_avail}  "
          f"cuda_device={cuda_name!r}  torch={torch.__version__}  "
          f"out_dir={out_dir}")
    set_seed(cfg["seed"])

    # ---- data -----------------------------------------------------------
    tok = get_tokenizer()
    train_iter = windowed_batches(
        cfg["dataset"], cfg["dataset_split_train"], cfg["text_field"],
        tok, cfg["seq_len"], cfg["batch_size"], seed=cfg["seed"],
    )
    val_iter = windowed_batches(
        cfg["dataset"], cfg["dataset_split_val"], cfg["text_field"],
        tok, cfg["seq_len"], cfg["batch_size"], seed=cfg["seed"] + 1,
    )

    # ---- model ----------------------------------------------------------
    router_factory = make_router_factory(router_name, cfg)
    model = make_model(cfg, router_factory).to(device)
    print(f"[{router_name}] params: {count_params(model):,}")

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=tuple(cfg["betas"]),
        weight_decay=cfg["weight_decay"],
    )

    # ---- train ----------------------------------------------------------
    step = 0
    grad_accum = cfg["grad_accum"]
    log_every = cfg["log_every"]
    val_every = cfg["val_every"]
    max_steps = cfg["max_steps"]

    t0 = time.time()
    running_loss = 0.0
    running_cv = 0.0
    running_drop = 0.0
    running_contig = 0.0
    running_pe = 0.0
    running_rt = 0.0
    running_n = 0

    model.train()
    while step < max_steps:
        opt.zero_grad(set_to_none=True)
        for micro in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                print(f"[{router_name}] training stream exhausted at step {step}")
                step = max_steps
                break
            batch = batch.to(device, non_blocking=True)
            x, y = batch[:, :-1], batch[:, 1:]
            loss, info = model(x, y)
            (loss / grad_accum).backward()
            running_loss += info["ce_loss"]
            running_cv += info["load_cv"]
            running_drop += info["dropped_frac"]
            running_contig += info.get("contig_frac", 0.0)
            running_pe += info.get("perm_entropy", 0.0)
            running_rt += info.get("route_time_ms", 0.0)
            running_n += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = cosine_lr(step, cfg["warmup"], max_steps, cfg["lr"])
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.step()

        if (step + 1) % log_every == 0 or step == 0:
            dt = time.time() - t0
            tokens_per_step = cfg["batch_size"] * cfg["seq_len"] * grad_accum
            throughput = tokens_per_step * (step + 1) / max(1e-9, dt)
            denom = max(1, running_n)
            # step_time_ms: wall time spent on the last `log_every` steps,
            # per step, in ms. Used to derive routing_overhead_pct in compare.py.
            step_time_ms = (dt * 1000.0) / max(1, step + 1)
            row = {
                "phase": "train",
                "step": step + 1,
                "ce": running_loss / denom,
                "load_cv": running_cv / denom,
                "drop": running_drop / denom,
                "contig_frac": running_contig / denom,
                "perm_entropy": running_pe / denom,
                "route_time_ms": running_rt / denom,
                "step_time_ms": step_time_ms,
                "lr": lr,
                "tok_per_s": throughput,
                "elapsed_s": dt,
            }
            metrics_f.write(json.dumps(row) + "\n")
            metrics_f.flush()
            print(f"[{router_name}] step={step+1:5d}  ce={row['ce']:.4f}  "
                  f"cv={row['load_cv']:.3f}  drop={row['drop']:.3f}  "
                  f"contig={row['contig_frac']:.3f}  H={row['perm_entropy']:.3f}  "
                  f"rt={row['route_time_ms']:.2f}ms  "
                  f"lr={lr:.2e}  tok/s={throughput:,.0f}")
            running_loss = running_cv = running_drop = 0.0
            running_contig = running_pe = running_rt = 0.0
            running_n = 0

        if (step + 1) % val_every == 0:
            metrics = evaluate(model, val_iter, cfg["val_steps"], device)
            row = {"phase": "val", "step": step + 1, **metrics}
            metrics_f.write(json.dumps(row) + "\n")
            metrics_f.flush()
            print(f"[{router_name}] VAL step={step+1}  ce={metrics['val_ce']:.4f}  "
                  f"cv={metrics['val_cv']:.3f}  drop={metrics['val_drop']:.3f}  "
                  f"contig={metrics['val_contig']:.3f}  "
                  f"H={metrics['val_pe']:.3f}  "
                  f"rt={metrics['val_rt_ms']:.2f}ms")

        step += 1

    # final eval + checkpoint
    metrics = evaluate(model, val_iter, cfg["val_steps"], device)
    row = {"phase": "val_final", "step": step, **metrics}
    metrics_f.write(json.dumps(row) + "\n")
    metrics_f.flush()
    print(f"[{router_name}] FINAL VAL ce={metrics['val_ce']:.4f}  "
          f"cv={metrics['val_cv']:.3f}  drop={metrics['val_drop']:.3f}  "
          f"contig={metrics['val_contig']:.3f}  "
          f"H={metrics['val_pe']:.3f}  "
          f"rt={metrics['val_rt_ms']:.2f}ms")

    ckpt = out_dir / "model.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg, "router": router_name}, ckpt)
    print(f"[{router_name}] saved checkpoint → {ckpt}")
    metrics_f.close()
