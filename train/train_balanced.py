"""Run C — Balanced Router (constrained-optimisation ensemble).

Top-k scores → phase-style quota → soft fall-through admission.
See `dev/ensemble_probe_plan.md` for the algorithm and motivation.

Usage:
    python train/train_balanced.py --config train/configs/stress.yaml
"""
from _loop import run

if __name__ == "__main__":
    run("balanced")
