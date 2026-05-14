"""Run B — Phase Router (Rust kernel) gating.

Usage:
    python train/train_phase.py --config train/configs/tiny.yaml
"""
from _loop import run

if __name__ == "__main__":
    run("phase")
