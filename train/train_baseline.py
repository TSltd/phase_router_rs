"""Run A — baseline top-k gating.

Usage:
    python train/train_baseline.py --config train/configs/tiny.yaml
"""
from _loop import run

if __name__ == "__main__":
    run("topk")
