#!/usr/bin/env python
"""
Entry point for Phase 1: Meta-learn the learning rule θ.

Usage
-----
::

    # Full training with Qwen2.5-1.5B
    python scripts/train_phase1.py --config configs/base.yaml

    # Small-scale debugging on Apple Silicon
    python scripts/train_phase1.py --config configs/small.yaml

    # Synthetic data (no downloads, for CI / smoke tests)
    python scripts/train_phase1.py --config configs/small.yaml --synthetic

    # With W&B logging
    python scripts/train_phase1.py --config configs/base.yaml --wandb
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from nat.config import NATConfig
from nat.model.nat_model import NATModel
from nat.training.phase1_meta_learn import train_phase1


def main():
    parser = argparse.ArgumentParser(
        description="NAT Phase 1: Meta-learn the learning rule θ"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (no downloads, fast)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override num_episodes from config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # ---- Setup logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy HTTP loggers from HF dataset streaming
    for _noisy in ("httpx", "urllib3", "filelock", "fsspec"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    # ---- Load config ----
    config = NATConfig.from_yaml(args.config)
    if args.episodes is not None:
        config.num_episodes_p1 = args.episodes
        config.num_episodes = args.episodes

    logging.info(f"Config: {config.to_dict()}")

    # ---- Build model ----
    logging.info(f"Loading base model: {config.base_model_name}")
    model = NATModel(config)

    # ---- Resume from checkpoint ----
    if args.resume:
        from nat.training.phase1_meta_learn import load_checkpoint
        episode = load_checkpoint(model, args.resume)
        logging.info(f"Resumed from episode {episode}")

    # ---- Print summary ----
    trainable = model.get_trainable_parameters()
    total_params = sum(p.numel() for p in trainable)
    logging.info(f"Trainable parameters: {total_params:,}")
    model.print_param_summary()

    # ---- Train ----
    result = train_phase1(
        model,
        config,
        use_wandb=args.wandb,
        synthetic=args.synthetic,
    )

    logging.info(
        f"Training complete — final loss: {result['final_loss']:.4f}, "
        f"saved to: {result['save_path']}"
    )


if __name__ == "__main__":
    main()
