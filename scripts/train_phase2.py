#!/usr/bin/env python
"""
Entry point for Phase 2: Episodic multi-task training.

Usage
-----
::

    # Full training (requires Phase 1 checkpoint)
    python scripts/train_phase2.py --config configs/base.yaml \\
        --resume checkpoints/phase1.pt

    # Small-scale debugging on Apple Silicon
    python scripts/train_phase2.py --config configs/small.yaml --synthetic

    # Synthetic data (no downloads, for CI / smoke tests)
    python scripts/train_phase2.py --config configs/small.yaml --synthetic

    # With W&B logging
    python scripts/train_phase2.py --config configs/base.yaml --wandb \\
        --resume checkpoints/phase1.pt
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
from nat.training.phase2_episodic import train_phase2
from nat.training.phase1_meta_learn import load_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="NAT Phase 2: Episodic multi-task training"
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
        help="Use synthetic episodic data (no downloads, fast)",
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
        help="Override num_episodes_p2 from config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (typically Phase 1 output)",
    )
    args = parser.parse_args()

    # ---- Setup logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- Load config ----
    config = NATConfig.from_yaml(args.config)
    if args.episodes is not None:
        config.num_episodes_p2 = args.episodes

    logging.info(f"Config: {config.to_dict()}")

    # ---- Build model ----
    logging.info(f"Loading base model: {config.base_model_name}")
    model = NATModel(config)

    # ---- Resume from checkpoint (Phase 1 or prior Phase 2) ----
    if args.resume:
        episode = load_checkpoint(model, args.resume)
        logging.info(f"Loaded checkpoint from episode {episode}")

    # ---- Print summary ----
    trainable = model.get_trainable_parameters()
    total_params = sum(p.numel() for p in trainable)
    logging.info(f"Trainable parameters: {total_params:,}")
    model.print_param_summary()

    # ---- Train ----
    result = train_phase2(
        model,
        config,
        use_wandb=args.wandb,
        synthetic=args.synthetic,
    )

    logging.info(
        f"Phase 2 complete — final loss: {result['final_loss']:.4f}, "
        f"saved to: {result['save_path']}"
    )


if __name__ == "__main__":
    main()
