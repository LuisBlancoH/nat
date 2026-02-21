#!/usr/bin/env python
"""
Entry point for Phase 3: Consolidation dynamics training.

Usage
-----
::

    # Full training (requires Phase 1+2 checkpoint)
    python scripts/train_phase3.py --config configs/base.yaml \\
        --resume checkpoints/phase2.pt

    # Small-scale debugging on Apple Silicon
    python scripts/train_phase3.py --config configs/small.yaml --synthetic

    # Synthetic data (no downloads, for CI / smoke tests)
    python scripts/train_phase3.py --config configs/small.yaml --synthetic

    # With W&B logging
    python scripts/train_phase3.py --config configs/base.yaml --wandb \\
        --resume checkpoints/phase2.pt
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
from nat.training.phase3_consolidation import train_phase3
from nat.training.phase1_meta_learn import load_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="NAT Phase 3: Consolidation dynamics training"
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
        help="Use synthetic domain data (no downloads, fast)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override num_runs_p3 from config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (typically Phase 2 output)",
    )
    args = parser.parse_args()

    # ---- Setup logging ----
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silence noisy HTTP loggers from HF dataset streaming
    for _noisy in ("httpx", "urllib3", "filelock", "fsspec", "huggingface_hub"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    # ---- Load config ----
    config = NATConfig.from_yaml(args.config)
    if args.runs is not None:
        config.num_runs_p3 = args.runs

    logging.info(f"Config: {config.to_dict()}")

    # ---- Build model ----
    logging.info(f"Loading base model: {config.base_model_name}")
    model = NATModel(config)

    # ---- Resume from checkpoint (Phase 2 or prior Phase 3) ----
    if args.resume:
        episode = load_checkpoint(model, args.resume)
        logging.info(f"Loaded checkpoint from episode {episode}")

    # ---- Print summary ----
    consolidation_params = sum(
        p.numel() for p in model.consolidation.parameters()
    )
    logging.info(f"Consolidation parameters: {consolidation_params:,}")
    logging.info("+ 2 scalar parameters: β (logit), α (logit)")
    model.print_param_summary()

    # ---- Train ----
    result = train_phase3(
        model,
        config,
        use_wandb=args.wandb,
        synthetic=args.synthetic,
    )

    logging.info(
        f"Phase 3 complete — "
        f"β={result['final_beta']:.4f}, α={result['final_alpha']:.3f}, "
        f"best improvement={result['best_improvement']:+.4f}, "
        f"saved to: {result['save_path']}"
    )


if __name__ == "__main__":
    main()
