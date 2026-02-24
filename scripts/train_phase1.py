#!/usr/bin/env python
"""
Entry point for Phase 1: Episodic multi-domain meta-learning.

Usage
-----
::

    # Full training with Qwen3-4B
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
from nat.training.phase1_episodic import train_phase1
from nat.training.train_utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="NAT Phase 1: Episodic multi-domain meta-learning"
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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override adapt_every_n (tokens per chunk). Default: 256",
    )
    parser.add_argument(
        "--set",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="Override any config field, e.g. --set lr_phase1=1e-4 truncated_bptt=2",
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
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    # ---- Load config ----
    config = NATConfig.from_yaml(args.config)
    if args.episodes is not None:
        config.num_episodes_p1 = args.episodes
    if args.chunk_size is not None:
        config.adapt_every_n = args.chunk_size

    # Generic --set overrides
    for kv in getattr(args, "set", []):
        if "=" not in kv:
            parser.error(f"--set values must be KEY=VALUE, got: {kv!r}")
        key, val = kv.split("=", 1)
        if not hasattr(config, key):
            parser.error(f"Unknown config field: {key!r}")
        field_type = type(getattr(config, key))
        try:
            if field_type is bool:
                coerced = val.lower() in ("true", "1", "yes")
            elif field_type is float:
                coerced = float(val)
            elif field_type is int:
                coerced = int(val)
            else:
                coerced = val
        except ValueError:
            parser.error(f"Cannot convert {val!r} to {field_type.__name__} for {key}")
        setattr(config, key, coerced)

    logging.info(f"Config: {config.to_dict()}")

    # ---- Build model ----
    logging.info(f"Loading base model: {config.base_model_name}")
    model = NATModel(config)

    # ---- Resume from checkpoint ----
    if args.resume:
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
