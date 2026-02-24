#!/usr/bin/env python
"""
Entry point for Phase 2: Consolidation dynamics training.

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
from nat.training.phase2_consolidation import train_phase2
from nat.training.train_utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="NAT Phase 2: Consolidation dynamics training"
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
        "--runs",
        type=int,
        default=None,
        help="Override num_runs_p2 from config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (typically Phase 1 output)",
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
        help="Override any config field, e.g. --set lr_phase2=1e-4 sessions_per_domain_p2=30",
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
    if args.runs is not None:
        config.num_runs_p2 = args.runs
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

    # ---- Resume from checkpoint (Phase 1 or prior Phase 2) ----
    if args.resume:
        episode = load_checkpoint(model, args.resume)
        logging.info(f"Loaded checkpoint from episode {episode}")

    # ---- Print summary ----
    model.print_param_summary()

    # ---- Train ----
    result = train_phase2(
        model,
        config,
        use_wandb=args.wandb,
        synthetic=args.synthetic,
    )

    logging.info(
        f"Phase 2 complete — "
        f"β={result['final_beta']:.4f}, α={result['final_alpha']:.3f}, "
        f"best improvement={result['best_improvement']:+.4f}, "
        f"saved to: {result['save_path']}"
    )


if __name__ == "__main__":
    main()
