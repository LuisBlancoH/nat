#!/usr/bin/env python
"""
Entry point for the NAT evaluation suite.

Runs all four evaluations and reports results.

Usage
-----
::

    # Full evaluation with a trained checkpoint
    python scripts/evaluate.py --config configs/base.yaml \\
        --checkpoint checkpoints/phase3.pt

    # Quick synthetic evaluation (no checkpoint needed)
    python scripts/evaluate.py --config configs/small.yaml --synthetic

    # Save results to disk
    python scripts/evaluate.py --config configs/small.yaml --synthetic \\
        --output-dir results/

    # With W&B logging
    python scripts/evaluate.py --config configs/base.yaml --wandb \\
        --checkpoint checkpoints/phase3.pt

    # Run only specific evaluations
    python scripts/evaluate.py --config configs/small.yaml --synthetic \\
        --only within_session cross_session
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
from nat.training.phase1_meta_learn import load_checkpoint
from nat.training.eval import (
    evaluate_within_session,
    evaluate_cross_session,
    evaluate_forgetting,
    evaluate_baseline,
    run_full_evaluation,
)


def main():
    parser = argparse.ArgumentParser(
        description="NAT Evaluation Suite"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint to evaluate",
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
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (JSON + summary)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["within_session", "cross_session", "forgetting", "baseline"],
        default=None,
        help="Run only specific evaluations",
    )
    parser.add_argument(
        "--within-episodes",
        type=int,
        default=10,
        help="Number of episodes for within-session eval",
    )
    parser.add_argument(
        "--cross-sessions",
        type=int,
        default=20,
        help="Number of sessions for cross-session eval",
    )
    parser.add_argument(
        "--cross-domain",
        type=str,
        default="math",
        help="Domain for cross-session eval",
    )
    parser.add_argument(
        "--forgetting-domain-a",
        type=str,
        default="math",
        help="First domain for forgetting test",
    )
    parser.add_argument(
        "--forgetting-domain-b",
        type=str,
        default="code",
        help="Second domain for forgetting test",
    )
    parser.add_argument(
        "--baseline-episodes",
        type=int,
        default=10,
        help="Number of episodes for baseline comparison",
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
    logging.info(f"Config: {config.to_dict()}")

    # ---- Build model ----
    logging.info(f"Loading base model: {config.base_model_name}")
    model = NATModel(config)

    # ---- Load checkpoint ----
    if args.checkpoint:
        episode = load_checkpoint(model, args.checkpoint)
        logging.info(f"Loaded checkpoint from episode {episode}")
    else:
        logging.info("No checkpoint specified — evaluating untrained model")

    # ---- Move to device ----
    device = torch.device(config.device)
    model = model.to(device)

    # ---- Run evaluations ----
    if args.only is None:
        # Full suite
        results = run_full_evaluation(
            model,
            config,
            synthetic=args.synthetic,
            num_within_episodes=args.within_episodes,
            num_cross_sessions=args.cross_sessions,
            num_baseline_episodes=args.baseline_episodes,
            cross_session_domain=args.cross_domain,
            forgetting_domain_a=args.forgetting_domain_a,
            forgetting_domain_b=args.forgetting_domain_b,
            output_dir=args.output_dir,
            use_wandb=args.wandb,
        )
        if results["all_passed"]:
            logging.info("All evaluations PASSED ✓")
        else:
            logging.warning("Some evaluations FAILED ✗")
    else:
        # Selective evaluations
        eval_funcs = {
            "within_session": lambda: evaluate_within_session(
                model, config,
                num_episodes=args.within_episodes,
                synthetic=args.synthetic,
            ),
            "cross_session": lambda: evaluate_cross_session(
                model, config,
                domain=args.cross_domain,
                num_sessions=args.cross_sessions,
                synthetic=args.synthetic,
            ),
            "forgetting": lambda: evaluate_forgetting(
                model, config,
                domain_a=args.forgetting_domain_a,
                domain_b=args.forgetting_domain_b,
                synthetic=args.synthetic,
            ),
            "baseline": lambda: evaluate_baseline(
                model, config,
                num_episodes=args.baseline_episodes,
                synthetic=args.synthetic,
            ),
        }

        for name in args.only:
            logging.info(f"─── Running: {name} ───")
            result = eval_funcs[name]()
            logging.info(result.summary)
            for k, v in result.metrics.items():
                if isinstance(v, float):
                    logging.info(f"  {k}: {v:.4f}")

            if args.output_dir:
                result.to_json(Path(args.output_dir) / f"{name}.json")


if __name__ == "__main__":
    main()
