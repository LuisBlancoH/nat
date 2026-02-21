#!/usr/bin/env python3
"""
Measure the frozen base model's loss (no adaptive/consolidation layers).

Computes average loss over N samples from C4 streaming data using ONLY
the frozen Qwen2.5-1.5B — no fast weights, no consolidation.  This gives
a constant reference to compare against Phase 1 training metrics.

Usage:
    python scripts/measure_frozen_baseline.py --config configs/a100.yaml
    python scripts/measure_frozen_baseline.py --config configs/a100.yaml --samples 200
"""

import argparse
import logging
import torch
from nat.config import NATConfig

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Measure frozen baseline loss")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Number of samples to average over",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = NATConfig.from_yaml(args.config)
    device = torch.device(config.device)

    # ---- Load raw base model (no NAT layers) ----
    logger.info(f"Loading frozen base model: {config.base_model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        dtype=getattr(config, "torch_dtype", torch.bfloat16),
    ).to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)

    # ---- Load streaming data ----
    logger.info("Loading C4 streaming data...")
    from nat.training.data import build_phase1_dataloader
    dataloader = build_phase1_dataloader(config, tokenizer=tokenizer)

    # ---- Measure loss ----
    logger.info(f"Computing frozen baseline over {args.samples} samples...")
    total_loss = 0.0
    data_iter = iter(dataloader)

    for i in range(args.samples):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = input_ids.clone()

        with torch.no_grad():
            output = base_model(input_ids, labels=labels)
            total_loss += output.loss.item()

        if (i + 1) % 10 == 0:
            avg_so_far = total_loss / (i + 1)
            logger.info(f"  [{i + 1}/{args.samples}] running avg loss = {avg_so_far:.4f}")

    avg_loss = total_loss / args.samples
    logger.info("=" * 60)
    logger.info(f"FROZEN BASELINE LOSS: {avg_loss:.4f}")
    logger.info(f"  (averaged over {args.samples} samples, seq_len={config.seq_len})")
    logger.info("=" * 60)
    logger.info(
        "Compare this to Phase 1 'loss' metric.\n"
        "  loss < frozen_baseline → NAT layers are helping\n"
        "  loss > frozen_baseline → NAT layers are hurting (early training, normal)"
    )


if __name__ == "__main__":
    main()
