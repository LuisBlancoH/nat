#!/usr/bin/env python3
"""
Measure the frozen base model's loss (no adaptive/consolidation layers).

Computes average loss over N samples using ONLY the frozen Qwen2.5-1.5B
— no fast weights, no consolidation.  This gives a constant reference
to compare against training metrics for any phase.

Usage:
    # Phase 1 — language modelling on C4
    python scripts/measure_frozen_baseline.py --config configs/base.yaml --phase 1

    # Phase 2 — episodic QA (GSM8K, ARC, MMLU, …)
    python scripts/measure_frozen_baseline.py --config configs/base.yaml --phase 2

    # Phase 3 — per-domain (math, code, reasoning, …)
    python scripts/measure_frozen_baseline.py --config configs/base.yaml --phase 3

    # More samples for tighter estimate
    python scripts/measure_frozen_baseline.py --config configs/base.yaml --phase 1 --samples 200
"""

import argparse
import logging
import torch
from nat.config import NATConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _measure_lm_loss(model, dataloader, device, n_samples: int) -> float:
    """Measure next-token loss on a plain language-modelling dataloader."""
    total_loss = 0.0
    data_iter = iter(dataloader)
    for i in range(n_samples):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        with torch.no_grad():
            out = model(input_ids, labels=input_ids.clone())
            total_loss += out.loss.item()

        if (i + 1) % 10 == 0:
            logger.info(
                f"  [{i + 1}/{n_samples}] running avg = {total_loss / (i + 1):.4f}"
            )
    return total_loss / max(n_samples, 1)


def _measure_episodic_loss(model, dataloader, device, n_samples: int) -> float:
    """Measure loss restricted to solution spans (Phase 2 format)."""
    total_loss = 0.0
    data_iter = iter(dataloader)
    loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(n_samples):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        spans = batch["problem_spans"]

        with torch.no_grad():
            logits = model(input_ids).logits  # (B, T, V)

        # Compute loss only on solution spans (same as Phase 2 training)
        span_losses = []
        for start, end in spans:
            if end <= start or start < 1:
                continue
            pred = logits[:, start - 1 : end - 1, :].contiguous()
            tgt = input_ids[:, start:end].contiguous()
            span_losses.append(
                loss_fn(pred.view(-1, pred.size(-1)), tgt.view(-1))
            )
        if span_losses:
            total_loss += torch.stack(span_losses).mean().item()

        if (i + 1) % 10 == 0:
            logger.info(
                f"  [{i + 1}/{n_samples}] running avg = {total_loss / (i + 1):.4f}"
            )
    return total_loss / max(n_samples, 1)


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Measure frozen baseline loss")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2, 3],
        help="Which phase's data to measure against (1=C4, 2=QA, 3=domains)",
    )
    parser.add_argument(
        "--samples", type=int, default=100,
        help="Number of batches to average over",
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

    # ---- Phase 1: C4 language modelling ----
    if args.phase == 1:
        logger.info("=== Phase 1 baseline: C4 language modelling ===")
        from nat.training.data import build_phase1_dataloader
        dl = build_phase1_dataloader(config, tokenizer=tokenizer)
        avg = _measure_lm_loss(base_model, dl, device, args.samples)

        logger.info("=" * 60)
        logger.info(f"FROZEN BASELINE — Phase 1 (C4):  {avg:.4f}")
        logger.info(f"  (averaged over {args.samples} batches, seq_len={config.seq_len})")
        logger.info("=" * 60)

    # ---- Phase 2: episodic QA ----
    elif args.phase == 2:
        logger.info("=== Phase 2 baseline: episodic QA (solution-span loss) ===")
        from nat.training.data import build_phase2_dataloader
        dl = build_phase2_dataloader(config, tokenizer=tokenizer, synthetic=False)
        avg = _measure_episodic_loss(base_model, dl, device, args.samples)

        logger.info("=" * 60)
        logger.info(f"FROZEN BASELINE — Phase 2 (QA):  {avg:.4f}")
        logger.info(f"  (averaged over {args.samples} batches, seq_len={config.seq_len})")
        logger.info("=" * 60)

    # ---- Phase 3: per-domain ----
    elif args.phase == 3:
        from nat.training.phase3_consolidation import DOMAINS, build_domain_dataloader

        logger.info("=== Phase 3 baseline: per-domain loss ===")
        results: dict[str, float] = {}

        for domain in DOMAINS:
            logger.info(f"\n--- Domain: {domain} ---")
            try:
                dl = build_domain_dataloader(
                    config, domain, tokenizer=tokenizer, synthetic=False,
                )
            except Exception as e:
                logger.warning(f"  Could not load domain '{domain}': {e}")
                continue

            avg = _measure_lm_loss(
                base_model, dl, device, min(args.samples, 30),
            )
            results[domain] = avg
            logger.info(f"  → {domain}: {avg:.4f}")

        logger.info("\n" + "=" * 60)
        logger.info("FROZEN BASELINE — Phase 3 (per-domain):")
        for d, v in sorted(results.items()):
            logger.info(f"  {d:>15s}:  {v:.4f}")
        overall = sum(results.values()) / max(len(results), 1)
        logger.info(f"  {'AVERAGE':>15s}:  {overall:.4f}")
        logger.info("=" * 60)

    logger.info(
        "\nCompare these numbers to training metrics.\n"
        "  loss < baseline → NAT layers are helping\n"
        "  loss > baseline → NAT layers are hurting (early training, normal)"
    )


if __name__ == "__main__":
    main()
