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

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass, field

import torch
from nat.config import NATConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Statistics container                                                 #
# ------------------------------------------------------------------ #

@dataclass
class LossStats:
    """Collects per-batch losses and computes summary statistics."""
    name: str
    losses: list[float] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def n(self) -> int:
        return len(self.losses)

    @property
    def mean(self) -> float:
        return sum(self.losses) / max(self.n, 1)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        m = self.mean
        return (sum((x - m) ** 2 for x in self.losses) / (self.n - 1)) ** 0.5

    @property
    def median(self) -> float:
        if not self.losses:
            return 0.0
        s = sorted(self.losses)
        mid = self.n // 2
        if self.n % 2 == 0:
            return (s[mid - 1] + s[mid]) / 2
        return s[mid]

    @property
    def min(self) -> float:
        return min(self.losses) if self.losses else 0.0

    @property
    def max(self) -> float:
        return max(self.losses) if self.losses else 0.0

    @property
    def ppl(self) -> float:
        """Perplexity = exp(mean loss), clamped to avoid overflow."""
        return math.exp(min(self.mean, 30.0))

    @property
    def ci95(self) -> float:
        """Approximate 95 % confidence interval half-width."""
        if self.n < 2:
            return 0.0
        return 1.96 * self.std / (self.n ** 0.5)


def _print_stats_table(stats_list: list[LossStats], title: str) -> None:
    """Pretty-print a table of statistics."""
    header = (
        f"{'Name':>18s}  {'N':>5s}  {'Mean':>8s}  {'±95%CI':>8s}  "
        f"{'Std':>8s}  {'Median':>8s}  {'Min':>8s}  {'Max':>8s}  "
        f"{'PPL':>10s}  {'Time':>7s}"
    )
    sep = "─" * len(header)

    lines = [
        "",
        sep,
        f"  {title}",
        sep,
        header,
        sep,
    ]

    for s in stats_list:
        lines.append(
            f"{s.name:>18s}  {s.n:5d}  {s.mean:8.4f}  {s.ci95:8.4f}  "
            f"{s.std:8.4f}  {s.median:8.4f}  {s.min:8.4f}  {s.max:8.4f}  "
            f"{s.ppl:10.2f}  {s.elapsed_s:6.1f}s"
        )

    lines.append(sep)
    print("\n".join(lines))


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _measure_lm_loss(
    model, dataloader, device, n_samples: int, name: str = "lm",
) -> LossStats:
    """Measure next-token loss on a plain language-modelling dataloader."""
    stats = LossStats(name=name)
    t0 = time.time()
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
            stats.losses.append(out.loss.item())

        if (i + 1) % 10 == 0:
            logger.info(
                f"  [{i + 1}/{n_samples}] running avg = {stats.mean:.4f}"
            )

    stats.elapsed_s = time.time() - t0
    return stats


def _measure_episodic_loss(
    model, dataloader, device, n_samples: int, name: str = "episodic",
) -> LossStats:
    """Measure loss restricted to solution spans (Phase 2 format).

    Uses ``labels`` from the batch (padding positions are ``-100``
    and excluded from the loss via ``ignore_index``).
    """
    stats = LossStats(name=name)
    t0 = time.time()
    data_iter = iter(dataloader)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    for i in range(n_samples):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(device)
        else:
            labels = input_ids
        spans = batch["problem_spans"]

        with torch.no_grad():
            logits = model(input_ids).logits  # (B, T, V)

        # Compute loss only on solution spans (same as Phase 2 training)
        span_losses = []
        for start, end in spans:
            if end <= start or start < 1:
                continue
            pred = logits[:, start - 1 : end - 1, :].contiguous()
            tgt = labels[:, start:end].contiguous()
            span_losses.append(
                loss_fn(pred.view(-1, pred.size(-1)), tgt.view(-1))
            )
        if span_losses:
            stats.losses.append(torch.stack(span_losses).mean().item())

        if (i + 1) % 10 == 0:
            logger.info(
                f"  [{i + 1}/{n_samples}] running avg = {stats.mean:.4f}"
            )

    stats.elapsed_s = time.time() - t0
    return stats


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

    total_t0 = time.time()

    # ---- Phase 1: C4 language modelling ----
    if args.phase == 1:
        logger.info("=== Phase 1 baseline: C4 language modelling ===")
        from nat.training.data import build_phase1_dataloader
        dl = build_phase1_dataloader(config, tokenizer=tokenizer)
        stats = _measure_lm_loss(
            base_model, dl, device, args.samples, name="C4 (Phase 1)",
        )
        _print_stats_table([stats], "FROZEN BASELINE — Phase 1 (C4)")

    # ---- Phase 2: episodic QA ----
    elif args.phase == 2:
        logger.info("=== Phase 2 baseline: episodic QA (solution-span loss) ===")
        from nat.training.data import build_phase2_dataloader
        dl = build_phase2_dataloader(config, tokenizer=tokenizer, synthetic=False)
        stats = _measure_episodic_loss(
            base_model, dl, device, args.samples, name="QA (Phase 2)",
        )
        _print_stats_table([stats], "FROZEN BASELINE — Phase 2 (Episodic QA)")

    # ---- Phase 3: per-domain ----
    elif args.phase == 3:
        from nat.training.phase3_consolidation import DOMAINS, build_domain_dataloader

        logger.info("=== Phase 3 baseline: per-domain loss ===")
        all_stats: list[LossStats] = []

        for domain in DOMAINS:
            logger.info(f"\n--- Domain: {domain} ---")
            try:
                dl = build_domain_dataloader(
                    config, domain, tokenizer=tokenizer, synthetic=False,
                )
            except Exception as e:
                logger.warning(f"  Could not load domain '{domain}': {e}")
                continue

            if dl is None:
                logger.warning(f"  Domain '{domain}' returned None")
                continue

            stats = _measure_lm_loss(
                base_model, dl, device, min(args.samples, 30), name=domain,
            )
            all_stats.append(stats)
            logger.info(f"  → {domain}: {stats.mean:.4f}")

        if all_stats:
            # Add an aggregate row
            aggregate = LossStats(name="── AVERAGE ──")
            aggregate.losses = [s.mean for s in all_stats]
            aggregate.elapsed_s = sum(s.elapsed_s for s in all_stats)

            _print_stats_table(
                all_stats + [aggregate],
                "FROZEN BASELINE — Phase 3 (Per-Domain)",
            )

    total_elapsed = time.time() - total_t0
    print(f"\nTotal wall time: {total_elapsed:.1f}s")
    print(
        "\nCompare these numbers to training metrics.\n"
        "  loss < baseline → NAT layers are helping\n"
        "  loss > baseline → NAT layers are hurting (early training, normal)"
    )


if __name__ == "__main__":
    main()
