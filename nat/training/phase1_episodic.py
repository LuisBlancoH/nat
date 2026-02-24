"""
Phase 1: Episodic multi-domain meta-learning.

Trains θ (slow parameters of adaptive + consolidation layers) on
episodes of related problems from multiple domains (math, code,
logic, reading, science).

Episode structure
-----------------
Each episode concatenates ``num_problems`` problem–solution pairs::

    Problem 1: …\\nSolution: …\\n\\n
    Problem 2: …\\nSolution: …\\n\\n
    …
    Problem N: …\\nSolution: …

The model processes the full sequence in one forward pass.
``problem_spans`` records where each solution starts/ends in token
indices so we can compute per-problem loss.

Loss
----
::

    total_loss = base_loss − improvement_weight × improvement_bonus

Where::

    base_loss        = mean of per-problem solution losses
    improvement_bonus = mean of max(0, loss_{i-1} − loss_i) for i=1..N

The improvement bonus is differentiable (uses ``torch.relu``) so
gradients flow to θ.

Usage
-----
::

    from nat.training.phase1_episodic import train_phase1
    train_phase1(model, config)
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nat.training.data import build_phase1_dataloader
from nat.training.train_utils import (
    maybe_truncate as _maybe_truncate,
    save_checkpoint as _save_checkpoint,
    load_checkpoint,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Per-problem loss with improvement bonus                              #
# ------------------------------------------------------------------ #

def compute_episodic_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    problem_spans: list[tuple[int, int]] | list[list[tuple[int, int]]],
    improvement_weight: float = 0.1,
    labels: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[float], torch.Tensor]:
    """
    Compute the episodic loss with improvement bonus.

    Parameters
    ----------
    logits : Tensor, shape ``(batch, seq_len, vocab_size)``
    input_ids : LongTensor, shape ``(batch, seq_len)``
        Used as labels (next-token prediction) when ``labels`` is None.
    problem_spans : list[(sol_start, sol_end)] or list[list[(sol_start, sol_end)]]
        Token index ranges for each solution.  Loss is computed only
        over solution tokens.  Accepts either shared spans (flat list)
        or per-example spans (list of lists).
    improvement_weight : float
        Coefficient for the improvement bonus term.
    labels : LongTensor, shape ``(batch, seq_len)``, optional
        If provided, used instead of ``input_ids`` as targets.
        Positions set to ``-100`` are excluded from the loss.

    Returns
    -------
    total_loss : scalar Tensor
    per_problem_losses : list[float]  (detached, for logging)
    improvement : scalar Tensor
    """
    targets = labels if labels is not None else input_ids
    batch_size = logits.shape[0]

    # Normalise to per-example spans (list of lists)
    if problem_spans and isinstance(problem_spans[0], tuple):
        per_example_spans = [problem_spans] * batch_size  # type: ignore[list-item]
    else:
        per_example_spans = problem_spans  # type: ignore[assignment]

    # Determine number of problems (max across batch)
    max_problems = max(
        (len(spans) for spans in per_example_spans), default=0,
    )
    if max_problems == 0:
        zero = torch.tensor(0.0, device=logits.device, requires_grad=True)
        return zero, [], zero

    # Compute per-problem loss (averaged across batch elements)
    problem_losses: list[torch.Tensor] = []
    for prob_idx in range(max_problems):
        batch_losses: list[torch.Tensor] = []
        for b in range(batch_size):
            if prob_idx >= len(per_example_spans[b]):
                continue
            sol_start, sol_end = per_example_spans[b][prob_idx]
            sol_logits = logits[b, sol_start - 1 : sol_end - 1, :]
            sol_labels = targets[b, sol_start:sol_end]

            if sol_logits.numel() == 0 or sol_labels.numel() == 0:
                continue

            loss_i = F.cross_entropy(
                sol_logits.reshape(-1, sol_logits.size(-1)),
                sol_labels.reshape(-1),
                ignore_index=-100,
            )
            batch_losses.append(loss_i)

        if batch_losses:
            problem_losses.append(torch.stack(batch_losses).mean())

    if len(problem_losses) == 0:
        zero = torch.tensor(0.0, device=logits.device, requires_grad=True)
        return zero, [], zero

    base_loss = torch.stack(problem_losses).mean()

    # Improvement bonus: reward decreasing loss across successive problems
    improvement = torch.tensor(0.0, device=logits.device)
    for i in range(1, len(problem_losses)):
        improvement = improvement + torch.relu(
            problem_losses[i - 1] - problem_losses[i]
        )
    if len(problem_losses) > 1:
        improvement = improvement / (len(problem_losses) - 1)

    total_loss = base_loss - improvement_weight * improvement

    per_problem_floats = [l.item() for l in problem_losses]
    return total_loss, per_problem_floats, improvement


# ------------------------------------------------------------------ #
# Single episodic training step                                        #
# ------------------------------------------------------------------ #
# Validation helper                                                    #
# ------------------------------------------------------------------ #

def _run_validation(
    model,
    val_dataloader: DataLoader,
    config,
    device: torch.device,
    max_batches: int = 0,
) -> dict[str, float]:
    """Run a no-grad validation pass over ``val_dataloader``.

    Returns averaged metrics: ``val_loss``, ``val_improvement``,
    ``val_adaptation_benefit``, ``val_baseline_loss``.

    If ``max_batches`` is 0 the entire loader is consumed.
    """
    model.eval()
    chunk_size = config.adapt_every_n
    improvement_weight = getattr(config, "improvement_weight", 0.1)
    adapt_problems = getattr(config, "adapt_problems_p1", 5)

    total_loss = 0.0
    total_improvement = 0.0
    total_baseline = 0.0
    total_benefit = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_dataloader:
            if max_batches and n_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.to(device)
            problem_spans = batch["problem_spans"]

            batch_size, seq_len = input_ids.shape

            # Normalise spans
            if problem_spans and isinstance(problem_spans[0], tuple):
                per_example_spans = [problem_spans] * batch_size
            else:
                per_example_spans = problem_spans

            num_problems = max(len(s) for s in per_example_spans)
            ap = min(adapt_problems, num_problems - 1)
            eval_spans_raw = [spans[ap:] for spans in per_example_spans]

            eval_start_token = seq_len
            for spans in eval_spans_raw:
                if spans:
                    eval_start_token = min(eval_start_token, spans[0][0] - 1)
            eval_logits_start_chunk = max(0, eval_start_token // chunk_size)
            logits_offset = eval_logits_start_chunk * chunk_size

            # ---- Adapted forward ----
            model.start_session(batch_size)
            adapted_chunks: list[torch.Tensor] = []
            nc = 0
            for cs in range(0, seq_len, chunk_size):
                ce = min(cs + chunk_size, seq_len)
                pos_ids = torch.arange(cs, ce, device=device).unsqueeze(0).expand(batch_size, -1)
                out = model(input_ids[:, cs:ce], position_ids=pos_ids)
                if nc >= eval_logits_start_chunk:
                    adapted_chunks.append(out["logits"])
                _maybe_truncate(model, nc, config)
                nc += 1
            adapted_logits = torch.cat(adapted_chunks, dim=1)

            eval_spans = [
                [(s - logits_offset, e - logits_offset) for s, e in spans]
                for spans in eval_spans_raw
            ]
            loss_t, per_prob, impr_t = compute_episodic_loss(
                adapted_logits, input_ids[:, logits_offset:],
                eval_spans, improvement_weight,
                labels=labels[:, logits_offset:] if labels is not None else None,
            )

            # ---- Baseline (no adaptation) ----
            model.start_session(batch_size)
            baseline_chunks: list[torch.Tensor] = []
            bnc = 0
            for cs in range(0, seq_len, chunk_size):
                ce = min(cs + chunk_size, seq_len)
                pos_ids = torch.arange(cs, ce, device=device).unsqueeze(0).expand(batch_size, -1)
                out = model(input_ids[:, cs:ce], position_ids=pos_ids, suppress_adapt=True)
                if bnc >= eval_logits_start_chunk:
                    baseline_chunks.append(out["logits"])
                bnc += 1
            baseline_logits = torch.cat(baseline_chunks, dim=1)
            _, bl_per_prob, _ = compute_episodic_loss(
                baseline_logits, input_ids[:, logits_offset:],
                eval_spans, 0.0,
                labels=labels[:, logits_offset:] if labels is not None else None,
            )

            total_loss += loss_t.item()
            total_improvement += impr_t.item()
            bl_val = sum(bl_per_prob) / len(bl_per_prob) if bl_per_prob else 0.0
            total_baseline += bl_val
            total_benefit += (bl_val - loss_t.item())
            n_batches += 1

    model.train()

    if n_batches == 0:
        return {}

    return {
        "val_loss": total_loss / n_batches,
        "val_improvement": total_improvement / n_batches,
        "val_baseline_loss": total_baseline / n_batches,
        "val_adaptation_benefit": total_benefit / n_batches,
    }


# ------------------------------------------------------------------ #
# Single episodic training step                                        #
# ------------------------------------------------------------------ #

def train_one_episodic_step(
    model,
    input_ids: torch.Tensor,
    problem_spans: list[tuple[int, int]] | list[list[tuple[int, int]]],
    optimizer: torch.optim.Optimizer,
    config,
    labels: torch.Tensor | None = None,
    compute_baseline: bool = False,
) -> dict[str, Any]:
    """
    One Phase-2 training step (one episode).

    The episode is split into **adapt** and **eval** portions at the
    problem level:

    - **Adapt** (first ``adapt_problems`` problems): the model forwards
      these tokens chunk-by-chunk so adaptation fires and fast weights
      update, but no loss is computed.
    - **Eval** (remaining problems): the model continues to forward
      chunk-by-chunk (adaptation still fires), and logits are collected
      to compute the episodic loss with improvement bonus.

    This mirrors Phase 1's 75/25 adapt/eval split: the gradient
    exclusively rewards *adapted* performance, giving a cleaner
    meta-learning signal.

    Parameters
    ----------
    model : NATModel
    input_ids : LongTensor ``(batch, seq_len)``
    problem_spans : list[(sol_start, sol_end)] or list[list[(sol_start, sol_end)]]
        Per-example problem spans (list of lists) or shared spans
        (flat list, broadcast to all batch elements).
    optimizer : Optimizer on ``model.get_trainable_parameters()``
    config : NATConfig
    labels : LongTensor ``(batch, seq_len)``, optional
        If provided, positions set to ``-100`` are excluded from loss.
    compute_baseline : bool
        If True, compute baseline loss (no adaptation) for diagnostics.

    Returns
    -------
    dict with ``"loss"``, ``"per_problem_losses"``, ``"improvement"``,
    ``"num_problems"``, and optionally ``"baseline_loss"`` and
    ``"adaptation_benefit"``.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # ---- Normalise spans to per-example format ----
    if problem_spans and isinstance(problem_spans[0], tuple):
        per_example_spans: list[list[tuple[int, int]]] = [
            problem_spans  # type: ignore[list-item]
        ] * batch_size
    else:
        per_example_spans = problem_spans  # type: ignore[assignment]

    num_problems = max(len(s) for s in per_example_spans)
    adapt_problems = getattr(config, "adapt_problems_p1", num_problems * 5 // 8)
    adapt_problems = min(adapt_problems, num_problems - 1)  # ≥1 eval problem

    # ---- Determine which chunks need logits ----
    # Only keep logits for chunks overlapping eval problem spans.
    # Adapt-only chunks still fire adaptation (updating fast weights)
    # but their logits are discarded to save GPU memory.
    eval_spans_raw = [spans[adapt_problems:] for spans in per_example_spans]

    eval_start_token = seq_len
    for spans in eval_spans_raw:
        if spans:
            # Need logits at sol_start - 1 for the shifted CE loss
            eval_start_token = min(eval_start_token, spans[0][0] - 1)

    # ---- Reset fast weights for the episode ----
    model.start_session(batch_size)

    # ---- Forward through the entire episode ----
    chunk_size = config.adapt_every_n
    num_chunks = 0

    # Align to chunk boundary
    eval_logits_start_chunk = max(0, eval_start_token // chunk_size)
    logits_offset = eval_logits_start_chunk * chunk_size

    all_logits_chunks: list[torch.Tensor] = []

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        chunk_ids = input_ids[:, chunk_start:chunk_end]

        # Absolute position IDs so RoPE embeddings are correct
        # even though attention is chunk-local.
        pos_ids = torch.arange(
            chunk_start, chunk_end, device=device,
        ).unsqueeze(0).expand(batch_size, -1)

        output = model(chunk_ids, position_ids=pos_ids)

        # Only keep logits that overlap with eval spans
        if num_chunks >= eval_logits_start_chunk:
            all_logits_chunks.append(output["logits"])

        _maybe_truncate(model, num_chunks, config)
        num_chunks += 1

    all_logits = torch.cat(all_logits_chunks, dim=1)

    # ---- Compute episodic loss on EVAL problems only ----
    # Adjust span indices to match the truncated logits tensor
    eval_spans = [
        [(s - logits_offset, e - logits_offset) for s, e in spans]
        for spans in eval_spans_raw
    ]

    improvement_weight = getattr(config, "improvement_weight", 0.1)
    total_loss, per_problem_losses, improvement = compute_episodic_loss(
        all_logits, input_ids[:, logits_offset:], eval_spans,
        improvement_weight,
        labels=labels[:, logits_offset:] if labels is not None else None,
    )

    # ---- Baseline: eval loss without adaptation ----
    baseline_loss_val = None
    adaptation_benefit = None

    if compute_baseline:
        saved_A_a = model.adaptive_A.fast_A
        saved_B_a = model.adaptive_A.fast_B
        saved_A_b = model.adaptive_B.fast_A
        saved_B_b = model.adaptive_B.fast_B
        saved_step = model._step_counter
        saved_do_adapt = model._do_adapt
        saved_adapt_cell = model._adapt_cell[0]

        # Free KV cache before baseline to reduce peak memory
        if hasattr(model, '_kv_cache'):
            model._kv_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        with torch.no_grad():
            # Fresh session — reset fast weights + KV cache
            model.start_session(batch_size)

            # Only collect logits for eval portion (same optimisation)
            baseline_chunks: list[torch.Tensor] = []
            bl_num_chunks = 0
            for chunk_start in range(0, seq_len, chunk_size):
                chunk_end = min(chunk_start + chunk_size, seq_len)
                chunk_ids = input_ids[:, chunk_start:chunk_end]
                pos_ids = torch.arange(
                    chunk_start, chunk_end, device=device,
                ).unsqueeze(0).expand(batch_size, -1)
                out = model(chunk_ids, position_ids=pos_ids, suppress_adapt=True)
                if bl_num_chunks >= eval_logits_start_chunk:
                    baseline_chunks.append(out["logits"])
                bl_num_chunks += 1

            baseline_logits = torch.cat(baseline_chunks, dim=1)
            _, baseline_per_prob, _ = compute_episodic_loss(
                baseline_logits, input_ids[:, logits_offset:],
                eval_spans, 0.0,
                labels=labels[:, logits_offset:] if labels is not None else None,
            )
            del baseline_logits, baseline_chunks
            if baseline_per_prob:
                baseline_loss_val = sum(baseline_per_prob) / len(baseline_per_prob)

        # Restore adapted state
        model.adaptive_A.fast_A = saved_A_a
        model.adaptive_A.fast_B = saved_B_a
        model.adaptive_B.fast_A = saved_A_b
        model.adaptive_B.fast_B = saved_B_b
        model._step_counter = saved_step
        model._do_adapt = saved_do_adapt
        model._adapt_cell[0] = saved_adapt_cell

        if baseline_loss_val is not None:
            adaptation_benefit = baseline_loss_val - total_loss.item()

    # ---- Backward + optimiser ----
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        return {
            "loss": float("nan"),
            "per_problem_losses": per_problem_losses,
            "improvement": float("nan"),
            "num_problems": len(per_problem_losses),
        }

    optimizer.zero_grad()
    total_loss.backward()

    grad_clip = getattr(config, "grad_clip", 1.0)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            model.get_trainable_parameters(), max_norm=grad_clip
        )
    optimizer.step()

    metrics: dict[str, Any] = {
        "loss": total_loss.item(),
        "per_problem_losses": per_problem_losses,
        "improvement": improvement.item(),
        "num_problems": len(per_problem_losses),
    }
    if compute_baseline:
        metrics["baseline_loss"] = baseline_loss_val
        metrics["adaptation_benefit"] = adaptation_benefit
    return metrics


# ------------------------------------------------------------------ #
# Full Phase 1 training loop                                           #
# ------------------------------------------------------------------ #

def train_phase1(
    model,
    config,
    *,
    dataloader: DataLoader | None = None,
    use_wandb: bool = False,
    synthetic: bool = False,
) -> dict[str, Any]:
    """
    Full Phase 1 episodic training loop.

    Parameters
    ----------
    model : NATModel
    config : NATConfig
        Key fields: ``lr_phase1``, ``num_episodes_p1``,
        ``improvement_weight``, ``num_problems_per_episode``.
    dataloader : DataLoader, optional
    use_wandb : bool
    synthetic : bool
        Use synthetic episodic data for testing.

    Returns
    -------
    dict with ``"final_loss"``, ``"num_episodes_run"``, ``"save_path"``.
    """
    device = torch.device(config.device)
    model = model.to(device)
    model.train()

    # ---- Optimizer & scheduler ----
    trainable = model.get_trainable_parameters()
    lr = getattr(config, "lr_phase1", 2e-4)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=getattr(config, "weight_decay", 0.01),
    )
    num_episodes = getattr(config, "num_episodes_p1", 50000)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_episodes
    )

    # ---- Data ----
    if dataloader is None:
        tokenizer = getattr(model, "tokenizer", None)
        dataloader, val_dataloader = build_phase1_dataloader(
            config,
            tokenizer=tokenizer,
            synthetic=synthetic,
        )
    else:
        val_dataloader = None

    # ---- W&B ----
    if use_wandb:
        try:
            import wandb
            if not wandb.run:
                if wandb.api.api_key is None:
                    raise RuntimeError(
                        "wandb API key not found. Run `wandb login` or "
                        "set WANDB_API_KEY before using --wandb."
                    )
                wandb.init(
                    project=config.wandb_project,
                    entity=getattr(config, "wandb_entity", None),
                    config=config.to_dict(),
                )
                if wandb.run:
                    logger.info(f"W&B run: {wandb.run.url}")
                    wandb.define_metric("episode")
                    wandb.define_metric("*", step_metric="episode")
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging")
            use_wandb = False

    # ---- Logging config ----
    log_every = getattr(config, "log_every", 50)
    save_every = getattr(config, "save_every", 1000)
    save_path = getattr(config, "save_path", "checkpoints/phase1.pt")

    logger.info(
        f"Phase 1 episodic training — {num_episodes} episodes, "
        f"lr={lr}, device={device}"
    )

    # ---- Training loop ----
    episode_idx = 0
    running_loss = 0.0
    running_improvement = 0.0
    valid_in_window = 0
    nan_total = 0
    t0 = time.time()
    final_loss = float("inf")

    data_iter = iter(dataloader)

    while episode_idx < num_episodes:
        # Fetch next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        problem_spans = batch["problem_spans"]
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(device)

        do_baseline = (
            (episode_idx % log_every == 0) or episode_idx <= 10
        )

        metrics = train_one_episodic_step(
            model, input_ids, problem_spans, optimizer, config,
            labels=labels,
            compute_baseline=do_baseline,
        )
        scheduler.step()

        if math.isnan(metrics["loss"]):
            nan_total += 1
            logger.warning(
                f"[Episode {episode_idx + 1}] NaN loss — skipping "
                f"(total NaN: {nan_total})"
            )
            episode_idx += 1
            continue

        running_loss += metrics["loss"]
        running_improvement += metrics["improvement"]
        final_loss = metrics["loss"]
        valid_in_window += 1
        episode_idx += 1

        # ---- Periodic logging ----
        if episode_idx % log_every == 0:
            n = max(valid_in_window, 1)
            avg_loss = running_loss / n
            avg_impr = running_improvement / n
            elapsed = time.time() - t0
            eps_per_sec = episode_idx / elapsed if elapsed > 0 else 0

            per_prob = metrics["per_problem_losses"]
            prob_str = " → ".join(f"{l:.3f}" for l in per_prob) if per_prob else "n/a"

            baseline_str = ""
            if metrics.get("baseline_loss") is not None:
                baseline_str = (
                    f"  baseline={metrics['baseline_loss']:.4f}"
                    f"  benefit={metrics['adaptation_benefit']:.4f}"
                )

            logger.info(
                f"[Episode {episode_idx}/{num_episodes}]  "
                f"loss={avg_loss:.4f}  improvement={avg_impr:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"eps/s={eps_per_sec:.1f}  "
                f"per-problem: [{prob_str}]"
                f"{baseline_str}"
            )

            if use_wandb:
                import wandb
                log_dict = {
                    "episode": episode_idx,
                    "loss": avg_loss,
                    "improvement": avg_impr,
                    "lr": scheduler.get_last_lr()[0],
                    "eps_per_sec": eps_per_sec,
                    "num_problems": metrics["num_problems"],
                }
                # Log individual problem losses
                for i, l in enumerate(per_prob):
                    log_dict[f"problem_{i}_loss"] = l
                if metrics.get("baseline_loss") is not None:
                    log_dict["baseline_loss"] = metrics["baseline_loss"]
                    log_dict["adaptation_benefit"] = metrics["adaptation_benefit"]
                log_dict.update(model.diagnostics())
                wandb.log(log_dict)

            # ---- Validation pass ----
            if val_dataloader is not None:
                val_metrics = _run_validation(
                    model, val_dataloader, config, device,
                    max_batches=20,  # cap to ~20 batches for speed
                )
                if val_metrics:
                    logger.info(
                        f"  [val] loss={val_metrics['val_loss']:.4f}  "
                        f"benefit={val_metrics['val_adaptation_benefit']:.4f}  "
                        f"baseline={val_metrics['val_baseline_loss']:.4f}"
                    )
                    if use_wandb:
                        import wandb
                        wandb.log({"episode": episode_idx, **val_metrics})

            if nan_total > 0:
                logger.info(f"  NaN episodes so far: {nan_total}")

            running_loss = 0.0
            running_improvement = 0.0
            valid_in_window = 0

        # ---- Periodic checkpoint ----
        if episode_idx % save_every == 0:
            _save_checkpoint(model, save_path, episode_idx)

    # ---- Final save ----
    _save_checkpoint(model, save_path, episode_idx)

    elapsed = time.time() - t0
    logger.info(
        f"Phase 1 complete — {episode_idx} episodes in {elapsed:.0f}s, "
        f"final loss={final_loss:.4f}"
    )

    return {
        "final_loss": final_loss,
        "num_episodes_run": episode_idx,
        "save_path": save_path,
    }
