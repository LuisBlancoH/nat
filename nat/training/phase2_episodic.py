"""
Phase 2: Episodic multi-task training.

What changes from Phase 1
-------------------------
- Data is *structured*: each episode is a sequence of related
  problems of increasing difficulty.
- The loss measures per-problem performance and includes an
  **improvement bonus** that rewards the model for doing better on
  later problems (evidence that the adaptive layers are learning
  within the episode).

What is trained
---------------
Same as Phase 1: θ (slow parameters of adaptive + consolidation).

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

    from nat.training.phase2_episodic import train_phase2
    train_phase2(model, config)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from nat.training.data import build_phase2_dataloader
from nat.training.phase1_meta_learn import (
    _maybe_truncate,
    _save_checkpoint,
    load_checkpoint,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Per-problem loss with improvement bonus                              #
# ------------------------------------------------------------------ #

def compute_episodic_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    problem_spans: list[tuple[int, int]],
    improvement_weight: float = 0.1,
) -> tuple[torch.Tensor, list[float], torch.Tensor]:
    """
    Compute the episodic loss with improvement bonus.

    Parameters
    ----------
    logits : Tensor, shape ``(batch, seq_len, vocab_size)``
    input_ids : LongTensor, shape ``(batch, seq_len)``
        Used as labels (next-token prediction).
    problem_spans : list[(sol_start, sol_end)]
        Token index ranges for each solution.  Loss is computed only
        over solution tokens.
    improvement_weight : float
        Coefficient for the improvement bonus term.

    Returns
    -------
    total_loss : scalar Tensor
    per_problem_losses : list[float]  (detached, for logging)
    improvement : scalar Tensor
    """
    problem_losses: list[torch.Tensor] = []

    for sol_start, sol_end in problem_spans:
        # Shift: logits at position t predict token at position t+1
        sol_logits = logits[:, sol_start - 1 : sol_end - 1, :]
        sol_labels = input_ids[:, sol_start:sol_end]

        if sol_logits.numel() == 0 or sol_labels.numel() == 0:
            continue

        loss_i = F.cross_entropy(
            sol_logits.reshape(-1, sol_logits.size(-1)),
            sol_labels.reshape(-1),
            ignore_index=-100,
        )
        problem_losses.append(loss_i)

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

def train_one_episodic_step(
    model,
    input_ids: torch.Tensor,
    problem_spans: list[tuple[int, int]],
    optimizer: torch.optim.Optimizer,
    config,
) -> dict[str, Any]:
    """
    One Phase-2 training step (one episode).

    Unlike Phase 1, here we do NOT split adapt/eval — the entire
    episode is both adaptation *and* evaluation.  The model adapts
    while processing problems 1..K, and we measure per-problem loss
    to see if it improves.

    Parameters
    ----------
    model : NATModel
    input_ids : LongTensor ``(batch, seq_len)``
    problem_spans : list[(sol_start, sol_end)]
    optimizer : Optimizer on ``model.get_trainable_parameters()``
    config : NATConfig

    Returns
    -------
    dict with ``"loss"``, ``"per_problem_losses"``, ``"improvement"``,
    ``"num_problems"``.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # ---- Reset fast weights for the episode ----
    model.start_session(batch_size)

    # ---- Forward through the entire episode ----
    # Process in chunks so adaptation fires at the right cadence.
    # We accumulate hidden states via multiple forward calls and the
    # fast-weight self-modification graph stays connected.
    chunk_size = config.adapt_every_n
    num_chunks = 0

    # We need logits for the WHOLE sequence to compute per-problem loss.
    # Process chunk-by-chunk, collect logits, then compute loss once.
    all_logits_chunks: list[torch.Tensor] = []

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        chunk_ids = input_ids[:, chunk_start:chunk_end]

        output = model(chunk_ids)
        all_logits_chunks.append(output["logits"])

        _maybe_truncate(model, num_chunks, config)
        num_chunks += 1

    # Reconstruct full logits
    all_logits = torch.cat(all_logits_chunks, dim=1)  # (batch, seq_len, vocab)

    # ---- Compute episodic loss ----
    improvement_weight = getattr(config, "improvement_weight", 0.1)
    total_loss, per_problem_losses, improvement = compute_episodic_loss(
        all_logits, input_ids, problem_spans, improvement_weight
    )

    # ---- Backward + optimiser ----
    optimizer.zero_grad()
    total_loss.backward()

    grad_clip = getattr(config, "grad_clip", 1.0)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            model.get_trainable_parameters(), max_norm=grad_clip
        )
    optimizer.step()

    return {
        "loss": total_loss.item(),
        "per_problem_losses": per_problem_losses,
        "improvement": improvement.item(),
        "num_problems": len(per_problem_losses),
    }


# ------------------------------------------------------------------ #
# Full Phase 2 training loop                                           #
# ------------------------------------------------------------------ #

def train_phase2(
    model,
    config,
    *,
    dataloader: DataLoader | None = None,
    use_wandb: bool = False,
    synthetic: bool = False,
) -> dict[str, Any]:
    """
    Full Phase 2 episodic training loop.

    Parameters
    ----------
    model : NATModel
    config : NATConfig
        Key fields: ``lr_phase2``, ``num_episodes_p2``,
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
    lr = getattr(config, "lr_phase2", 3e-4)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=getattr(config, "weight_decay", 0.01),
    )
    num_episodes = getattr(config, "num_episodes_p2", 30000)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_episodes
    )

    # ---- Data ----
    if dataloader is None:
        tokenizer = getattr(model, "tokenizer", None)
        dataloader = build_phase2_dataloader(
            config,
            tokenizer=tokenizer,
            synthetic=synthetic,
        )

    # ---- W&B ----
    if use_wandb:
        try:
            import wandb
            if not wandb.run:
                wandb.init(
                    project=config.wandb_project,
                    entity=getattr(config, "wandb_entity", None),
                    config=config.to_dict(),
                )
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging")
            use_wandb = False

    # ---- Logging config ----
    log_every = getattr(config, "log_every", 50)
    save_every = getattr(config, "save_every", 1000)
    save_path = getattr(config, "save_path", "checkpoints/phase2.pt")

    logger.info(
        f"Phase 2 episodic training — {num_episodes} episodes, "
        f"lr={lr}, device={device}"
    )

    # ---- Training loop ----
    episode_idx = 0
    running_loss = 0.0
    running_improvement = 0.0
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

        metrics = train_one_episodic_step(
            model, input_ids, problem_spans, optimizer, config,
        )
        scheduler.step()

        running_loss += metrics["loss"]
        running_improvement += metrics["improvement"]
        final_loss = metrics["loss"]
        episode_idx += 1

        # ---- Periodic logging ----
        if episode_idx % log_every == 0:
            avg_loss = running_loss / log_every
            avg_impr = running_improvement / log_every
            elapsed = time.time() - t0
            eps_per_sec = episode_idx / elapsed if elapsed > 0 else 0

            per_prob = metrics["per_problem_losses"]
            prob_str = " → ".join(f"{l:.3f}" for l in per_prob) if per_prob else "n/a"

            logger.info(
                f"[Episode {episode_idx}/{num_episodes}]  "
                f"loss={avg_loss:.4f}  improvement={avg_impr:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"eps/s={eps_per_sec:.1f}  "
                f"per-problem: [{prob_str}]"
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
                log_dict.update(model.diagnostics())
                wandb.log(log_dict, step=episode_idx)

            running_loss = 0.0
            running_improvement = 0.0

        # ---- Periodic checkpoint ----
        if episode_idx % save_every == 0:
            _save_checkpoint(model, save_path, episode_idx)

    # ---- Final save ----
    _save_checkpoint(model, save_path, episode_idx)

    elapsed = time.time() - t0
    logger.info(
        f"Phase 2 complete — {episode_idx} episodes in {elapsed:.0f}s, "
        f"final loss={final_loss:.4f}"
    )

    return {
        "final_loss": final_loss,
        "num_episodes_run": episode_idx,
        "save_path": save_path,
    }
