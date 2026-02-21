"""
Phase 1: Meta-learn the learning rule θ.

What is trained
---------------
θ — the slow parameters of the adaptive memory layers and the
consolidation layer.  These define *how* the model learns at
inference time.

What is NOT trained
-------------------
- Base model weights (frozen).
- Fast weights W (updated by the learned rule, not by the optimiser).

Gradient flow (BPTT through the adaptation chain)
--------------------------------------------------
::

    L_eval → logits → frozen layers → adaptive read →
    fast_W_K → W_{K-1} → … → W_0 → θ

The evaluation loss on the *last* 25 % of the episode back-propagates
through the entire chain of self-modification steps that occurred
during the *first* 75 % (the adaptation phase).  This teaches θ to
produce a learning rule that improves the model's predictions on
novel tokens.

Truncated BPTT
--------------
Full BPTT through many adaptation steps is memory-intensive.  We
support truncated BPTT: every ``config.truncated_bptt`` adaptation
steps, the fast weights are detached from the graph and given fresh
``requires_grad``.  Set ``truncated_bptt = 0`` to disable truncation
(full BPTT).

Usage
-----
::

    from nat.model.nat_model import NATModel
    from nat.training.phase1_meta_learn import train_phase1
    from nat.config import NATConfig

    config = NATConfig.from_yaml("configs/base.yaml")
    model  = NATModel(config)
    train_phase1(model, config)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from nat.training.data import build_phase1_dataloader

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Truncated BPTT helper                                                #
# ------------------------------------------------------------------ #

def _maybe_truncate(model, chunk_idx: int, config) -> None:
    """
    Detach fast weights every ``truncated_bptt`` adaptation chunks
    to bound the length of the BPTT chain.

    After detaching we re-enable ``requires_grad`` so the *next*
    segment of adaptations can still be differentiated.
    """
    tbptt = getattr(config, "truncated_bptt", 0)
    if tbptt <= 0:
        return
    if chunk_idx > 0 and chunk_idx % tbptt == 0:
        for layer in (model.adaptive_A, model.adaptive_B):
            if layer.fast_A is not None:
                layer.fast_A = layer.fast_A.detach().requires_grad_(True)
            if layer.fast_B is not None:
                layer.fast_B = layer.fast_B.detach().requires_grad_(True)


# ------------------------------------------------------------------ #
# Core training step (one episode)                                     #
# ------------------------------------------------------------------ #

def train_one_episode(
    model,
    input_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    config,
    *,
    compute_baseline: bool = False,
) -> dict[str, float]:
    """
    Run one Phase-1 training episode.

    Parameters
    ----------
    model : NATModel
    input_ids : LongTensor, shape ``(batch, seq_len)``
    optimizer : Optimizer over ``model.get_trainable_parameters()``
    config : NATConfig
    compute_baseline : bool
        If True, also compute loss *without* adaptation for comparison.
        Adds overhead — use only for periodic logging.

    Returns
    -------
    dict with keys:
        ``"loss"``              — eval loss after adaptation
        ``"adaptation_benefit"`` — baseline - adapted (only if compute_baseline)
        ``"baseline_loss"``      — loss without adaptation (only if compute_baseline)
        ``"num_adapt_steps"``    — how many adaptation chunks were processed
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # ---- Split: 75 % adaptation, 25 % evaluation ----
    adapt_len = int(seq_len * 0.75)
    # Make sure adapt_len is aligned to adapt_every_n for clean chunking
    chunk_size = config.adapt_every_n
    adapt_len = (adapt_len // chunk_size) * chunk_size
    if adapt_len == 0:
        adapt_len = chunk_size  # at least one chunk

    eval_start = adapt_len

    # ---- Reset fast weights ----
    model.start_session(batch_size)

    # ================================================================
    # ADAPTATION PHASE — process chunks, self-modify fast weights
    # ================================================================
    num_adapt_chunks = 0
    for chunk_start in range(0, adapt_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, adapt_len)
        chunk_ids = input_ids[:, chunk_start:chunk_end]

        # Forward — adaptive layers self-modify (no loss needed here)
        with torch.no_grad() if not model.training else torch.enable_grad():
            _ = model(chunk_ids)

        # Truncated BPTT: periodically detach fast weights
        _maybe_truncate(model, num_adapt_chunks, config)
        num_adapt_chunks += 1

    # ================================================================
    # EVALUATION PHASE — measure quality of adapted fast weights
    # ================================================================
    eval_ids = input_ids[:, eval_start:]
    eval_labels = eval_ids.clone()

    output = model(eval_ids, labels=eval_labels)
    loss = output["loss"]

    # ================================================================
    # (Optional) BASELINE — loss without adaptation
    # ================================================================
    baseline_loss_val = None
    adaptation_benefit = None

    if compute_baseline:
        # Snapshot current adapted fast weights
        saved_A_a = model.adaptive_A.fast_A
        saved_B_a = model.adaptive_A.fast_B
        saved_A_b = model.adaptive_B.fast_A
        saved_B_b = model.adaptive_B.fast_B

        with torch.no_grad():
            model.start_session(batch_size)  # reset to init
            baseline_output = model(eval_ids, labels=eval_labels)
            baseline_loss_val = baseline_output["loss"].item()

        # Restore adapted weights (for correct backward pass)
        model.adaptive_A.fast_A = saved_A_a
        model.adaptive_A.fast_B = saved_B_a
        model.adaptive_B.fast_A = saved_A_b
        model.adaptive_B.fast_B = saved_B_b

        adaptation_benefit = baseline_loss_val - loss.item()

    # ================================================================
    # BACKWARD + OPTIMISER STEP
    # ================================================================
    optimizer.zero_grad()
    loss.backward()

    grad_clip = getattr(config, "grad_clip", 1.0)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(
            model.get_trainable_parameters(), max_norm=grad_clip
        )

    optimizer.step()

    # ---- Collect metrics ----
    metrics: dict[str, float] = {
        "loss": loss.item(),
        "num_adapt_steps": num_adapt_chunks,
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
    dataloader=None,
    use_wandb: bool = False,
    synthetic: bool = False,
) -> dict[str, Any]:
    """
    Full Phase 1 meta-learning loop.

    Parameters
    ----------
    model : NATModel
        The NAT model (base frozen, adaptive layers trainable).
    config : NATConfig
        Must include ``lr``, ``weight_decay``, ``num_episodes``,
        ``log_every``, ``save_every``, ``save_path``, ``grad_clip``,
        ``adapt_every_n``, ``truncated_bptt``, ``device``.
    dataloader : DataLoader, optional
        If not provided, one is built from ``config`` (and ``model.tokenizer``).
    use_wandb : bool
        Whether to log to Weights & Biases.
    synthetic : bool
        If True and no ``dataloader`` is given, use synthetic data.

    Returns
    -------
    dict with ``"final_loss"``, ``"num_episodes_run"``, ``"save_path"``.
    """
    device = torch.device(config.device)
    model = model.to(device)
    model.train()

    # ---- Optimizer & scheduler ----
    trainable = model.get_trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.lr,
        weight_decay=getattr(config, "weight_decay", 0.01),
    )
    num_episodes = config.num_episodes
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_episodes
    )

    # ---- Data ----
    if dataloader is None:
        tokenizer = getattr(model, "tokenizer", None)
        dataloader = build_phase1_dataloader(
            config,
            tokenizer=tokenizer,
            synthetic=synthetic,
        )

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
                    logger.info(f"W&B run: {wandb.run.get_url()}")
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging")
            use_wandb = False

    # ---- Logging ----
    log_every = getattr(config, "log_every", 50)
    save_every = getattr(config, "save_every", 1000)
    save_path = getattr(config, "save_path", "checkpoints/phase1.pt")

    logger.info(
        f"Phase 1 training — {num_episodes} episodes, "
        f"lr={config.lr}, batch={config.batch_size}, "
        f"seq_len={config.seq_len}, device={device}"
    )

    # ---- Training loop ----
    episode_idx = 0
    running_loss = 0.0
    t0 = time.time()
    final_loss = float("inf")

    data_iter = iter(dataloader)

    while episode_idx < num_episodes:
        # Fetch next batch (restart iterator if exhausted)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)

        # Compute baseline only at logging steps (saves compute)
        # +1 because episode_idx is incremented after train_one_episode
        do_baseline = ((episode_idx + 1) % log_every == 0)

        metrics = train_one_episode(
            model, input_ids, optimizer, config,
            compute_baseline=do_baseline,
        )
        scheduler.step()

        running_loss += metrics["loss"]
        final_loss = metrics["loss"]
        episode_idx += 1

        # ---- Periodic logging ----
        if episode_idx % log_every == 0:
            avg_loss = running_loss / log_every
            elapsed = time.time() - t0
            eps_per_sec = episode_idx / elapsed if elapsed > 0 else 0

            log_msg = (
                f"[Episode {episode_idx}/{num_episodes}]  "
                f"loss={avg_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"eps/s={eps_per_sec:.1f}"
            )
            if do_baseline and "adaptation_benefit" in metrics:
                log_msg += (
                    f"  baseline={metrics['baseline_loss']:.4f}"
                    f"  benefit={metrics['adaptation_benefit']:.4f}"
                )
            logger.info(log_msg)

            # Diagnostics
            diag = model.diagnostics()

            if use_wandb:
                import wandb
                log_dict = {
                    "episode": episode_idx,
                    "loss": avg_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "eps_per_sec": eps_per_sec,
                }
                if "adaptation_benefit" in metrics:
                    log_dict["baseline_loss"] = metrics["baseline_loss"]
                    log_dict["adaptation_benefit"] = metrics["adaptation_benefit"]
                log_dict.update(diag)
                wandb.log(log_dict, step=episode_idx)

            running_loss = 0.0

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


# ------------------------------------------------------------------ #
# Checkpoint utilities                                                 #
# ------------------------------------------------------------------ #

def _save_checkpoint(model, path: str, episode_idx: int) -> None:
    """Save trainable parameters to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode_idx,
            "adaptive_A": model.adaptive_A.state_dict(),
            "adaptive_B": model.adaptive_B.state_dict(),
            "consolidation": model.consolidation.state_dict(),
        },
        path,
    )
    logger.info(f"Checkpoint saved → {path}  (episode {episode_idx})")


def load_checkpoint(model, path: str) -> int:
    """
    Load trainable parameters from a Phase 1 checkpoint.

    Returns the episode index stored in the checkpoint.
    """
    state = torch.load(path, weights_only=True, map_location="cpu")
    model.adaptive_A.load_state_dict(state["adaptive_A"])
    model.adaptive_B.load_state_dict(state["adaptive_B"])
    model.consolidation.load_state_dict(state["consolidation"])
    episode = state.get("episode", 0)
    logger.info(f"Checkpoint loaded ← {path}  (episode {episode})")
    return episode
