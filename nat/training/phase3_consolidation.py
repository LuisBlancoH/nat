"""
Phase 3: Train consolidation dynamics.

What is trained
---------------
- Consolidation layer parameters (``read_net``, ``gate_net``, ``layer_norm``)
  — via standard forward-pass back-propagation.
- β (EMA decay rate) — via differentiable EMA that keeps β in the
  computation graph.
- α (session reset rate) — via differentiable partial reset that keeps
  α in the computation graph.

What is frozen
--------------
- Base model weights (always frozen).
- Adaptive layer θ (slow params learned in Phases 1-2).

Training procedure
------------------
Each "run" processes a **domain sequence**::

    D1 × N_sessions → D2 × N_sessions → D1 × K_forgetting_test

Within each session the model processes a chunk of domain-specific
tokens.  The adaptive layers self-modify (with frozen θ) and the
consolidation layer's read / gate networks are trained through the
forward-pass loss.

Between sessions:

- **Differentiable consolidation** — the EMA update
  ``W_c ← β·W_c + (1−β)·avg(fast weights)`` is performed *without*
  ``@torch.no_grad()``, so that β participates in the computation
  graph and receives gradients from subsequent sessions.
- **Differentiable partial reset** —
  ``fast_W ← α·W_init + (1−α)·fast_W`` keeps α in the graph.

The consolidated weights are **detached** every
``p3_truncate_sessions`` sessions (truncated cross-session BPTT) to
bound memory usage.

Training signal
---------------
1. **Forward-pass loss** — standard next-token prediction through the
   consolidation layer (trains ``read_net`` / ``gate_net``).
2. **Cross-session improvement** — later sessions in a domain should
   have lower loss than earlier ones.  Measured but not used as a
   direct loss term (the forward-pass loss captures this indirectly
   through the consolidated weights).
3. **Forgetting** — returning to domain D1 after learning D2 should
   not show a large loss increase.  Measured for diagnostics.

Gradient paths
--------------
- **β** receives gradient through the consolidation layer's forward
  pass: ``loss → consolidation.forward(h) → W_c → β``.
- **α** receives gradient through the adaptive layer's read path:
  ``loss → adaptive.read(h) → fast_A → α``.
- **Consolidation read/gate** receive gradient through standard
  forward-pass back-propagation in every session.

Usage
-----
::

    from nat.training.phase3_consolidation import train_phase3
    train_phase3(model, config)
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from nat.training.phase1_meta_learn import _save_checkpoint

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

DOMAINS: list[str] = ["math", "code", "reasoning", "text", "science"]
"""Default domain names for Phase 3 training."""


# ------------------------------------------------------------------ #
# Domain-specific synthetic data                                       #
# ------------------------------------------------------------------ #

class SyntheticDomainDataset(Dataset):
    """
    Domain-specific synthetic data for Phase 3 consolidation training.

    Each domain produces token sequences with a characteristic
    statistical profile: 70 % of tokens are drawn from a
    domain-specific vocabulary slice, 30 % from the full vocabulary.
    This makes sequences from the same domain share enough structure
    for the consolidation layer to exploit, while remaining distinct
    from other domains.

    Parameters
    ----------
    domain : str
        Domain name (must be in ``DOMAINS`` or any hashable string).
    num_episodes : int
        Number of episodes to pre-generate.
    seq_len : int
        Tokens per episode.
    vocab_size : int
        Vocabulary size (shared across domains).
    seed : int
        Base random seed (combined with domain offset for reproducibility).
    """

    def __init__(
        self,
        domain: str,
        num_episodes: int = 64,
        seq_len: int = 256,
        vocab_size: int = 1000,
        seed: int = 42,
    ):
        super().__init__()
        self.domain = domain
        self.num_episodes = num_episodes
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Domain-specific seed → reproducible but distinct distributions
        if domain in DOMAINS:
            domain_offset = DOMAINS.index(domain)
        else:
            domain_offset = hash(domain) % 100

        rng = torch.Generator().manual_seed(seed + domain_offset * 1000)

        # Each domain emphasises a different slice of the vocabulary
        num_domains = max(len(DOMAINS), domain_offset + 1)
        slice_size = vocab_size // num_domains
        domain_start = domain_offset * slice_size
        domain_end = min(domain_start + slice_size, vocab_size)

        # Ensure at least 10 tokens in the domain slice
        domain_end = max(domain_end, domain_start + 10)
        domain_end = min(domain_end, vocab_size)

        self.data: list[torch.Tensor] = []
        num_domain_tokens = int(seq_len * 0.7)
        num_random_tokens = seq_len - num_domain_tokens

        for _ in range(num_episodes):
            domain_tokens = torch.randint(
                domain_start, domain_end,
                (num_domain_tokens,), generator=rng,
            )
            random_tokens = torch.randint(
                0, vocab_size,
                (num_random_tokens,), generator=rng,
            )
            tokens = torch.cat([domain_tokens, random_tokens])
            # Shuffle to interleave domain and random tokens
            perm = torch.randperm(seq_len, generator=rng)
            tokens = tokens[perm]
            self.data.append(tokens)

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.data[idx]}


def build_domain_dataloader(
    config,
    domain: str,
    tokenizer=None,
    *,
    synthetic: bool = True,
) -> DataLoader:
    """
    Build a ``DataLoader`` for a single domain.

    Parameters
    ----------
    config : NATConfig
    domain : str
    tokenizer : optional
    synthetic : bool
        If ``True``, use ``SyntheticDomainDataset``.

    Returns
    -------
    DataLoader
    """
    if synthetic:
        sessions_per_domain = getattr(config, "sessions_per_domain_p3", 20)
        dataset = SyntheticDomainDataset(
            domain=domain,
            num_episodes=max(sessions_per_domain * 4, 64),
            seq_len=config.seq_len,
            vocab_size=getattr(config, "vocab_size", 1000),
        )
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

    # ---- Real data ----
    # Map domain names to HuggingFace datasets:
    #   "math"      → gsm8k
    #   "code"      → openai_humaneval
    #   "reasoning" → allenai/ai2_arc
    #   "text"      → c4 (en)
    #   "science"   → sciq
    # Left as extension point — the synthetic path covers the training
    # loop mechanics; real data wiring is task-dependent.
    assert tokenizer is not None, (
        "tokenizer is required for non-synthetic data. "
        "Pass synthetic=True for testing."
    )
    raise NotImplementedError(
        f"Real domain data loading for '{domain}' not implemented. "
        "Use synthetic=True for testing."
    )


# ------------------------------------------------------------------ #
# Domain sequence builder                                              #
# ------------------------------------------------------------------ #

def build_domain_sequence(
    config,
    domains: list[str] | None = None,
) -> list[str]:
    """
    Build a domain sequence for one Phase 3 run.

    Default structure::

        D1 × N  →  D2 × N  →  D1 × K

    where ``N = sessions_per_domain_p3`` and
    ``K = forgetting_test_sessions_p3``.

    Two domains are sampled randomly from the pool on each call.

    Parameters
    ----------
    config : NATConfig
    domains : list[str], optional
        Domain pool.  Defaults to ``DOMAINS``.

    Returns
    -------
    list[str]
        Flat list of domain names, one per session.
    """
    if domains is None:
        domains = DOMAINS

    sessions_per_domain = getattr(config, "sessions_per_domain_p3", 20)
    forgetting_sessions = getattr(config, "forgetting_test_sessions_p3", 5)

    d1, d2 = random.sample(domains, 2)

    sequence: list[str] = (
        [d1] * sessions_per_domain
        + [d2] * sessions_per_domain
        + [d1] * forgetting_sessions
    )
    return sequence


# ------------------------------------------------------------------ #
# Differentiable consolidation helpers                                 #
# ------------------------------------------------------------------ #

def _consolidate_differentiable(
    model,
    beta: torch.Tensor,
) -> None:
    """
    Differentiable EMA update of consolidated weights.

    Unlike ``ConsolidationLayer.consolidate()`` (``@torch.no_grad``),
    this version keeps ``beta`` in the computation graph so that
    gradients from subsequent forward passes flow back to β.

    Fast weights are **detached** — we do not want gradients flowing
    back through the adaptive layer's adaptation chain (that is
    Phase 1-2's job).
    """
    adaptive_layers = [model.adaptive_A, model.adaptive_B]

    # Average fast weights (detached from adaptive graph)
    avg_A = torch.stack(
        [layer.fast_A.detach().mean(dim=0) for layer in adaptive_layers]
    ).mean(dim=0, keepdim=True)  # (1, d_model, rank)

    avg_B = torch.stack(
        [layer.fast_B.detach().mean(dim=0) for layer in adaptive_layers]
    ).mean(dim=0, keepdim=True)  # (1, rank, d_model)

    # Move to consolidated-weight device/dtype
    avg_A = avg_A.to(model.consolidation.W_c_A.device)
    avg_B = avg_B.to(model.consolidation.W_c_B.device)

    # Differentiable EMA — beta remains in the computation graph
    model.consolidation.W_c_A = (
        beta * model.consolidation.W_c_A + (1 - beta) * avg_A
    )
    model.consolidation.W_c_B = (
        beta * model.consolidation.W_c_B + (1 - beta) * avg_B
    )


def _partial_reset_differentiable(
    model,
    alpha: torch.Tensor,
) -> None:
    """
    Differentiable partial reset of fast weights.

    Keeps ``alpha`` in the computation graph so that gradients from
    subsequent forward passes flow back to α.

    The previous fast weights are **detached** so gradients do not
    propagate through the prior session's adaptation chain.
    """
    for layer in [model.adaptive_A, model.adaptive_B]:
        if layer.fast_A is None:
            continue

        init_A = layer.fast_A_init.detach().unsqueeze(0).expand_as(layer.fast_A)
        init_B = layer.fast_B_init.detach().unsqueeze(0).expand_as(layer.fast_B)

        layer.fast_A = alpha * init_A + (1 - alpha) * layer.fast_A.detach()
        layer.fast_B = alpha * init_B + (1 - alpha) * layer.fast_B.detach()

    # Reset previous hidden state for surprise computation
    for layer in [model.adaptive_A, model.adaptive_B]:
        layer.prev_h = None


def _truncate_consolidated_weights(model) -> None:
    """
    Detach consolidated weights to bound the cross-session BPTT chain.

    Call every ``p3_truncate_sessions`` sessions to keep memory
    usage bounded.
    """
    W_c_A = model.consolidation.W_c_A
    W_c_B = model.consolidation.W_c_B

    if isinstance(W_c_A, torch.Tensor):
        model.consolidation.W_c_A = W_c_A.detach().clone()
        model.consolidation.W_c_B = W_c_B.detach().clone()


# ------------------------------------------------------------------ #
# Consolidation metrics                                                #
# ------------------------------------------------------------------ #

def compute_consolidation_metrics(
    domain_losses: dict[str, list[float]],
    sequence: list[str],
    sessions_per_domain: int,
) -> dict[str, float]:
    """
    Compute cross-session improvement and forgetting metrics.

    Parameters
    ----------
    domain_losses : dict[str, list[float]]
        Per-domain ordered list of session losses.
    sequence : list[str]
        The domain sequence that was run.
    sessions_per_domain : int
        Number of sessions per domain in the first two blocks.

    Returns
    -------
    dict with keys:
        ``cross_session_improvement`` — mean(early D1) − mean(late D1)
        ``forgetting`` — mean(return D1) − mean(late D1)
        ``d1_early_loss``, ``d1_late_loss``, ``d1_return_loss``,
        ``d2_loss``
    """
    # Identify the two domains
    d1 = sequence[0]
    d2 = None
    for d in sequence:
        if d != d1:
            d2 = d
            break

    d1_losses = domain_losses.get(d1, [])
    d2_losses = domain_losses.get(d2, []) if d2 else []

    metrics: dict[str, float] = {}

    # --- Cross-session improvement for D1 (first block) ---
    d1_first_block = d1_losses[:sessions_per_domain]
    if len(d1_first_block) >= 2:
        third = max(1, len(d1_first_block) // 3)
        early = d1_first_block[:third]
        late = d1_first_block[-third:]
        metrics["d1_early_loss"] = sum(early) / len(early)
        metrics["d1_late_loss"] = sum(late) / len(late)
        metrics["cross_session_improvement"] = (
            metrics["d1_early_loss"] - metrics["d1_late_loss"]
        )
    else:
        metrics["d1_early_loss"] = d1_losses[0] if d1_losses else 0.0
        metrics["d1_late_loss"] = d1_losses[-1] if d1_losses else 0.0
        metrics["cross_session_improvement"] = 0.0

    # --- D2 loss ---
    metrics["d2_loss"] = (
        sum(d2_losses) / len(d2_losses) if d2_losses else 0.0
    )

    # --- Forgetting: D1 return loss vs D1 late loss ---
    d1_return = d1_losses[sessions_per_domain:]
    if d1_return and len(d1_first_block) >= 2:
        metrics["d1_return_loss"] = sum(d1_return) / len(d1_return)
        metrics["forgetting"] = (
            metrics["d1_return_loss"] - metrics["d1_late_loss"]
        )
    else:
        metrics["d1_return_loss"] = metrics.get("d1_late_loss", 0.0)
        metrics["forgetting"] = 0.0

    return metrics


# ------------------------------------------------------------------ #
# Single Phase 3 run (multi-session domain sequence)                   #
# ------------------------------------------------------------------ #

def train_one_run(
    model,
    config,
    optimizer: torch.optim.Optimizer,
    beta_logit: nn.Parameter,
    alpha_logit: nn.Parameter,
    domain_dataloaders: dict[str, DataLoader],
    domain_iters: dict[str, Any],
    run_idx: int,
) -> dict[str, Any]:
    """
    Execute one Phase 3 consolidation run.

    Processes a full domain sequence (D1 × N → D2 × N → D1 × K),
    training the consolidation layer, β, and α along the way.

    Parameters
    ----------
    model : NATModel
    config : NATConfig
    optimizer : Optimizer
    beta_logit : nn.Parameter
        Unconstrained logit for β (``β = sigmoid(beta_logit)``).
    alpha_logit : nn.Parameter
        Unconstrained logit for α (``α = sigmoid(alpha_logit)``).
    domain_dataloaders : dict[str, DataLoader]
    domain_iters : dict[str, iterator]
        Mutable iterator cache (refreshed on ``StopIteration``).
    run_idx : int
        Current run index (for logging).

    Returns
    -------
    dict with ``"mean_loss"``, ``"cross_session_improvement"``,
    ``"forgetting"``, ``"beta"``, ``"alpha"``, and per-domain metrics.
    """
    batch_size = config.batch_size
    device = torch.device(config.device)
    sessions_per_domain = getattr(config, "sessions_per_domain_p3", 20)
    truncate_every = getattr(config, "p3_truncate_sessions", 4)
    chunk_size = config.adapt_every_n
    grad_clip = getattr(config, "grad_clip", 1.0)

    # Build domain sequence for this run
    sequence = build_domain_sequence(config)

    # Reset consolidated weights to zero
    model.consolidation.W_c_A = torch.zeros_like(model.consolidation.W_c_A)
    model.consolidation.W_c_B = torch.zeros_like(model.consolidation.W_c_B)

    # Track per-domain losses
    domain_losses: dict[str, list[float]] = defaultdict(list)

    # Loss accumulator for batched backprop
    accumulated_loss = torch.tensor(0.0, device=device, requires_grad=True)
    num_accumulated = 0

    for session_idx, domain in enumerate(sequence):
        # ---- Current beta and alpha (differentiable) ----
        beta = torch.sigmoid(beta_logit)
        alpha = torch.sigmoid(alpha_logit)

        # ---- Start session ----
        if session_idx == 0:
            # First session: full reset to learned initial values
            model.adaptive_A.reset_fast_weights(batch_size)
            model.adaptive_B.reset_fast_weights(batch_size)
        else:
            # Subsequent sessions: differentiable partial reset
            # (preserves some knowledge from the previous session)
            _partial_reset_differentiable(model, alpha)

        model._step_counter = 0

        # ---- Fetch domain data ----
        if domain not in domain_iters:
            domain_iters[domain] = iter(domain_dataloaders[domain])
        try:
            batch = next(domain_iters[domain])
        except StopIteration:
            domain_iters[domain] = iter(domain_dataloaders[domain])
            batch = next(domain_iters[domain])

        input_ids = batch["input_ids"].to(device)
        seq_len = input_ids.shape[1]

        # ---- Forward pass in chunks (adaptation fires at correct cadence) ----
        session_logits: list[torch.Tensor] = []
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_ids = input_ids[:, chunk_start:chunk_end]
            output = model(chunk_ids)
            session_logits.append(output["logits"])

        all_logits = torch.cat(session_logits, dim=1)

        # ---- Compute session loss ----
        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        session_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        domain_losses[domain].append(session_loss.item())
        accumulated_loss = accumulated_loss + session_loss
        num_accumulated += 1

        # ---- End session: differentiable consolidation ----
        _consolidate_differentiable(model, beta)

        # ---- Periodic backprop + truncation ----
        if num_accumulated >= truncate_every or session_idx == len(sequence) - 1:
            optimizer.zero_grad()
            avg_loss = accumulated_loss / num_accumulated
            avg_loss.backward()

            if grad_clip > 0:
                all_params = (
                    list(model.consolidation.parameters())
                    + [beta_logit, alpha_logit]
                )
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip)

            optimizer.step()

            # Reset accumulator (new graph segment)
            accumulated_loss = torch.tensor(
                0.0, device=device, requires_grad=True
            )
            num_accumulated = 0

            # Truncate consolidated weights to bound BPTT chain
            _truncate_consolidated_weights(model)

    # ---- Compute run-level metrics ----
    run_metrics: dict[str, Any] = dict(compute_consolidation_metrics(
        domain_losses, sequence, sessions_per_domain,
    ))
    total_losses = [
        l for losses in domain_losses.values() for l in losses
    ]
    run_metrics["mean_loss"] = (
        sum(total_losses) / len(total_losses) if total_losses else 0.0
    )
    run_metrics["beta"] = torch.sigmoid(beta_logit).item()
    run_metrics["alpha"] = torch.sigmoid(alpha_logit).item()
    run_metrics["num_sessions"] = len(sequence)
    run_metrics["d1_name"] = sequence[0]
    run_metrics["d2_name"] = next((d for d in sequence if d != sequence[0]), "n/a")

    return run_metrics


# ------------------------------------------------------------------ #
# Full Phase 3 training loop                                           #
# ------------------------------------------------------------------ #

def train_phase3(
    model,
    config,
    *,
    use_wandb: bool = False,
    synthetic: bool = False,
) -> dict[str, Any]:
    """
    Full Phase 3 consolidation training loop.

    Parameters
    ----------
    model : NATModel
        Must have trained adaptive θ (from Phase 1-2).
    config : NATConfig
        Key fields: ``lr_phase3``, ``num_runs_p3``,
        ``sessions_per_domain_p3``, ``forgetting_test_sessions_p3``,
        ``p3_truncate_sessions``, ``beta``, ``session_reset_alpha``.
    use_wandb : bool
        Enable Weights & Biases logging.
    synthetic : bool
        Use synthetic domain data (for testing / development).

    Returns
    -------
    dict with ``"final_beta"``, ``"final_alpha"``,
    ``"best_improvement"``, ``"num_runs"``, ``"save_path"``.
    """
    device = torch.device(config.device)
    model = model.to(device)
    model.train()

    # ================================================================
    # Freeze adaptive θ — only consolidation is trained in Phase 3
    # ================================================================
    for param in model.adaptive_A.parameters():
        param.requires_grad = False
    for param in model.adaptive_B.parameters():
        param.requires_grad = False

    adaptive_A_params = sum(
        p.numel() for p in model.adaptive_A.parameters()
    )
    adaptive_B_params = sum(
        p.numel() for p in model.adaptive_B.parameters()
    )
    logger.info(
        f"Froze adaptive θ: {adaptive_A_params + adaptive_B_params:,} params"
    )

    # ================================================================
    # Learnable β and α — parameterised in logit space
    # ================================================================
    def _inv_sigmoid(x: float) -> float:
        x = max(min(x, 1.0 - 1e-7), 1e-7)
        return math.log(x / (1.0 - x))

    beta_init = getattr(config, "beta", 0.999)
    alpha_init = getattr(config, "session_reset_alpha", 0.5)

    beta_logit = nn.Parameter(
        torch.tensor(_inv_sigmoid(beta_init), dtype=torch.float32, device=device)
    )
    alpha_logit = nn.Parameter(
        torch.tensor(_inv_sigmoid(alpha_init), dtype=torch.float32, device=device)
    )

    logger.info(
        f"Learnable dynamics: β_init={beta_init:.4f} "
        f"(logit={beta_logit.item():.3f}), "
        f"α_init={alpha_init:.3f} "
        f"(logit={alpha_logit.item():.3f})"
    )

    # ================================================================
    # Optimizer
    # ================================================================
    lr = float(getattr(config, "lr_phase3", 1e-4))
    optimizer = torch.optim.Adam(
        [
            {"params": model.consolidation.parameters(), "lr": lr},
            {"params": [beta_logit], "lr": lr * 0.1},   # slower LR for β
            {"params": [alpha_logit], "lr": lr * 0.1},   # slower LR for α
        ],
    )

    num_runs = getattr(config, "num_runs_p3", 100)

    # ================================================================
    # Build domain dataloaders
    # ================================================================
    domain_dataloaders: dict[str, DataLoader] = {}
    domain_iters: dict[str, Any] = {}
    for domain in DOMAINS:
        dl = build_domain_dataloader(config, domain, synthetic=True)
        domain_dataloaders[domain] = dl
        domain_iters[domain] = iter(dl)

    # ================================================================
    # W&B
    # ================================================================
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
                    tags=["phase3"],
                )
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging")
            use_wandb = False

    # ================================================================
    # Logging config
    # ================================================================
    log_every = getattr(config, "log_every", 10)
    save_every = getattr(config, "save_every", 50)
    save_path = str(
        Path(getattr(config, "save_dir", "checkpoints")) / "phase3.pt"
    )

    logger.info(
        f"Phase 3 consolidation training — {num_runs} runs, "
        f"lr={lr}, device={device}"
    )

    # ================================================================
    # Training loop
    # ================================================================
    best_improvement = float("-inf")
    t0 = time.time()

    running_loss = 0.0
    running_improvement = 0.0
    running_forgetting = 0.0
    final_metrics: dict[str, Any] = {}

    for run_idx in range(num_runs):
        metrics = train_one_run(
            model,
            config,
            optimizer,
            beta_logit,
            alpha_logit,
            domain_dataloaders,
            domain_iters,
            run_idx,
        )

        running_loss += metrics["mean_loss"]
        running_improvement += metrics["cross_session_improvement"]
        running_forgetting += metrics["forgetting"]
        final_metrics = metrics

        # ---- Periodic logging ----
        if (run_idx + 1) % log_every == 0:
            avg_loss = running_loss / log_every
            avg_impr = running_improvement / log_every
            avg_forg = running_forgetting / log_every
            elapsed = time.time() - t0

            logger.info(
                f"[Run {run_idx + 1}/{num_runs}]  "
                f"loss={avg_loss:.4f}  "
                f"improvement={avg_impr:+.4f}  "
                f"forgetting={avg_forg:+.4f}  "
                f"β={metrics['beta']:.4f}  "
                f"α={metrics['alpha']:.3f}  "
                f"domains={metrics['d1_name']}/{metrics['d2_name']}  "
                f"elapsed={elapsed:.0f}s"
            )

            if use_wandb:
                import wandb

                log_dict: dict[str, Any] = {
                    "run": run_idx + 1,
                    "mean_loss": avg_loss,
                    "cross_session_improvement": avg_impr,
                    "forgetting": avg_forg,
                    "beta": metrics["beta"],
                    "alpha": metrics["alpha"],
                    "d1_early_loss": metrics.get("d1_early_loss", 0),
                    "d1_late_loss": metrics.get("d1_late_loss", 0),
                    "d1_return_loss": metrics.get("d1_return_loss", 0),
                    "d2_loss": metrics.get("d2_loss", 0),
                }
                log_dict.update(
                    model.consolidation.consolidated_weight_stats()
                )
                wandb.log(log_dict, step=run_idx + 1)

            if avg_impr > best_improvement:
                best_improvement = avg_impr

            running_loss = 0.0
            running_improvement = 0.0
            running_forgetting = 0.0

        # ---- Periodic checkpoint ----
        if (run_idx + 1) % save_every == 0:
            _save_phase3_checkpoint(
                model, beta_logit, alpha_logit, save_path, run_idx + 1,
            )

    # ================================================================
    # Final save and cleanup
    # ================================================================
    _save_phase3_checkpoint(
        model, beta_logit, alpha_logit, save_path, num_runs,
    )

    # Apply final β / α to model config
    final_beta = torch.sigmoid(beta_logit).item()
    final_alpha = torch.sigmoid(alpha_logit).item()
    model.consolidation.beta = final_beta
    model.config.session_reset_alpha = final_alpha

    # Unfreeze adaptive θ (restore original state)
    for param in model.adaptive_A.parameters():
        param.requires_grad = True
    for param in model.adaptive_B.parameters():
        param.requires_grad = True

    elapsed = time.time() - t0
    logger.info(
        f"Phase 3 complete — {num_runs} runs in {elapsed:.0f}s, "
        f"final β={final_beta:.4f}, final α={final_alpha:.3f}, "
        f"best improvement={best_improvement:+.4f}"
    )

    return {
        "final_beta": final_beta,
        "final_alpha": final_alpha,
        "best_improvement": best_improvement,
        "num_runs": num_runs,
        "save_path": save_path,
    }


# ------------------------------------------------------------------ #
# Checkpoint utilities                                                 #
# ------------------------------------------------------------------ #

def _save_phase3_checkpoint(
    model,
    beta_logit: nn.Parameter,
    alpha_logit: nn.Parameter,
    path: str | Path,
    run_idx: int,
) -> None:
    """Save Phase 3 training state to disk."""
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run": run_idx,
            "consolidation": model.consolidation.state_dict(),
            "adaptive_A": model.adaptive_A.state_dict(),
            "adaptive_B": model.adaptive_B.state_dict(),
            "beta_logit": beta_logit.data.cpu(),
            "alpha_logit": alpha_logit.data.cpu(),
            "beta": torch.sigmoid(beta_logit).item(),
            "alpha": torch.sigmoid(alpha_logit).item(),
        },
        save_path,
    )
    logger.info(f"Phase 3 checkpoint saved → {save_path}  (run {run_idx})")


def load_phase3_checkpoint(
    model,
    path: str,
) -> tuple[int, float, float]:
    """
    Load Phase 3 checkpoint.

    Restores consolidation, adaptive-A and adaptive-B state dicts.

    Parameters
    ----------
    model : NATModel
    path : str
        Path to the checkpoint file.

    Returns
    -------
    (run_idx, beta_value, alpha_value)
    """
    state = torch.load(path, weights_only=True, map_location="cpu")
    model.consolidation.load_state_dict(state["consolidation"])
    if "adaptive_A" in state:
        model.adaptive_A.load_state_dict(state["adaptive_A"])
    if "adaptive_B" in state:
        model.adaptive_B.load_state_dict(state["adaptive_B"])

    run_idx = state.get("run", 0)
    beta_val = state.get("beta", 0.999)
    alpha_val = state.get("alpha", 0.5)

    model.consolidation.beta = beta_val
    model.config.session_reset_alpha = alpha_val

    logger.info(
        f"Phase 3 checkpoint loaded ← {path}  "
        f"(run {run_idx}, β={beta_val:.4f}, α={alpha_val:.3f})"
    )
    return run_idx, beta_val, alpha_val
