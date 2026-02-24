"""
Shared training utilities for NAT.

Provides checkpoint save/load, truncated BPTT, and other helpers
used across both training phases.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# KV cache helpers                                                     #
# ------------------------------------------------------------------ #

def detach_kv_cache(past_key_values):
    """Detach all tensors in a HuggingFace past_key_values object.

    This breaks the gradient path through the KV cache so that
    gradients for θ flow exclusively through the fast-weight updates,
    not through the frozen base-model's attention.

    Works with ``DynamicCache`` (any transformers version) and legacy
    tuple-of-tuples.
    """
    if past_key_values is None:
        return None

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError:
        DynamicCache = None

    # ---- DynamicCache: convert → detach → reconstruct ----
    if DynamicCache is not None and isinstance(past_key_values, DynamicCache):
        # to_legacy_cache() → tuple of (key, value) per layer
        legacy = past_key_values.to_legacy_cache()
        detached = tuple(
            (k.detach(), v.detach()) for k, v in legacy
        )
        return DynamicCache.from_legacy_cache(detached)

    # ---- Legacy tuple-of-tuples ----
    detached = tuple(
        tuple(t.detach() if t is not None else None for t in layer_kv)
        for layer_kv in past_key_values
    )
    # Wrap into DynamicCache if available so .get_seq_length() works
    if DynamicCache is not None:
        try:
            return DynamicCache.from_legacy_cache(detached)
        except Exception:
            pass
    return detached


# ------------------------------------------------------------------ #
# Truncated BPTT helper                                                #
# ------------------------------------------------------------------ #

def maybe_truncate(model, chunk_idx: int, config) -> None:
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


# Keep old name as alias for backward compatibility
_maybe_truncate = maybe_truncate


# ------------------------------------------------------------------ #
# Checkpoint utilities                                                 #
# ------------------------------------------------------------------ #

def save_checkpoint(model, path: str, episode_idx: int) -> None:
    """Save trainable parameters to disk (strips torch.compile prefix)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode_idx,
            "adaptive_A": _strip_compile_prefix(model.adaptive_A.state_dict()),
            "adaptive_B": _strip_compile_prefix(model.adaptive_B.state_dict()),
            "consolidation": _strip_compile_prefix(model.consolidation.state_dict()),
        },
        path,
    )
    logger.info(f"Checkpoint saved → {path}  (episode {episode_idx})")


# Keep old name as alias for backward compatibility
_save_checkpoint = save_checkpoint


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Strip ``_orig_mod.`` prefix added by ``torch.compile()``."""
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        cleaned[new_key] = v
    return cleaned


def load_checkpoint(model, path: str) -> int:
    """
    Load trainable parameters from a checkpoint.

    Handles checkpoints saved with or without ``torch.compile()``.
    Returns the episode index stored in the checkpoint.
    """
    state = torch.load(path, weights_only=True, map_location="cpu")
    model.adaptive_A.load_state_dict(
        _strip_compile_prefix(state["adaptive_A"])
    )
    model.adaptive_B.load_state_dict(
        _strip_compile_prefix(state["adaptive_B"])
    )
    model.consolidation.load_state_dict(
        _strip_compile_prefix(state["consolidation"])
    )
    episode = state.get("episode", 0)
    logger.info(f"Checkpoint loaded ← {path}  (episode {episode})")
    return episode
