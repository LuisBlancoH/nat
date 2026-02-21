"""
Utility helpers for NAT model components.

- Low-rank matrix operations
- Gradient checkpointing wrappers
- Numerical stability helpers
- Device / dtype management
- Device-specific optimisation (Apple Silicon / A100)
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn
from typing import Sequence

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Device / dtype helpers                                               #
# ------------------------------------------------------------------ #

def resolve_device(device: str = "auto") -> torch.device:
    """Pick the best available device."""
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_cast(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast tensor to dtype only if it differs (avoids unnecessary copy)."""
    if tensor.dtype == dtype:
        return tensor
    return tensor.to(dtype)


# ------------------------------------------------------------------ #
# Device-specific optimisation setup                                   #
# ------------------------------------------------------------------ #

def setup_device_optimisations(config) -> None:
    """
    Apply one-time global settings based on the target device.

    Call this *before* model construction.

    Parameters
    ----------
    config : NATConfig
        Must have ``device``, ``tf32_matmul``, ``cudnn_benchmark``,
        ``gradient_checkpointing``, ``compile_model`` attributes
        (all have defaults in NATConfig).

        Lightweight test configs that lack these attributes are handled
        gracefully — missing fields default to safe no-op values.
    """
    device = getattr(config, "device", "cpu")

    if device == "cuda":
        _setup_cuda(config)
    elif device == "mps":
        _setup_mps(config)
    else:
        logger.info("Running on CPU — no device-specific optimisations applied.")


def _setup_cuda(config) -> None:
    """CUDA / A100-specific global settings."""
    if getattr(config, "tf32_matmul", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled TF32 matmul (A100 tensor cores).")

    if getattr(config, "cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
        logger.info("Enabled cuDNN benchmark auto-tuner.")

    # Log GPU info
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        logger.info(
            f"CUDA device: {gpu.name}, "
            f"{gpu.total_mem / 1024**3:.1f} GB VRAM, "
            f"compute capability {gpu.major}.{gpu.minor}"
        )


def _setup_mps(config) -> None:
    """Apple Silicon MPS-specific settings."""
    logger.info("Running on Apple Silicon (MPS).")
    logger.info(
        "Tip: batch_size=1, seq_len≤1024 recommended for 16 GB unified memory."
    )
    if getattr(config, "compile_model", False):
        logger.warning(
            "torch.compile is not fully supported on MPS — disabling."
        )
        config.compile_model = False


def maybe_empty_cache(config, step: int) -> None:
    """
    Periodically clear the device memory cache.

    On MPS, ``torch.mps.empty_cache()`` releases unused memory back to
    the system.  On CUDA, ``torch.cuda.empty_cache()`` does the same but
    is rarely needed.

    Parameters
    ----------
    config : NATConfig
    step : int
        Current global step / episode number.
    """
    every = getattr(config, "empty_cache_every", 0)
    if every <= 0:
        return
    if step % every != 0:
        return

    device = getattr(config, "device", "cpu")
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def device_synchronise(config) -> None:
    """Synchronise the device (useful for accurate timing)."""
    device = getattr(config, "device", "cpu")
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def get_device_memory_info(config) -> dict[str, float]:
    """
    Return current memory usage in GB.

    Returns
    -------
    dict with keys ``"allocated"``, ``"reserved"``, ``"free"``
    (values in GB).  Returns empty dict for unsupported devices.
    """
    device = getattr(config, "device", "cpu")
    if device == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_mem / 1024**3
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(total - allocated, 2),
            "total_gb": round(total, 2),
        }
    elif device == "mps":
        allocated = torch.mps.current_allocated_memory() / 1024**3
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": 0.0,   # MPS doesn't expose this
            "free_gb": 0.0,       # unified memory — not directly queryable
            "total_gb": 0.0,
        }
    return {}


def log_device_memory(config, label: str = "") -> None:
    """Log current device memory usage."""
    info = get_device_memory_info(config)
    if not info:
        return
    prefix = f"[{label}] " if label else ""
    device = getattr(config, "device", "cpu")
    if device == "cuda":
        logger.info(
            f"{prefix}GPU memory: "
            f"{info['allocated_gb']:.2f} GB allocated, "
            f"{info['free_gb']:.2f} GB free / {info['total_gb']:.1f} GB total"
        )
    elif device == "mps":
        logger.info(
            f"{prefix}MPS memory: {info['allocated_gb']:.2f} GB allocated"
        )


# ------------------------------------------------------------------ #
# Low-rank helpers                                                     #
# ------------------------------------------------------------------ #

def low_rank_product(
    A: torch.Tensor,
    B: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ``(A @ B) @ x`` efficiently without materialising A @ B.

    Parameters
    ----------
    A : (batch, d_model, rank) or (d_model, rank)
    B : (batch, rank, d_model) or (rank, d_model)
    x : (batch, seq_len, d_model)

    Returns
    -------
    (batch, seq_len, d_model)
    """
    # x^T : (batch, d_model, seq_len)
    x_T = x.transpose(1, 2)
    # B @ x^T : (batch, rank, seq_len)
    # A @ (B @ x^T) : (batch, d_model, seq_len)
    if A.dim() == 2:
        # Unbatched A, B — broadcast
        result = A @ (B @ x_T)
    else:
        result = torch.bmm(A, torch.bmm(B, x_T))
    return result.transpose(1, 2)


def frobenius_norm_batched(M: torch.Tensor) -> torch.Tensor:
    """Batched Frobenius norm.  M: (batch, *, *)."""
    return torch.norm(M.flatten(start_dim=1), dim=1)


# ------------------------------------------------------------------ #
# Gradient / stability helpers                                         #
# ------------------------------------------------------------------ #

def clip_fast_weight_norm(
    W: torch.Tensor,
    max_norm: float = 10.0,
) -> torch.Tensor:
    """
    Norm-clamp a batched weight tensor.

    The scale factor is computed **without grad** so it doesn't interfere
    with the learning-rule gradient, but the multiplication itself stays
    in-graph.

    Parameters
    ----------
    W : (batch, d1, d2)
    max_norm : float

    Returns
    -------
    Clamped W (same shape).
    """
    with torch.no_grad():
        norm = torch.norm(W, dim=(1, 2), keepdim=True)
        scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
    return W * scale


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Count (optionally trainable-only) parameters in a module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def print_parameter_summary(modules: dict[str, nn.Module]) -> None:
    """Print a table of parameter counts."""
    total = 0
    print(f"{'Module':<30} {'Trainable':>12} {'Total':>12}")
    print("-" * 56)
    for name, mod in modules.items():
        trainable = count_parameters(mod, trainable_only=True)
        all_params = count_parameters(mod, trainable_only=False)
        total += trainable
        print(f"{name:<30} {trainable:>12,} {all_params:>12,}")
    print("-" * 56)
    print(f"{'TOTAL trainable':<30} {total:>12,}")


# ------------------------------------------------------------------ #
# Gradient checkpointing wrapper                                       #
# ------------------------------------------------------------------ #

class CheckpointedSequential(nn.Module):
    """
    Wraps a sequence of modules with ``torch.utils.checkpoint``.

    Useful for the slow-parameter networks in the adaptive layer when
    memory is tight (Apple Silicon / single GPU).
    """

    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            if self.training and x.requires_grad:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False
                )
            else:
                x = layer(x)
        return x
