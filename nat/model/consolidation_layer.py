"""
Consolidation Layer — persistent slow-learning memory for NAT.

The consolidation layer accumulates knowledge across sessions via an
exponential moving average (EMA) of the adaptive layers' fast weights.

Lifecycle:
  1. During a session's forward pass the consolidation layer is **read-only**:
     it queries its consolidated weights W_c with the current hidden states
     and produces a gated residual output (identical mechanics to the
     adaptive layer's read path).
  2. **Between sessions** the ``consolidate()`` method is called.  It blends
     the adaptive layers' current fast weights into W_c via EMA:
         W_c ← β·W_c + (1 − β)·mean(fast weights from adaptive layers)
     This is a ``@torch.no_grad()`` operation — consolidation is not trained
     end-to-end; instead the EMA rate β and the consolidation read / gate
     networks are trained in Phase 3.

The consolidated weights are stored as registered buffers so they are
automatically saved / loaded with ``state_dict`` and are persistent across
``model.eval()`` / ``model.train()`` transitions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nat.model.adaptive_layer import AdaptiveMemoryLayer


class ConsolidationLayer(nn.Module):
    """
    Slow-learning layer that accumulates knowledge via EMA.

    Parameters
    ----------
    d_model : int
        Dimensionality of the transformer hidden states.
    rank : int
        Rank of the low-rank consolidated weight matrices.
    d_hidden : int
        Hidden dimension for read / gate networks.
    beta : float
        EMA decay rate for consolidation (higher = slower change).
        Typical value: 0.999.
    """

    def __init__(
        self,
        d_model: int,
        rank: int = 32,
        d_hidden: int = 256,
        beta: float = 0.999,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.beta = beta

        # --- Read network ---
        # Transforms raw memory readout before gating.
        self.read_net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

        # --- Gate network ---
        # Decides how much to trust consolidated memory vs. pass-through.
        # Bias initialised to -5.0 → initial gate ≈ 0.007 (near-identity).
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_net[-2].bias, -5.0)

        # Layer norm for output stability
        self.layer_norm = nn.LayerNorm(d_model)

        # ================================================================
        # Consolidated weights — persistent across sessions.
        # Registered as buffers (not parameters) so they survive
        # state_dict round-trips but are NOT included in optimizer
        # param groups by default.
        # ================================================================
        self.register_buffer("W_c_A", torch.zeros(1, d_model, rank))
        self.register_buffer("W_c_B", torch.zeros(1, rank, d_model))

    # ------------------------------------------------------------------ #
    # Consolidation (between-session update)                               #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def consolidate(self, adaptive_layers: list[AdaptiveMemoryLayer]) -> None:
        """
        EMA update of consolidated weights from adaptive layers.

        Call this **after** each session (``NATModel.end_session``).

        The update rule:
            W_c_A ← β·W_c_A + (1 − β)·mean_over_layers(mean_over_batch(fast_A))
            W_c_B ← β·W_c_B + (1 − β)·mean_over_layers(mean_over_batch(fast_B))

        Parameters
        ----------
        adaptive_layers : list[AdaptiveMemoryLayer]
            The adaptive layers whose fast weights will be consolidated.
            Each layer's fast weights have shape ``(batch, d_model, rank)``
            / ``(batch, rank, d_model)``; we average over the batch dim
            first (``dim=0``), then over layers.
        """
        # Collect and average fast weights
        # Per-layer: mean over batch → (d_model, rank)
        # Then mean over layers → (d_model, rank)
        avg_A = torch.stack(
            [layer.fast_A.mean(dim=0) for layer in adaptive_layers]
        ).mean(dim=0, keepdim=True)  # (1, d_model, rank)

        avg_B = torch.stack(
            [layer.fast_B.mean(dim=0) for layer in adaptive_layers]
        ).mean(dim=0, keepdim=True)  # (1, rank, d_model)

        # EMA update
        self.W_c_A = self.beta * self.W_c_A + (1 - self.beta) * avg_A
        self.W_c_B = self.beta * self.W_c_B + (1 - self.beta) * avg_B

    # ------------------------------------------------------------------ #
    # Forward (read-only)                                                  #
    # ------------------------------------------------------------------ #

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Read-only forward pass through consolidated memory.

        Parameters
        ----------
        h_t : Tensor, shape ``(batch, seq_len, d_model)``
            Hidden states from the frozen transformer.

        Returns
        -------
        Tensor, shape ``(batch, seq_len, d_model)``
            Modified hidden states (residual + gated memory readout).
        """
        batch, seq_len, d_model = h_t.shape

        # Expand consolidated weights to match batch size
        W_A = self.W_c_A.expand(batch, -1, -1)  # (batch, d_model, rank)
        W_B = self.W_c_B.expand(batch, -1, -1)  # (batch, rank, d_model)

        # Query consolidated weights: W_c @ h = (W_A @ W_B) @ h
        h_T = h_t.transpose(1, 2)               # (batch, d_model, seq_len)
        memory_raw = torch.bmm(
            W_A, torch.bmm(W_B, h_T)
        ).transpose(1, 2)                        # (batch, seq_len, d_model)

        # Process through read network + normalise the memory branch
        memory_output = self.layer_norm(
            self.read_net(memory_raw)
        )                                            # (batch, seq_len, d_model)

        # Gate: how much to trust consolidated memory
        gate_input = torch.cat([h_t, memory_output], dim=-1)
        gate = self.gate_net(gate_input)           # (batch, seq_len, 1)

        # Residual connection with gated memory readout.
        # When gate ≈ 0 the output is exactly h_t (identity).
        return h_t + gate * memory_output

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_state(self, path: str | Path) -> None:
        """Save consolidated weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"W_c_A": self.W_c_A, "W_c_B": self.W_c_B, "beta": self.beta},
            path,
        )

    def load_state(self, path: str | Path) -> None:
        """Load consolidated weights from disk."""
        state = torch.load(path, weights_only=True, map_location="cpu")
        self.W_c_A.copy_(state["W_c_A"])
        self.W_c_B.copy_(state["W_c_B"])
        if "beta" in state:
            self.beta = state["beta"]

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def consolidated_weight_stats(self) -> dict[str, float]:
        """Return diagnostic statistics about consolidated weights."""
        return {
            "W_c_A_norm": torch.norm(self.W_c_A).item(),
            "W_c_B_norm": torch.norm(self.W_c_B).item(),
            "W_c_A_mean": self.W_c_A.mean().item(),
            "W_c_B_mean": self.W_c_B.mean().item(),
            "W_c_A_max": self.W_c_A.abs().max().item(),
            "W_c_B_max": self.W_c_B.abs().max().item(),
            "beta": self.beta,
        }

    @property
    def is_empty(self) -> bool:
        """True if consolidated weights are still all-zero (no consolidation yet)."""
        return (self.W_c_A.abs().sum() + self.W_c_B.abs().sum()).item() == 0.0
