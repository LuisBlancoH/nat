"""
Adaptive Memory Layer — the core self-modifying component of NAT.

This layer has two kinds of parameters:
- Slow parameters (θ): define HOW to learn. Trained via meta-learning (Phase 1-2),
  then frozen forever. These are standard nn.Module parameters.
- Fast weights (W): working memory that changes DURING inference via the learned
  learning rule. Stored as plain tensors (not nn.Parameters), updated by the
  learning rule defined by θ. Never directly trained by an optimizer.

The fast weights are low-rank: W = fast_A @ fast_B where
    fast_A is (batch, d_model, rank)
    fast_B is (batch, rank, d_model)

This keeps the fast-weight footprint small (~100K-200K parameters per layer).

Key invariants:
  1. All operations on fast weights are NON-in-place during training so that
     autograd can trace through the entire adaptation chain (BPTT).
  2. Each batch element has independent fast weights.
  3. Norm clamping keeps fast weights bounded for numerical stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMemoryLayer(nn.Module):
    """
    A self-modifying layer that learns from hidden states during inference.

    Parameters
    ----------
    d_model : int
        Dimensionality of the transformer hidden states.
    rank : int
        Rank of the low-rank fast weight matrices (W = A @ B).
    d_hidden : int
        Hidden dimension for the slow-parameter networks (θ).
    lr_clamp : float
        Maximum adaptive learning rate (stability constraint).
    fast_weight_max_norm : float
        Maximum Frobenius norm of fast_A (stability constraint).
    """

    def __init__(
        self,
        d_model: int,
        rank: int = 32,
        d_hidden: int = 256,
        lr_clamp: float = 0.1,
        fast_weight_max_norm: float = 10.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.lr_clamp = lr_clamp
        self.fast_weight_max_norm = fast_weight_max_norm

        # ================================================================
        # SLOW PARAMETERS (θ) — trained via meta-learning, then frozen
        # ================================================================

        # --- Surprise network ---
        # Reads the prediction error (h_t - predicted_h) and outputs a
        # scalar surprise score in [0, 1].
        self.surprise_net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid(),
        )

        # --- Learning-rate network ---
        # Maps surprise magnitude to how aggressively fast weights update.
        self.lr_net = nn.Sequential(
            nn.Linear(1, d_hidden // 4),
            nn.GELU(),
            nn.Linear(d_hidden // 4, 1),
            nn.Softplus(),  # lr > 0 always
        )

        # --- Write networks ---
        # Compute an outer-product weight update:
        #   delta_A = lr * write_value ⊗ write_key
        # write_key_net:   (d_model + 1) → rank
        # write_value_net: (d_model + 1) → d_model
        self.write_key_net = nn.Sequential(
            nn.Linear(d_model + 1, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, rank),
        )
        self.write_value_net = nn.Sequential(
            nn.Linear(d_model + 1, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

        # --- Read network ---
        # Transforms the raw memory readout before gating.
        self.read_net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

        # --- Gate network ---
        # Decides how much to trust the memory readout vs. the pass-through.
        # Bias initialised to -1.0 so initial gate ≈ 0.27.  With LayerNorm
        # on the memory branch (not the output), this injects 27% of a
        # unit-variance normalised readout — a moderate perturbation that
        # gives the sigmoid gradient (~0.20) enough strength for the
        # write/key/lr networks to learn quickly.  If adaptation is not
        # useful the gate will close; if it is, it stays open.
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_net[-2].bias, -1.0)

        # --- State predictor ---
        # Predicts the next hidden state for surprise computation.
        self.state_predictor = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )

        # Layer norm for output stability
        self.layer_norm = nn.LayerNorm(d_model)

        # ================================================================
        # FAST WEIGHTS (W) — NOT optimizer parameters
        # ================================================================
        # fast_A starts at zero every session so the model cannot cheat by
        # packing task-specific biases into the initial fast weights.
        # It must earn all improvement through the learned adaptation mechanics.
        self.register_buffer('fast_A_init', torch.zeros(d_model, rank))

        # fast_B is a learned FIXED projection (never updated during inference).
        # It defines HOW memories are decoded — a genuine slow parameter.
        self.fast_B_init = nn.Parameter(torch.randn(rank, d_model) * 0.01)

        # Runtime fast weights — set by reset_fast_weights().
        # These are plain tensors, NOT nn.Parameters.
        self.fast_A: torch.Tensor | None = None   # (batch, d_model, rank)
        self.fast_B: torch.Tensor | None = None   # (batch, rank, d_model)

        # Previous hidden state for surprise computation (per-session state).
        self.prev_h: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def reset_fast_weights(self, batch_size: int = 1) -> None:
        """
        Reset fast weights to zero (the fixed initial values).

        Call at the start of every session / episode.  Uses ``.clone()``
        so the resulting tensor is a fresh leaf in the autograd graph.
        """
        self.fast_A = (
            self.fast_A_init
            .clone()
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .contiguous()
        )
        self.fast_B = (
            self.fast_B_init
            .clone()
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .contiguous()
        )
        self.prev_h = None

    def partial_reset(self, alpha: float = 0.5) -> None:
        """
        Blend fast weights back toward the learned initial values.

        ``alpha = 1.0`` → full reset to init.
        ``alpha = 0.0`` → no reset (keep everything).

        Used between sessions so that some learned knowledge persists.
        """
        assert self.fast_A is not None and self.fast_B is not None, (
            "partial_reset called before reset_fast_weights"
        )
        init_A = self.fast_A_init.unsqueeze(0).expand_as(self.fast_A)
        init_B = self.fast_B_init.unsqueeze(0).expand_as(self.fast_B)
        self.fast_A = alpha * init_A + (1 - alpha) * self.fast_A.detach()
        self.fast_B = alpha * init_B + (1 - alpha) * self.fast_B.detach()

    # ------------------------------------------------------------------ #
    # Self-modification (adaptation)                                       #
    # ------------------------------------------------------------------ #

    def adapt(self, h_t: torch.Tensor) -> None:
        """
        Self-modification step — update fast weights from hidden states.

        Called every ``adapt_every_n`` tokens (not every token).

        CRITICAL: every operation here must be **differentiable** during
        training so that gradients can flow from the evaluation loss back
        through the chain of adaptation steps to θ.

        We use::

            self.fast_A = self.fast_A + delta   # new tensor in graph ✓
            NOT  self.fast_A += delta            # in-place, breaks autograd ✗

        Parameters
        ----------
        h_t : Tensor, shape ``(batch, d_model)``
            Aggregated hidden state (typically the mean over the last
            ``adapt_every_n`` hidden states).
        """
        assert self.fast_A is not None, "adapt() called before reset_fast_weights()"

        # --- Step 1: compute surprise ---
        if self.prev_h is not None:
            predicted_h = self.state_predictor(self.prev_h)
            error = h_t - predicted_h
            surprise = self.surprise_net(error)          # (batch, 1)
        else:
            # First step → maximum surprise
            surprise = torch.ones(
                h_t.shape[0], 1, device=h_t.device, dtype=h_t.dtype
            )

        # --- Step 2: surprise → adaptive learning rate ---
        lr = self.lr_net(surprise)                       # (batch, 1)
        lr = lr.clamp(max=self.lr_clamp)

        # --- Step 3: compute outer-product weight update ---
        write_input = torch.cat([h_t, surprise], dim=-1) # (batch, d_model+1)
        write_key   = self.write_key_net(write_input)     # (batch, rank)
        write_value = self.write_value_net(write_input)   # (batch, d_model)

        # delta_A = lr * (write_value ⊗ write_key)
        # shape:  (batch, d_model, 1) @ (batch, 1, rank) → (batch, d_model, rank)
        delta_A = lr.unsqueeze(-1) * torch.bmm(
            write_value.unsqueeze(-1),   # (batch, d_model, 1)
            write_key.unsqueeze(-2),     # (batch, 1, rank)
        )

        # Non-in-place update → new node in computation graph
        self.fast_A = self.fast_A + delta_A

        # --- Step 4: store hidden state for next surprise computation ---
        # Training: keep in graph for gradient flow
        # Inference: detach to save memory
        self.prev_h = h_t if self.training else h_t.detach()

        # --- Step 5: stability — norm-clamp fast weights ---
        # The scale factor is computed without grad so it doesn't warp the
        # learning-rule gradients, but the multiplication itself IS in-graph.
        with torch.no_grad():
            norm = torch.norm(self.fast_A, dim=(1, 2), keepdim=True)
            scale = torch.clamp(
                self.fast_weight_max_norm / (norm + 1e-8), max=1.0
            )
        self.fast_A = self.fast_A * scale

    # ------------------------------------------------------------------ #
    # Memory read                                                          #
    # ------------------------------------------------------------------ #

    def read(self, h_t: torch.Tensor) -> torch.Tensor:
        """
        Read from fast weights and produce gated output.

        Called for **every token** (not just every N tokens).

        Parameters
        ----------
        h_t : Tensor, shape ``(batch, seq_len, d_model)``

        Returns
        -------
        Tensor, shape ``(batch, seq_len, d_model)``
            Modified hidden states (residual + gated memory readout).
        """
        assert self.fast_A is not None and self.fast_B is not None, (
            "read() called before reset_fast_weights()"
        )

        # If fast_A is still zero (no adaptation has happened this session),
        # return h_t unchanged. This ensures the pre-adaptation baseline is a
        # true identity pass-through and the slow params cannot learn a static
        # offset via the bias terms in read_net / gate_net.
        if self.fast_A.abs().max() < 1e-9:
            return h_t

        batch, seq_len, d_model = h_t.shape

        # Query fast weights: W @ h = (fast_A @ fast_B) @ h
        h_T = h_t.transpose(1, 2)                        # (batch, d_model, seq_len)
        memory_raw = torch.bmm(
            self.fast_A,
            torch.bmm(self.fast_B, h_T),
        ).transpose(1, 2)                                 # (batch, seq_len, d_model)

        # Process through read network + normalise the memory branch
        memory_output = self.layer_norm(
            self.read_net(memory_raw)
        )                                                  # (batch, seq_len, d_model)

        # Gate: how much to trust memory vs. pass-through
        gate_input = torch.cat([h_t, memory_output], dim=-1)
        gate = self.gate_net(gate_input)                   # (batch, seq_len, 1)

        # Residual connection with gated memory readout.
        # When gate ≈ 0 the output is exactly h_t (identity).
        return h_t + gate * memory_output

    # ------------------------------------------------------------------ #
    # Forward pass                                                         #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        h_t: torch.Tensor,
        do_adapt: bool = False,
    ) -> torch.Tensor:
        """
        Full forward pass: optionally adapt, then read.

        Parameters
        ----------
        h_t : Tensor, shape ``(batch, seq_len, d_model)``
            Hidden states from the frozen transformer.
        do_adapt : bool
            Whether to run the self-modification step this call.

        Returns
        -------
        Tensor, shape ``(batch, seq_len, d_model)``
        """
        if do_adapt:
            # Aggregate hidden states → single vector for adaptation
            adapt_signal = h_t.mean(dim=1)  # (batch, d_model)
            self.adapt(adapt_signal)

        return self.read(h_t)

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def fast_weight_stats(self) -> dict[str, float]:
        """Return diagnostic statistics about fast weights and gate."""
        if self.fast_A is None or self.fast_B is None:
            return {"fast_A_norm": 0.0, "fast_B_norm": 0.0}
        # Gate bias (last Linear before Sigmoid in gate_net)
        gate_bias = self.gate_net[-2].bias.item()
        return {
            "fast_A_norm": torch.norm(self.fast_A).item(),
            "fast_B_norm": torch.norm(self.fast_B).item(),
            "fast_A_mean": self.fast_A.mean().item(),
            "fast_B_mean": self.fast_B.mean().item(),
            "fast_A_max":  self.fast_A.abs().max().item(),
            "fast_B_max":  self.fast_B.abs().max().item(),
            "gate_bias":   gate_bias,
            "gate_sigmoid": torch.sigmoid(torch.tensor(gate_bias)).item(),
        }
