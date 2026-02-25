"""
FastNeuron — adaptive memory neuron for NAT v2.

Hooks into a frozen base model layer and runs a 7-step pipeline each chunk:
  1. Observe    — predict hidden state, compute surprise
  2. Write      — outer-product write to associative memory
  3. Read       — attention-based memory recall
  4. Project    — bottleneck projection with residual
  5. Proj Write — surprise-gated rank-1 updates to projection weights
  6. Gate+Inject— gated injection back into hidden stream
  7. Report     — compress summary for slow neuron

See NAT_v2_Spec.md for full details.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FastNeuron(nn.Module):

    def __init__(
        self,
        d_model: int = 2560,
        rank: int = 64,
        d_query: int = 128,
        d_value: int = 512,       # internal attention value dim, projected up to d_model
        d_proj: int = 128,
        d_context: int = 128,
        d_report: int = 128,
        d_hidden: int = 384,
        max_norm: float = 10.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.d_query = d_query
        self.d_value = d_value
        self.d_proj = d_proj
        self.d_context = d_context
        self.d_report = d_report
        self.d_hidden = d_hidden
        self.max_norm = max_norm

        # ---- Step 1: OBSERVE ----
        # state_predictor: (d_model + d_context) -> d_hidden -> d_model
        self.state_predictor = nn.Sequential(
            nn.Linear(d_model + d_context, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )
        # surprise_net: (d_model + d_context) -> d_hidden -> 1 -> sigmoid
        self.surprise_net = nn.Sequential(
            nn.Linear(d_model + d_context, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid(),
        )

        # ---- Step 2: MEMORY WRITE ----
        write_in = d_model + 1 + d_context
        self.write_key_net = nn.Sequential(
            nn.Linear(write_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, rank),
        )
        self.write_value_net = nn.Sequential(
            nn.Linear(write_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )
        # lr_net: (1 + d_context) -> d_hidden//2 -> 1 -> softplus, clamped to 0.1
        self.lr_net = nn.Sequential(
            nn.Linear(1 + d_context, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Softplus(),
        )
        # Init bias negative so softplus starts below 0.1 clamp (preserves grad)
        nn.init.constant_(self.lr_net[-2].bias, -3.0)

        # ---- Step 3: MEMORY READ (Attention) ----
        self.slot_layer_norm = nn.LayerNorm(d_model)
        # read_query_net is a single linear (not MLP) per spec param count
        self.read_query_net = nn.Linear(d_model + d_context, d_query)
        self.W_K = nn.Parameter(torch.randn(d_model, d_query) * 0.02)
        self.W_V = nn.Parameter(torch.randn(d_model, d_value) * 0.02)
        # Project compressed value back up to d_model for clean residual
        self.value_up_proj = nn.Linear(d_value, d_model)

        # ---- Step 4: PROJECTION (bottleneck with residual) ----
        # Initialized small so projected ~ 0 at init, output ~ mem_read
        self.W_down_base = nn.Parameter(torch.randn(d_model, d_proj) * 0.01)
        self.W_up_base = nn.Parameter(torch.randn(d_proj, d_model) * 0.01)

        # ---- Step 5: PROJECTION WRITE ----
        self.threshold_net = nn.Sequential(
            nn.Linear(d_context, 1),
            nn.Sigmoid(),
        )
        self.fixed_threshold: float | None = 0.5
        proj_write_in = d_model + 1 + d_model + d_context
        self.proj_write_down_net = nn.Sequential(
            nn.Linear(proj_write_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model + d_proj),
        )
        self.proj_write_up_net = nn.Sequential(
            nn.Linear(proj_write_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj + d_model),
        )
        # proj_lr_net: (1+d_context) -> 64 -> 1 -> softplus (per spec param count)
        self.proj_lr_net = nn.Sequential(
            nn.Linear(1 + d_context, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        nn.init.constant_(self.proj_lr_net[-2].bias, -3.0)

        # ---- Step 6: GATE AND INJECT ----
        self.gate_net = nn.Sequential(
            nn.Linear(d_model + d_model + d_context, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid(),
        )
        # Gate starts low (bias = -1.0 on the linear before sigmoid)
        nn.init.constant_(self.gate_net[-2].bias, -1.0)
        self.layer_norm = nn.LayerNorm(d_model)

        # ---- Step 7: REPORT ----
        report_in = d_model + 1 + d_proj + 1  # error + surprise + down + gate_value
        self.report_net = nn.Sequential(
            nn.Linear(report_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_report),
        )

        # Per-user state (initialized by start_session)
        self.mem_A = None
        self.W_down_mod = None
        self.W_up_mod = None
        self.prev_h_avg = None
        self.context = None
        self.last_report = None
        self.adapt_mode = True

        # Diagnostics (set each forward call, for logging)
        self.last_surprise = None
        self.last_gate = None
        self.last_threshold = None
        self.last_proj_write_mask = None

    def start_session(self, batch_size: int, device: torch.device):
        """Reset all per-user state for a new episode."""
        self.mem_A = torch.zeros(batch_size, self.d_model, self.rank, device=device)
        self.W_down_mod = torch.zeros(batch_size, self.d_model, self.d_proj, device=device)
        self.W_up_mod = torch.zeros(batch_size, self.d_proj, self.d_model, device=device)
        self.prev_h_avg = None
        self.context = torch.zeros(batch_size, self.d_context, device=device)

    def forward(self, h: torch.Tensor, chunk_idx=None) -> torch.Tensor:
        """
        Full 7-step pipeline.

        Args:
            h: (batch, seq, d_model) hidden states from base model layer
            chunk_idx: chunk position in episode (0-based). Writes skipped on chunk 0.

        Returns:
            h_new: (batch, seq, d_model) modified hidden states
        """
        batch_size = h.shape[0]
        device = h.device

        # Allow chunk_idx to be set as attribute (for hook-based calling)
        if chunk_idx is None:
            chunk_idx = getattr(self, 'chunk_idx', None)

        # Reset diagnostics for this forward pass
        self.last_surprise = None
        self.last_gate = None
        self.last_threshold = None
        self.last_proj_write_mask = None

        if self.mem_A is None:
            self.start_session(batch_size, device)

        # ================================================================
        # Step 1: OBSERVE
        # ================================================================
        h_avg = h.mean(dim=1)                                          # (batch, d_model)

        prev = torch.zeros_like(h_avg) if self.prev_h_avg is None else self.prev_h_avg
        predicted_h = self.state_predictor(
            torch.cat([prev, self.context], dim=-1)
        )                                                              # (batch, d_model)
        error = h_avg - predicted_h                                    # (batch, d_model)
        surprise = self.surprise_net(
            torch.cat([error, self.context], dim=-1)
        )                                                              # (batch, 1)
        self.last_surprise = surprise.detach()

        # State update: prev_h_avg is detached (not part of gradient chain)
        self.prev_h_avg = h_avg.detach()

        # ================================================================
        # Step 2: MEMORY WRITE (adapt mode only, skip chunk 0)
        # ================================================================
        if self.adapt_mode and chunk_idx != 0:
            write_input = torch.cat([h_avg, surprise, self.context], dim=-1)
            key = self.write_key_net(write_input)                      # (batch, rank)
            value = self.write_value_net(write_input)                  # (batch, d_model)

            lr = self.lr_net(torch.cat([surprise, self.context], dim=-1))
            lr = torch.clamp(lr, max=0.1)                             # (batch, 1)

            # Outer product write (= not +=)
            self.mem_A = self.mem_A + lr.unsqueeze(-1) * torch.bmm(
                value.unsqueeze(2), key.unsqueeze(1)
            )                                                          # (batch, d_model, rank)

            # Norm clamp for stability
            norm = torch.norm(self.mem_A, dim=(1, 2), keepdim=True)
            self.mem_A = self.mem_A * torch.where(
                norm > self.max_norm,
                self.max_norm / (norm + 1e-8),
                torch.ones_like(norm),
            )

        # ================================================================
        # Early exit: pure passthrough when no memory content.
        # Prevents θ bias terms (in read_query_net, value_up_proj,
        # gate_net, etc.) from acting as static task-specific offsets.
        # ================================================================
        if self.mem_A.abs().max() < 1e-9:
            self.last_report = torch.zeros(
                batch_size, self.d_report, device=device
            )
            self.last_gate = torch.zeros(batch_size, 1, device=device)
            return h

        # ================================================================
        # Step 3: MEMORY READ (Attention)
        # ================================================================
        slots = self.mem_A.transpose(1, 2)                             # (batch, rank, d_model)
        slots = self.slot_layer_norm(slots)

        query = self.read_query_net(
            torch.cat([h_avg, self.context], dim=-1)
        )                                                              # (batch, d_query)

        W_K_exp = self.W_K.unsqueeze(0).expand(batch_size, -1, -1)
        W_V_exp = self.W_V.unsqueeze(0).expand(batch_size, -1, -1)

        keys = torch.bmm(slots, W_K_exp)                              # (batch, rank, d_query)
        values = torch.bmm(slots, W_V_exp)                            # (batch, rank, d_value)

        attn_scores = torch.bmm(
            query.unsqueeze(1), keys.transpose(1, 2)
        ) / math.sqrt(self.d_query)                                    # (batch, 1, rank)
        attn_weights = torch.softmax(attn_scores, dim=-1)             # (batch, 1, rank)

        mem_read_compressed = torch.bmm(attn_weights, values).squeeze(1)  # (batch, d_value)
        mem_read = self.value_up_proj(mem_read_compressed)             # (batch, d_model)

        # ================================================================
        # Step 4: PROJECTION (bottleneck with residual)
        # ================================================================
        W_down_eff = self.W_down_base.unsqueeze(0) + self.W_down_mod  # (batch, d_model, d_proj)
        W_up_eff = self.W_up_base.unsqueeze(0) + self.W_up_mod       # (batch, d_proj, d_model)

        down = F.gelu(
            torch.bmm(mem_read.unsqueeze(1), W_down_eff).squeeze(1)
        )                                                              # (batch, d_proj)
        projected = torch.bmm(down.unsqueeze(1), W_up_eff).squeeze(1) # (batch, d_model)
        output = 0.5 * mem_read + projected                           # (batch, d_model)

        # ================================================================
        # Step 5: PROJECTION WRITE (adapt mode, skip chunk 0, soft threshold)
        # ================================================================
        if self.adapt_mode and chunk_idx != 0:
            if self.fixed_threshold is not None:
                threshold = torch.full((batch_size, 1), self.fixed_threshold, device=device)
            else:
                threshold = self.threshold_net(self.context)           # (batch, 1)
            write_strength = torch.sigmoid(10.0 * (surprise - threshold))  # soft mask
            self.last_threshold = threshold.detach()
            self.last_proj_write_mask = write_strength.detach()

            proj_write_input = torch.cat(
                [h_avg, surprise, mem_read, self.context], dim=-1
            )
            raw_down = self.proj_write_down_net(proj_write_input)
            d_pat  = raw_down[:, :self.d_model]
            d_addr = raw_down[:, self.d_model:]

            raw_up = self.proj_write_up_net(proj_write_input)
            u_pat  = raw_up[:, :self.d_proj]
            u_addr = raw_up[:, self.d_proj:]

            proj_lr = self.proj_lr_net(
                torch.cat([surprise, self.context], dim=-1)
            )
            proj_lr = torch.clamp(proj_lr, max=0.1)                   # (batch, 1)

            # Rank-1 outer product writes gated by soft threshold (= not +=)
            self.W_down_mod = self.W_down_mod + (
                write_strength.unsqueeze(-1) * proj_lr.unsqueeze(-1)
                * torch.bmm(d_pat.unsqueeze(2), d_addr.unsqueeze(1))
            )                                                          # (batch, d_model, d_proj)

            self.W_up_mod = self.W_up_mod + (
                write_strength.unsqueeze(-1) * proj_lr.unsqueeze(-1)
                * torch.bmm(u_pat.unsqueeze(2), u_addr.unsqueeze(1))
            )                                                          # (batch, d_proj, d_model)

        # ================================================================
        # Step 6: GATE AND INJECT
        # ================================================================
        g = self.gate_net(
            torch.cat([h_avg, output, self.context], dim=-1)
        )                                                              # (batch, 1)
        self.last_gate = g.detach()

        h_new = h + g.unsqueeze(1) * output.unsqueeze(1)              # (batch, seq, d_model)
        h_new = self.layer_norm(h_new)

        # ================================================================
        # Step 7: REPORT
        # ================================================================
        self.last_report = self.report_net(
            torch.cat([error, surprise, down, g], dim=-1)
        )                                                              # (batch, d_report)

        return h_new
