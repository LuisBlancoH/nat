"""
SlowNeuron — nested adaptive neuron for NAT v2.

Fires every 16 chunks. Accumulates reports from both fast neurons,
runs the 5-step internal pipeline (observe → write → read → project →
proj_write), then produces two outputs:

  A. Context vector — shapes all fast neuron decisions until next firing
  B. Consolidation writes — rank-1 outer-product updates to each fast
     neuron's W_down_mod / W_up_mod projection matrices

The slow neuron operates on aggregated reports (d_model_slow = 256)
with smaller internal dimensions than the fast neurons. Its own context
is a learned constant (no glacial neuron).

See NAT_v2_Spec.md §Slow Neuron for full details.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.fast_neuron import FastNeuron


class SlowNeuron(nn.Module):

    def __init__(
        self,
        d_model_slow: int = 256,    # 2 * d_report — input size
        rank: int = 32,
        d_query: int = 64,
        d_value: int = 128,          # internal value dim, projected up to d_model_slow
        d_proj: int = 64,
        d_context: int = 128,        # output context size (shared with fast neurons)
        d_hidden: int = 192,
        max_norm: float = 10.0,
        fast_d_model: int = 2560,    # fast neuron d_model for consolidation writes
        fast_d_proj: int = 128,      # fast neuron d_proj for consolidation writes
        num_fast_neurons: int = 2,
    ):
        super().__init__()
        self.d_model = d_model_slow
        self.rank = rank
        self.d_query = d_query
        self.d_value = d_value
        self.d_proj = d_proj
        self.d_context = d_context
        self.d_hidden = d_hidden
        self.max_norm = max_norm
        self.fast_d_model = fast_d_model
        self.fast_d_proj = fast_d_proj
        self.num_fast_neurons = num_fast_neurons

        # Slow neuron's own context (learned constant — no glacial neuron)
        self.default_context = nn.Parameter(torch.zeros(d_context))

        # ==============================================================
        # Internal pipeline (steps 1-5, same structure as FastNeuron)
        # ==============================================================

        # ---- Step 1: OBSERVE ----
        # Memory-based surprise: error = h_avg - prev_mem_read
        # Same fix as FastNeuron — prevents state_predictor collapse.
        self.surprise_net = nn.Sequential(
            nn.Linear(d_model_slow + d_context, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid(),
        )

        # ---- Step 2: MEMORY WRITE ----
        write_in = d_model_slow + 1 + d_context
        self.write_key_net = nn.Sequential(
            nn.Linear(write_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, rank),
        )
        self.write_value_net = nn.Sequential(
            nn.Linear(write_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model_slow),
        )
        self.lr_net = nn.Sequential(
            nn.Linear(1 + d_context, d_hidden // 2),
            nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
            nn.Softplus(),
        )
        nn.init.constant_(self.lr_net[-2].bias, -3.0)

        # ---- Step 3: MEMORY READ (Attention) ----
        self.read_query_net = nn.Linear(d_model_slow + d_context, d_query)
        self.W_K = nn.Parameter(torch.randn(d_model_slow, d_query) * 0.02)
        self.W_V = nn.Parameter(torch.randn(d_model_slow, d_value) * 0.02)
        self.value_up_proj = nn.Linear(d_value, d_model_slow)

        # ---- Step 4: PROJECTION (bottleneck with residual) ----
        self.W_down_base = nn.Parameter(torch.randn(d_model_slow, d_proj) * 0.01)
        self.W_up_base = nn.Parameter(torch.randn(d_proj, d_model_slow) * 0.01)

        # ---- Step 5: PROJECTION WRITE ----
        self.threshold_net = nn.Sequential(
            nn.Linear(d_context, 1),
            nn.Sigmoid(),
        )
        proj_write_in = d_model_slow + 1 + d_model_slow + d_context
        proj_write_out = d_model_slow + d_proj + d_proj + d_model_slow
        self.proj_write_net = nn.Sequential(
            nn.Linear(proj_write_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, proj_write_out),
        )
        self.proj_lr_net = nn.Sequential(
            nn.Linear(1 + d_context, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
        nn.init.constant_(self.proj_lr_net[-2].bias, -3.0)

        # ==============================================================
        # Output A: Context compression
        # ==============================================================
        self.context_compress = nn.Linear(d_model_slow, d_context)

        # ==============================================================
        # Output B: Consolidation networks
        # ==============================================================
        # One write net per fast neuron (spec: "per fast neuron, or shared")
        consol_out = fast_d_model + fast_d_proj + fast_d_proj + fast_d_model
        self.consolidation_write_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model_slow + 2, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, consol_out),
            )
            for _ in range(num_fast_neurons)
        ])
        # Shared lr net: Linear(d_model_slow, 1) → Softplus, clamped ~0.01
        self.consolidation_lr_net = nn.Sequential(
            nn.Linear(d_model_slow, 1),
            nn.Softplus(),
        )
        # Start with very small consolidation lr
        nn.init.constant_(self.consolidation_lr_net[0].bias, -5.0)

        # Per-user state (initialized by start_session)
        self.mem_A = None
        self.W_down_mod = None
        self.W_up_mod = None
        self.prev_h_avg = None
        self.prev_mem_read = None
        self.report_buffer: List[torch.Tensor] = []
        self.adapt_mode = True

    def start_session(self, batch_size: int, device: torch.device):
        """Reset all per-user state for a new episode."""
        self.mem_A = torch.zeros(batch_size, self.d_model, self.rank, device=device)
        self.W_down_mod = torch.zeros(batch_size, self.d_model, self.d_proj, device=device)
        self.W_up_mod = torch.zeros(batch_size, self.d_proj, self.d_model, device=device)
        self.prev_h_avg = None
        self.prev_mem_read = None
        self.report_buffer = []

    def detach_state(self):
        """Detach all persistent state from computation graph (Phase 2 window boundary)."""
        if self.mem_A is not None:
            self.mem_A = self.mem_A.detach()
        if self.W_down_mod is not None:
            self.W_down_mod = self.W_down_mod.detach()
        if self.W_up_mod is not None:
            self.W_up_mod = self.W_up_mod.detach()
        if self.prev_h_avg is not None:
            self.prev_h_avg = self.prev_h_avg.detach()
        if self.prev_mem_read is not None:
            self.prev_mem_read = self.prev_mem_read.detach()
        self.report_buffer = [r.detach() for r in self.report_buffer]

    def accumulate_report(self, report: torch.Tensor):
        """
        Append a combined report to the buffer.

        Args:
            report: (batch, 2 * d_report) — concatenated reports from both fast neurons
        """
        self.report_buffer.append(report)

    def fire(self, fast_neurons: List[FastNeuron]) -> torch.Tensor:
        """
        Run the slow neuron pipeline on accumulated reports.

        1. Averages the report buffer to form the input
        2. Runs steps 1-5 (observe, write, read, project, proj_write)
        3. Compresses output to context vector
        4. Performs consolidation writes to each fast neuron's projection

        Args:
            fast_neurons: list of FastNeuron instances to consolidate into

        Returns:
            context: (batch, d_context) — new context for fast neurons
        """
        # Average accumulated reports → slow neuron input
        accumulated = torch.stack(self.report_buffer).mean(dim=0)   # (batch, d_model)
        self.report_buffer = []

        h_avg = accumulated
        batch_size = h_avg.shape[0]
        device = h_avg.device

        if self.mem_A is None:
            self.start_session(batch_size, device)

        # Expand learned default context to batch
        context = self.default_context.unsqueeze(0).expand(batch_size, -1)

        # ==============================================================
        # Step 1: OBSERVE — memory-based surprise
        # ==============================================================
        prev_read = torch.zeros_like(h_avg) if self.prev_mem_read is None else self.prev_mem_read
        error = h_avg - prev_read
        surprise = self.surprise_net(
            torch.cat([error, context], dim=-1)
        )                                                              # (batch, 1)

        self.prev_h_avg = h_avg.detach()

        # ==============================================================
        # Step 2: MEMORY WRITE (adapt mode only)
        # ==============================================================
        if self.adapt_mode:
            write_input = torch.cat([h_avg, surprise, context], dim=-1)
            key = self.write_key_net(write_input)                      # (batch, rank)
            value = self.write_value_net(write_input)                  # (batch, d_model)

            lr = self.lr_net(torch.cat([surprise, context], dim=-1))
            lr = torch.clamp(lr, max=0.1)                             # (batch, 1)

            self.mem_A = self.mem_A + lr.unsqueeze(-1) * torch.bmm(
                value.unsqueeze(2), key.unsqueeze(1)
            )

            norm = torch.norm(self.mem_A, dim=(1, 2), keepdim=True)
            self.mem_A = self.mem_A * torch.where(
                norm > self.max_norm,
                self.max_norm / (norm + 1e-8),
                torch.ones_like(norm),
            )

        # ==============================================================
        # Step 3: MEMORY READ (Attention)
        # ==============================================================
        slots = self.mem_A.transpose(1, 2)                             # (batch, rank, d_model)

        query = self.read_query_net(
            torch.cat([h_avg, context], dim=-1)
        )                                                              # (batch, d_query)

        W_K_exp = self.W_K.unsqueeze(0).expand(batch_size, -1, -1)
        W_V_exp = self.W_V.unsqueeze(0).expand(batch_size, -1, -1)

        keys = torch.bmm(slots, W_K_exp)                              # (batch, rank, d_query)
        values = torch.bmm(slots, W_V_exp)                            # (batch, rank, d_value)

        attn_scores = torch.bmm(
            query.unsqueeze(1), keys.transpose(1, 2)
        ) / math.sqrt(self.d_query)                                    # (batch, 1, rank)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        mem_read_compressed = torch.bmm(attn_weights, values).squeeze(1)
        mem_read = self.value_up_proj(mem_read_compressed)             # (batch, d_model)

        # Save mem_read for next firing's surprise signal
        self.prev_mem_read = mem_read.detach()

        # ==============================================================
        # Step 4: PROJECTION (bottleneck with residual)
        # ==============================================================
        W_down_eff = self.W_down_base.unsqueeze(0) + self.W_down_mod
        W_up_eff = self.W_up_base.unsqueeze(0) + self.W_up_mod

        down = F.gelu(
            torch.bmm(mem_read.unsqueeze(1), W_down_eff).squeeze(1)
        )                                                              # (batch, d_proj)
        projected = torch.bmm(down.unsqueeze(1), W_up_eff).squeeze(1) # (batch, d_model)
        slow_output = mem_read + projected                             # (batch, d_model)

        # ==============================================================
        # Step 5: PROJECTION WRITE (adapt mode, surprise > threshold)
        # ==============================================================
        if self.adapt_mode:
            threshold = self.threshold_net(context)
            mask = (surprise > threshold).float()

            proj_write_input = torch.cat(
                [h_avg, surprise, mem_read, context], dim=-1
            )
            raw = self.proj_write_net(proj_write_input)

            d_pat  = raw[:, :self.d_model]
            d_addr = raw[:, self.d_model : self.d_model + self.d_proj]
            u_pat  = raw[:, self.d_model + self.d_proj : self.d_model + 2 * self.d_proj]
            u_addr = raw[:, self.d_model + 2 * self.d_proj :]

            proj_lr = self.proj_lr_net(
                torch.cat([surprise, context], dim=-1)
            )
            proj_lr = torch.clamp(proj_lr, max=0.1)

            self.W_down_mod = self.W_down_mod + (
                mask.unsqueeze(-1) * proj_lr.unsqueeze(-1)
                * torch.bmm(d_pat.unsqueeze(2), d_addr.unsqueeze(1))
            )
            self.W_up_mod = self.W_up_mod + (
                mask.unsqueeze(-1) * proj_lr.unsqueeze(-1)
                * torch.bmm(u_pat.unsqueeze(2), u_addr.unsqueeze(1))
            )

        # ==============================================================
        # Output A: Context
        # ==============================================================
        new_context = self.context_compress(slow_output)               # (batch, d_context)

        # ==============================================================
        # Output B: Consolidation writes to fast neuron projections
        # ==============================================================
        consol_lr = self.consolidation_lr_net(slow_output)
        consol_lr = torch.clamp(consol_lr, max=0.01)                  # (batch, 1)

        for i, fast_neuron in enumerate(fast_neurons):
            # Summarize current fast neuron projection state
            down_norm = torch.norm(fast_neuron.W_down_mod, dim=(1, 2))  # (batch,)
            up_norm = torch.norm(fast_neuron.W_up_mod, dim=(1, 2))      # (batch,)
            proj_state = torch.cat([
                down_norm.unsqueeze(1), up_norm.unsqueeze(1)
            ], dim=-1)                                                  # (batch, 2)

            consol_input = torch.cat([slow_output, proj_state], dim=-1)
            consol_raw = self.consolidation_write_nets[i](consol_input)

            # Split into pattern/address pairs (fast neuron dimensions)
            # L2-normalize to prevent explosion (same fix as fast neuron proj writes)
            d_pat  = F.normalize(consol_raw[:, :self.fast_d_model], dim=-1)
            d_addr = F.normalize(consol_raw[:, self.fast_d_model : self.fast_d_model + self.fast_d_proj], dim=-1)
            u_pat  = F.normalize(consol_raw[:, self.fast_d_model + self.fast_d_proj
                                 : self.fast_d_model + 2 * self.fast_d_proj], dim=-1)
            u_addr = F.normalize(consol_raw[:, self.fast_d_model + 2 * self.fast_d_proj :], dim=-1)

            # Rank-1 outer product writes (= not +=)
            fast_neuron.W_down_mod = fast_neuron.W_down_mod + (
                consol_lr.unsqueeze(-1) * torch.bmm(
                    d_pat.unsqueeze(2), d_addr.unsqueeze(1)
                )
            )
            fast_neuron.W_up_mod = fast_neuron.W_up_mod + (
                consol_lr.unsqueeze(-1) * torch.bmm(
                    u_pat.unsqueeze(2), u_addr.unsqueeze(1)
                )
            )

        return new_context
