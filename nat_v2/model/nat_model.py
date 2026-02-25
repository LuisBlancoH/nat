"""
NATv2Model — Nested Adaptive Transformer wrapping frozen Qwen3-4B.

Hooks FastNeuron A at layer 9 and FastNeuron B at layer 18.
SlowNeuron runs separately (not hooked), fired every 16 chunks.

Key v1 lessons applied:
  - Frozen baseline: remove_hooks() / register_hooks() for clean measurement
  - Static bias prevention: FastNeuron early-exits when mem_A is zero
  - Hook casts bf16 hidden states to fp32 for neuron, casts back after
  - No KV cache during training (each chunk independent)

See NAT_v2_Spec.md §Critical Implementation Details.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from model.fast_neuron import FastNeuron
from model.slow_neuron import SlowNeuron


class NATv2Model(nn.Module):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        layer_A: int = 9,
        layer_B: int = 18,
        slow_fire_interval: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        base_model: Optional[nn.Module] = None,
        enable_neuron_A: bool = False,
    ):
        super().__init__()

        # ---- Load frozen base model ----
        if base_model is not None:
            self.base_model = base_model
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype,
            )
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Disable thinking mode for Qwen3 (generation-time setting)
        if hasattr(self.base_model, "generation_config"):
            self.base_model.generation_config.do_sample = False

        self.layer_A = layer_A
        self.layer_B = layer_B
        self.slow_fire_interval = slow_fire_interval
        self.enable_neuron_A = enable_neuron_A

        # ---- Read d_model from base model config ----
        d_model = self.base_model.config.hidden_size

        # ---- Create neurons (fp32 parameters for training precision) ----
        self.fast_neuron_A = FastNeuron(d_model=d_model)
        self.fast_neuron_B = FastNeuron(d_model=d_model)
        num_fast = 2 if enable_neuron_A else 1
        self.slow_neuron = SlowNeuron(
            fast_d_model=d_model,
            fast_d_proj=self.fast_neuron_A.d_proj,
            num_fast_neurons=num_fast,
        )

        # Freeze neuron A params when disabled (saves memory in optimizer)
        if not enable_neuron_A:
            for p in self.fast_neuron_A.parameters():
                p.requires_grad = False

        # ---- Hook management ----
        self._hook_handles = []
        self.register_hooks()

        # ---- Runtime state ----
        self.chunk_counter = 0
        self.slow_neuron_active = False  # Phase 1: inactive

    # ==================================================================
    # Hook management
    # ==================================================================

    def _make_hook(self, neuron):
        """Create a forward hook that runs the neuron on hidden states."""
        def hook_fn(module, input, output):
            # Qwen3 decoder layers may return a plain tensor or a tuple
            if isinstance(output, torch.Tensor):
                h = output                        # (batch, seq, d_model)
                h_new = neuron(h.float())         # fp32 for neuron computation
                return h_new.to(h.dtype)
            else:
                h = output[0]                     # hidden states from tuple
                h_new = neuron(h.float())
                return (h_new.to(h.dtype),) + output[1:]
        return hook_fn

    def register_hooks(self):
        """Register forward hooks at the target layers."""
        if self._hook_handles:
            return  # already registered
        layers = self.base_model.model.layers
        if self.enable_neuron_A:
            self._hook_handles.append(
                layers[self.layer_A].register_forward_hook(
                    self._make_hook(self.fast_neuron_A)
                )
            )
        self._hook_handles.append(
            layers[self.layer_B].register_forward_hook(
                self._make_hook(self.fast_neuron_B)
            )
        )

    def remove_hooks(self):
        """Remove all forward hooks (needed for clean frozen baseline)."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    # ==================================================================
    # Forward pass
    # ==================================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        adapt: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_idx=None,
    ):
        """
        Process one chunk through the base model with adaptive hooks.

        Args:
            input_ids: (batch, seq_len) token ids for this chunk
            adapt: if True, neurons write to memory. If False, read only.
            attention_mask: optional attention mask
            chunk_idx: chunk position in episode (0-based). Writes skipped on chunk 0.

        Returns:
            outputs: CausalLMOutput from the base model
        """
        if self.enable_neuron_A:
            self.fast_neuron_A.adapt_mode = adapt
            self.fast_neuron_A.chunk_idx = chunk_idx
        self.fast_neuron_B.adapt_mode = adapt
        self.fast_neuron_B.chunk_idx = chunk_idx

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        # Collect reports for slow neuron
        report_B = self.fast_neuron_B.last_report
        if self.enable_neuron_A:
            report_A = self.fast_neuron_A.last_report
            if report_A is not None and report_B is not None:
                combined = torch.cat([report_A, report_B], dim=-1)
                self.slow_neuron.accumulate_report(combined)
        elif report_B is not None:
            # Pad with zeros in place of neuron A's report
            zeros_A = torch.zeros_like(report_B)
            combined = torch.cat([zeros_A, report_B], dim=-1)
            self.slow_neuron.accumulate_report(combined)

        # Slow neuron firing
        self.chunk_counter += 1
        if (
            self.slow_neuron_active
            and self.chunk_counter % self.slow_fire_interval == 0
            and len(self.slow_neuron.report_buffer) > 0
        ):
            fast_neurons = [self.fast_neuron_A, self.fast_neuron_B] if self.enable_neuron_A else [self.fast_neuron_B]
            new_context = self.slow_neuron.fire(fast_neurons)
            if self.enable_neuron_A:
                self.fast_neuron_A.context = new_context
            self.fast_neuron_B.context = new_context

        return outputs

    def frozen_baseline_forward(self, input_ids, attention_mask=None):
        """
        Forward pass without hooks — pure frozen Qwen3 output.

        Removes hooks, runs base model, re-registers hooks.
        Used for measuring adaptation benefit.
        """
        self.remove_hooks()
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        self.register_hooks()
        return outputs

    # ==================================================================
    # State management
    # ==================================================================

    def start_window(self, batch_size: int, device: torch.device):
        """Reset fast neuron memory for new window. W_mod, slow neuron, context persist."""
        if self.enable_neuron_A:
            self.fast_neuron_A.start_window(batch_size, device)
        self.fast_neuron_B.start_window(batch_size, device)
        # chunk_counter NOT reset — cumulative for slow neuron firing

    def detach_all_state(self):
        """Detach all persistent state (Phase 2 window boundary)."""
        if self.enable_neuron_A:
            self.fast_neuron_A.detach_state()
        self.fast_neuron_B.detach_state()
        self.slow_neuron.detach_state()

    def start_session(self, batch_size: int, device: torch.device):
        """
        Reset fast neuron state for a new session/window.

        Slow neuron state persists (it accumulates across sessions).
        Used between windows in Phase 2.
        """
        if self.enable_neuron_A:
            self.fast_neuron_A.start_session(batch_size, device)
        self.fast_neuron_B.start_session(batch_size, device)
        self.chunk_counter = 0

    def start_episode(self, batch_size: int, device: torch.device):
        """
        Reset everything including slow neuron for a new training episode.

        Used at the start of each training episode.
        """
        self.start_session(batch_size, device)
        self.slow_neuron.start_session(batch_size, device)

    def save_state(self, path: str):
        """
        Save all per-user state for deployment continuity.

        Saves: fast neuron state (mem_A, W_mod, prev_h_avg, prev_mem_read, context),
        slow neuron state, report buffer, chunk counter.
        """
        state = {}

        for name, neuron in [
            ("fast_A", self.fast_neuron_A),
            ("fast_B", self.fast_neuron_B),
        ]:
            state[name] = {
                "mem_A": neuron.mem_A,
                "W_down_mod": neuron.W_down_mod,
                "W_up_mod": neuron.W_up_mod,
                "prev_h_avg": neuron.prev_h_avg,
                "prev_mem_read": neuron.prev_mem_read,
                "context": neuron.context,
            }

        rb = self.slow_neuron.report_buffer
        state["slow"] = {
            "mem_A": self.slow_neuron.mem_A,
            "W_down_mod": self.slow_neuron.W_down_mod,
            "W_up_mod": self.slow_neuron.W_up_mod,
            "prev_h_avg": self.slow_neuron.prev_h_avg,
            "report_buffer": torch.stack(rb) if rb else torch.empty(0),
        }
        state["chunk_counter"] = self.chunk_counter

        torch.save(state, path)

    def load_state(self, path: str, map_location=None):
        """Load per-user state from file."""
        state = torch.load(path, map_location=map_location, weights_only=True)

        for name, neuron in [
            ("fast_A", self.fast_neuron_A),
            ("fast_B", self.fast_neuron_B),
        ]:
            for attr in ["mem_A", "W_down_mod", "W_up_mod", "prev_h_avg", "prev_mem_read", "context"]:
                setattr(neuron, attr, state[name].get(attr, None))

        slow_s = state["slow"]
        for attr in ["mem_A", "W_down_mod", "W_up_mod", "prev_h_avg"]:
            setattr(self.slow_neuron, attr, slow_s[attr])

        rb = slow_s["report_buffer"]
        if rb.numel() > 0:
            self.slow_neuron.report_buffer = [rb[i] for i in range(rb.shape[0])]
        else:
            self.slow_neuron.report_buffer = []

        self.chunk_counter = state["chunk_counter"]

    # ==================================================================
    # Parameter helpers
    # ==================================================================

    def theta_params(self):
        """Yield all trainable θ parameters (excluding frozen base model)."""
        return (p for p in self.parameters() if p.requires_grad)

    def count_theta_params(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
