"""
NATModel — Full Nested Adaptive Transformer.

Wraps a frozen pretrained transformer (Qwen3 / Qwen2.5 / Llama-family)
with two adaptive memory layers and one consolidation layer inserted
between specific transformer layers using PyTorch forward hooks.

The forward pass delegates to the base model's own forward method and
uses hooks registered on specific decoder layers to inject adaptive /
consolidation logic.  This avoids replicating model internals (QK-Norm,
RoPE, GQA, KV-cache) and works across architectures.

Supported base architectures
-----------------------------
- Qwen3 / Qwen2 / Qwen2.5
- Llama / Llama-3.x
- Mistral

All share the ``model.model.layers`` layout.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nat.model.adaptive_layer import AdaptiveMemoryLayer
from nat.model.consolidation_layer import ConsolidationLayer
from nat.model.utils import (
    count_parameters,
    print_parameter_summary,
    setup_device_optimisations,
    log_device_memory,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# NATModel                                                             #
# ------------------------------------------------------------------ #

class NATModel(nn.Module):
    """
    Nested Adaptive Transformer.

    Wraps a frozen pretrained causal LM with:
      - Two ``AdaptiveMemoryLayer`` instances hooked after layers A and B.
      - One ``ConsolidationLayer`` hooked after layer C.

    Insertion points default to 1/3, 2/3, 5/6 of the total depth.

    Parameters
    ----------
    config : NATConfig
        Configuration dataclass.  Must have at least:
        ``base_model_name``, ``rank``, ``d_hidden``, ``adapt_every_n``,
        ``beta``, ``session_reset_alpha``, ``lr_clamp``,
        ``fast_weight_max_norm``.
    base_model : nn.Module or None
        If provided, use this as the base model instead of loading from
        ``config.base_model_name``.  Useful for testing with a tiny model.
    tokenizer : optional
        If provided, use this tokenizer.  Otherwise one is loaded.
    """

    def __init__(
        self,
        config,
        base_model: nn.Module | None = None,
        tokenizer=None,
    ):
        super().__init__()
        self.config = config

        # -------------------------------------------------------------- #
        # Base model                                                       #
        # -------------------------------------------------------------- #
        if base_model is not None:
            self.base_model = base_model
        else:
            from transformers import AutoModelForCausalLM
            self.base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                dtype=getattr(config, "torch_dtype", torch.bfloat16),
            )

        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif base_model is not None:
            # Mock / externally-provided model — no tokenizer needed
            self.tokenizer = None
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze ALL base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # -------------------------------------------------------------- #
        # Discover model internals (auto-detect d_model, num_layers)       #
        # -------------------------------------------------------------- #
        self._discover_base_model()

        # -------------------------------------------------------------- #
        # Adaptive & Consolidation layers                                  #
        # -------------------------------------------------------------- #
        self.adaptive_A = AdaptiveMemoryLayer(
            self.d_model,
            rank=config.rank,
            d_hidden=config.d_hidden,
            lr_clamp=getattr(config, "lr_clamp", 0.05),
            fast_weight_max_norm=getattr(config, "fast_weight_max_norm", 8.0),
        ).float()

        self.adaptive_B = AdaptiveMemoryLayer(
            self.d_model,
            rank=config.rank,
            d_hidden=config.d_hidden,
            lr_clamp=getattr(config, "lr_clamp", 0.05),
            fast_weight_max_norm=getattr(config, "fast_weight_max_norm", 8.0),
        ).float()

        self.consolidation = ConsolidationLayer(
            self.d_model,
            rank=config.rank,
            d_hidden=config.d_hidden,
            beta=config.beta,
        ).float()

        # Insertion points (after these layer indices)
        self.insert_A = self.num_layers // 3
        self.insert_B = (2 * self.num_layers) // 3
        self.insert_C = (5 * self.num_layers) // 6

        self.adapt_every_n = config.adapt_every_n
        self._step_counter = 0
        self._do_adapt = True  # toggled by training loop
        self._adapt_cell: list[bool] = [True]  # stable per-forward cell for hooks

        # -------------------------------------------------------------- #
        # Register hooks on base model layers                              #
        # -------------------------------------------------------------- #
        self._hook_handles: list = []
        self._register_hooks()

        # -------------------------------------------------------------- #
        # Device-specific optimisations                                    #
        # -------------------------------------------------------------- #
        setup_device_optimisations(config)

        # Gradient checkpointing
        self._gradient_checkpointing = getattr(
            config, "gradient_checkpointing", False
        )
        if self._gradient_checkpointing:
            if hasattr(self.base_model, "gradient_checkpointing_enable"):
                self.base_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            logger.info("Gradient checkpointing enabled.")

        # torch.compile — fuses kernels on CUDA (A100+)
        self._compile_model = getattr(config, "compile_model", False)
        device_str = getattr(config, "device", "cpu")
        if self._compile_model and device_str == "cuda":
            self.adaptive_A = torch.compile(self.adaptive_A)
            self.adaptive_B = torch.compile(self.adaptive_B)
            self.consolidation = torch.compile(self.consolidation)
            logger.info("torch.compile applied to adaptive & consolidation layers.")

        # Periodic cache clearing (MPS)
        self._empty_cache_every = getattr(config, "empty_cache_every", 0)

        logger.info(
            f"NAT model built (hooks): {self.num_layers} base layers, "
            f"d_model={self.d_model}, "
            f"hooks at {self.insert_A}/{self.insert_B}/{self.insert_C}, "
            f"device={device_str}, "
            f"grad_ckpt={self._gradient_checkpointing}, "
            f"compile={self._compile_model}"
        )
        log_device_memory(config, "after model init")

    # ------------------------------------------------------------------ #
    # Internal: discover base-model structure                              #
    # ------------------------------------------------------------------ #

    def _discover_base_model(self) -> None:
        """
        Find the transformer backbone, decoder layers, embeddings, norm,
        and lm_head regardless of the exact HuggingFace model class.

        Supports the ``model.model.layers`` layout used by Llama, Qwen2,
        Qwen3, Mistral, Gemma, etc.  Also supports a flat mock layout
        where ``embed_tokens``, ``layers``, ``norm`` live directly on
        the model.
        """
        # --- Transformer backbone ---
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"):
            self._transformer = self.base_model.model
        elif hasattr(self.base_model, "layers") and hasattr(self.base_model, "embed_tokens"):
            self._transformer = self.base_model
        else:
            raise ValueError(
                "Cannot find transformer backbone.  Expected "
                "`base_model.model.layers` or `base_model.layers`."
            )

        self._layers = self._transformer.layers
        self.num_layers = len(self._layers)

        # --- Hidden size (auto-detect) ---
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "hidden_size"):
            self.d_model = self.base_model.config.hidden_size
        else:
            # Infer from embedding weight
            self.d_model = self._transformer.embed_tokens.weight.shape[1]

        # --- LM head ---
        if hasattr(self.base_model, "lm_head"):
            self._lm_head = self.base_model.lm_head
        elif hasattr(self.base_model, "output"):
            self._lm_head = self.base_model.output
        else:
            self._lm_head = None  # tests without lm_head

    # ------------------------------------------------------------------ #
    # Hook registration                                                    #
    # ------------------------------------------------------------------ #

    def _register_hooks(self) -> None:
        """
        Register forward hooks on base model layers to inject adaptive
        and consolidation layers.

        Each hook intercepts the layer's output, casts hidden states to
        float32 for numerical stability in the adaptive/consolidation
        layers, and returns the modified output in the original dtype.

        The hook returns the EXACT same tuple format as the original
        layer output.

        NOTE: hooks read ``self._adapt_cell[0]`` rather than
        ``self._do_adapt`` directly.  ``_adapt_cell`` is a 1-element list
        set once at the start of each forward() call and never mutated
        again during that call (including gradient-checkpoint recompute),
        so both the original pass and the recomputation see the same flag.
        """
        layers = self._layers

        def make_adaptive_hook(adaptive_layer):
            def hook(module, input, output):
                # Read from the stable per-forward cell, not self._do_adapt
                do_adapt = self._adapt_cell[0]
                # output is a tuple: (hidden_states, ...) or just a tensor
                if isinstance(output, tuple):
                    hidden = output[0]
                    base_dtype = hidden.dtype
                    h_float = hidden.float()
                    h_float = adaptive_layer(h_float, do_adapt=do_adapt)
                    return (h_float.to(base_dtype),) + output[1:]
                else:
                    base_dtype = output.dtype
                    h_float = output.float()
                    h_float = adaptive_layer(h_float, do_adapt=do_adapt)
                    return h_float.to(base_dtype)
            return hook

        def consolidation_hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                base_dtype = hidden.dtype
                h_float = hidden.float()
                h_float = self.consolidation(h_float)
                return (h_float.to(base_dtype),) + output[1:]
            else:
                base_dtype = output.dtype
                h_float = output.float()
                h_float = self.consolidation(h_float)
                return h_float.to(base_dtype)

        # Register hooks
        h1 = layers[self.insert_A].register_forward_hook(
            make_adaptive_hook(self.adaptive_A)
        )
        h2 = layers[self.insert_B].register_forward_hook(
            make_adaptive_hook(self.adaptive_B)
        )
        h3 = layers[self.insert_C].register_forward_hook(
            consolidation_hook
        )
        self._hook_handles = [h1, h2, h3]

    def remove_hooks(self) -> None:
        """Remove all registered hooks (cleanup)."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    # ------------------------------------------------------------------ #
    # Trainable parameter helpers                                          #
    # ------------------------------------------------------------------ #

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        """Return only parameters that should be trained (θ)."""
        params: list[nn.Parameter] = []
        params.extend(self.adaptive_A.parameters())
        params.extend(self.adaptive_B.parameters())
        params.extend(self.consolidation.parameters())
        return params

    def get_trainable_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Named version — useful for weight-decay filtering."""
        pairs: list[tuple[str, nn.Parameter]] = []
        for prefix, mod in [
            ("adaptive_A", self.adaptive_A),
            ("adaptive_B", self.adaptive_B),
            ("consolidation", self.consolidation),
        ]:
            for name, param in mod.named_parameters():
                pairs.append((f"{prefix}.{name}", param))
        return pairs

    def print_param_summary(self) -> None:
        print_parameter_summary({
            "adaptive_A": self.adaptive_A,
            "adaptive_B": self.adaptive_B,
            "consolidation": self.consolidation,
        })

    # ------------------------------------------------------------------ #
    # Session lifecycle                                                    #
    # ------------------------------------------------------------------ #

    def start_session(self, batch_size: int = 1) -> None:
        """Reset fast weights at the start of a new session / episode."""
        self.adaptive_A.reset_fast_weights(batch_size)
        self.adaptive_B.reset_fast_weights(batch_size)
        self._step_counter = 0

    def end_session(self) -> None:
        """Consolidate learned fast weights, then partial-reset."""
        self.consolidation.consolidate([self.adaptive_A, self.adaptive_B])
        self.adaptive_A.partial_reset(self.config.session_reset_alpha)
        self.adaptive_B.partial_reset(self.config.session_reset_alpha)

    # ------------------------------------------------------------------ #
    # Forward pass (delegates to base model via hooks)                     #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple | None = None,
        use_cache: bool = False,
        labels: torch.Tensor | None = None,
        suppress_adapt: bool = False,
    ) -> dict[str, Any]:
        """
        Forward pass with adaptive layer intervention via hooks.

        The base model's own forward method handles all internals
        (embedding, rotary, attention, KV cache, norm, lm_head).
        Hooks on specific layers inject adaptive/consolidation logic.

        Parameters
        ----------
        input_ids : LongTensor, shape ``(batch, seq_len)``
        attention_mask : Tensor, shape ``(batch, seq_len)``, optional
        position_ids : LongTensor, shape ``(batch, seq_len)``, optional
            Absolute position indices for RoPE.  When forwarding chunks
            of a longer sequence, pass the absolute positions so RoPE
            embeddings are correct even though attention is chunk-local.
        past_key_values : tuple, optional
            KV cache from a previous chunk.  Pass between successive
            chunk forwards to give the model full causal attention over
            the entire sequence processed so far.
        use_cache : bool
            If True, return ``past_key_values`` in the output dict so
            the next chunk can attend to all previous tokens.
        labels : LongTensor, shape ``(batch, seq_len)``, optional
            Shifted internally for next-token prediction.
        suppress_adapt : bool
            If True, force ``do_adapt=False`` so adaptation never fires.

        Returns
        -------
        dict with keys ``"loss"`` (optional), ``"logits"``, and
        ``"past_key_values"`` (when ``use_cache=True``).
        """
        batch_size, seq_len = input_ids.shape

        # Initialise fast weights if needed
        if self.adaptive_A.fast_A is None:
            self.start_session(batch_size)

        # Update step counter & determine whether to adapt.
        # IMPORTANT: compute do_adapt from the counter BEFORE incrementing,
        # then freeze it in _adapt_cell so gradient-checkpoint recomputation
        # sees the exact same flag as the original forward pass.
        do_adapt = (
            not suppress_adapt
            and (self._step_counter % self.adapt_every_n) < seq_len
        )
        self._do_adapt = do_adapt
        self._adapt_cell[0] = do_adapt  # hooks read this, never mutate it

        # Delegate to base model — hooks handle adaptive/consolidation
        base_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "use_cache": use_cache,
        }
        if attention_mask is not None:
            base_kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            base_kwargs["position_ids"] = position_ids
        if past_key_values is not None:
            base_kwargs["past_key_values"] = past_key_values

        output = self.base_model(**base_kwargs)

        # Increment counter AFTER the forward (and after any recompute)
        self._step_counter += seq_len

        # Extract logits
        if hasattr(output, "logits"):
            logits = output.logits.float()
        elif isinstance(output, tuple):
            logits = output[0].float()
        elif isinstance(output, dict):
            logits = output["logits"].float()
        else:
            logits = output.float()

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        result = {"loss": loss, "logits": logits}
        if use_cache:
            pkv = getattr(output, "past_key_values", None)
            if pkv is None and isinstance(output, dict):
                pkv = output.get("past_key_values")
            result["past_key_values"] = pkv
        return result

    # ------------------------------------------------------------------ #
    # Parity check                                                         #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def check_parity(
        self,
        input_ids: torch.Tensor,
        atol: float = 1e-3,
    ) -> dict[str, Any]:
        """
        Verify that the hook-based forward pass matches the base model's
        output when adaptive layers have reset fast weights (gate ≈ 0.007,
        LayerNorm on memory branch only → near-identity).

        Returns dict with ``"max_diff"``, ``"mean_diff"``, ``"passes"``.
        """
        # Remove hooks temporarily for clean base model output
        self.remove_hooks()
        ref = self.base_model(input_ids, use_cache=False)
        ref_logits = ref.logits if hasattr(ref, "logits") else ref[0]
        self._register_hooks()

        # Our forward (with hooks)
        self.start_session(input_ids.shape[0])
        out = self.forward(input_ids)
        nat_logits = out["logits"]

        diff = (ref_logits.float() - nat_logits.float()).abs()
        return {
            "max_diff": diff.max().item(),
            "mean_diff": diff.mean().item(),
            "passes": diff.max().item() < atol,
        }

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    def diagnostics(self) -> dict[str, Any]:
        """Return a dict of diagnostic stats for logging."""
        d: dict[str, Any] = {}
        d.update({f"adaptive_A/{k}": v
                  for k, v in self.adaptive_A.fast_weight_stats().items()})
        d.update({f"adaptive_B/{k}": v
                  for k, v in self.adaptive_B.fast_weight_stats().items()})
        d.update({f"consolidation/{k}": v
                  for k, v in self.consolidation.consolidated_weight_stats().items()})
        d["step_counter"] = self._step_counter
        return d
