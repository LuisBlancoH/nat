"""
NATModel — Full Nested Adaptive Transformer.

Wraps a frozen pretrained transformer (Qwen2.5 / Llama-family) with two
adaptive memory layers and one consolidation layer inserted between specific
transformer layers.

The forward pass manually iterates through the base model's decoder layers
so that adaptive / consolidation layers can be applied between them.

Supported base architectures
-----------------------------
- Qwen2 / Qwen2.5   (``model.model.layers``, ``model.model.embed_tokens``, etc.)
- Llama / Llama-3.x  (same module structure)
- Mistral            (same module structure)

All three share the ``LlamaForCausalLM``-style layout, so a single
implementation covers them.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
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
# Helper: build causal mask for manual layer-by-layer forward pass     #
# ------------------------------------------------------------------ #

def _make_causal_mask(
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a lower-triangular causal mask.

    Returns shape ``(1, 1, seq_len, seq_len)`` with 0 for attend and
    ``-inf`` for masked positions.  Compatible with HuggingFace's additive
    attention mask convention.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)


# ------------------------------------------------------------------ #
# NATModel                                                             #
# ------------------------------------------------------------------ #

class NATModel(nn.Module):
    """
    Nested Adaptive Transformer.

    Wraps a frozen pretrained causal LM with:
      - Two ``AdaptiveMemoryLayer`` instances inserted after layers A and B.
      - One ``ConsolidationLayer`` inserted after layer C.

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
        # Discover model internals                                         #
        # -------------------------------------------------------------- #
        self._discover_base_model()

        # -------------------------------------------------------------- #
        # Adaptive & Consolidation layers                                  #
        # -------------------------------------------------------------- #
        self.adaptive_A = AdaptiveMemoryLayer(
            self.d_model,
            rank=config.rank,
            d_hidden=config.d_hidden,
            lr_clamp=getattr(config, "lr_clamp", 0.1),
            fast_weight_max_norm=getattr(config, "fast_weight_max_norm", 10.0),
        ).float()

        self.adaptive_B = AdaptiveMemoryLayer(
            self.d_model,
            rank=config.rank,
            d_hidden=config.d_hidden,
            lr_clamp=getattr(config, "lr_clamp", 0.1),
            fast_weight_max_norm=getattr(config, "fast_weight_max_norm", 10.0),
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

        # -------------------------------------------------------------- #
        # Device-specific optimisations                                    #
        # -------------------------------------------------------------- #
        setup_device_optimisations(config)

        # Gradient checkpointing — saves memory on MPS / small GPUs
        self._gradient_checkpointing = getattr(
            config, "gradient_checkpointing", False
        )
        if self._gradient_checkpointing:
            # Enable on the base transformer layers (if supported)
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

        # CUDA AMP autocast for frozen base layers
        self._cuda_amp = (
            getattr(config, "cuda_amp", False) and device_str == "cuda"
        )

        # Periodic cache clearing (MPS)
        self._empty_cache_every = getattr(config, "empty_cache_every", 0)

        logger.info(
            f"NAT model built: {self.num_layers} base layers, "
            f"d_model={self.d_model}, "
            f"inserts at {self.insert_A}/{self.insert_B}/{self.insert_C}, "
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
        Mistral, Gemma, etc.  Also supports a flat mock layout where
        ``embed_tokens``, ``layers``, ``norm`` live directly on the model.
        """
        # --- Transformer backbone ---
        # Most HF causal LMs: model.model  (e.g. Qwen2ForCausalLM.model)
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "layers"):
            self._transformer = self.base_model.model
        # Flat mock models used in tests
        elif hasattr(self.base_model, "layers") and hasattr(self.base_model, "embed_tokens"):
            self._transformer = self.base_model
        else:
            raise ValueError(
                "Cannot find transformer backbone.  Expected "
                "`base_model.model.layers` or `base_model.layers`."
            )

        self._layers = self._transformer.layers
        self.num_layers = len(self._layers)

        # --- Hidden size ---
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

        # --- Rotary embeddings ---
        self._has_rotary = hasattr(self._transformer, "rotary_emb")

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
    # Forward pass                                                         #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Forward pass with adaptive layer intervention.

        We manually iterate through the base model's decoder layers so
        that our adaptive / consolidation layers can be applied between them.

        Parameters
        ----------
        input_ids : LongTensor, shape ``(batch, seq_len)``
        attention_mask : Tensor, shape ``(batch, seq_len)``, optional
            1 = attend, 0 = masked.  If None, assumes all-attend.
        labels : LongTensor, shape ``(batch, seq_len)``, optional
            Shifted internally for next-token prediction.  ``-100`` is ignored.

        Returns
        -------
        dict with keys ``"loss"`` (optional) and ``"logits"``.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialise fast weights if needed
        if self.adaptive_A.fast_A is None:
            self.start_session(batch_size)

        # ---- Embedding ----
        hidden_states = self._transformer.embed_tokens(input_ids)
        base_dtype = hidden_states.dtype

        # ---- Position IDs / Rotary embeddings ----
        cache_position = torch.arange(seq_len, device=device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        position_embeddings = None
        if self._has_rotary:
            position_embeddings = self._transformer.rotary_emb(
                hidden_states, position_ids
            )

        # ---- Causal mask ----
        causal_mask = _make_causal_mask(seq_len, hidden_states.dtype, device)

        # ---- Determine whether to adapt this step ----
        self._step_counter += seq_len
        do_adapt = (self._step_counter % self.adapt_every_n) < seq_len

        # ---- Iterate through decoder layers ----
        # Optionally wrap frozen layers in AMP autocast (CUDA only).
        # Adaptive/consolidation layers always run in float32.
        amp_ctx = (
            torch.amp.autocast(
                "cuda",
                dtype=getattr(self.config, "torch_dtype", torch.bfloat16),
            )
            if self._cuda_amp
            else nullcontext()
        )

        for i, layer in enumerate(self._layers):
            # Build kwargs for this layer
            layer_kwargs: dict[str, Any] = {
                "attention_mask": causal_mask,
                "position_ids": position_ids,
                "use_cache": False,
                "cache_position": cache_position,
            }
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            # Run frozen transformer layer (under AMP if enabled)
            with amp_ctx:
                layer_output = layer(hidden_states, **layer_kwargs)

            # Handle output: some layers return a tensor, some a tuple
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

            # ---- Insert adaptive / consolidation layers ----
            if i == self.insert_A:
                h_float = hidden_states.float()
                h_float = self.adaptive_A(h_float, do_adapt=do_adapt)
                hidden_states = h_float.to(base_dtype)
            elif i == self.insert_B:
                h_float = hidden_states.float()
                h_float = self.adaptive_B(h_float, do_adapt=do_adapt)
                hidden_states = h_float.to(base_dtype)
            elif i == self.insert_C:
                h_float = hidden_states.float()
                h_float = self.consolidation(h_float)
                hidden_states = h_float.to(base_dtype)

        # ---- Final norm ----
        hidden_states = self._transformer.norm(hidden_states)

        # ---- LM head ----
        if self._lm_head is not None:
            logits = self._lm_head(hidden_states).float()
        else:
            logits = hidden_states.float()

        # ---- Loss ----
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}

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
        Verify that the manual forward pass matches the base model's output
        when adaptive layers are disabled (zero fast weights, gate ≈ 0.27).

        Returns dict with ``"max_diff"``, ``"mean_diff"``, ``"passes"``.
        """
        # Base model reference
        ref = self.base_model(input_ids, use_cache=False)
        ref_logits = ref.logits if hasattr(ref, "logits") else ref[0]

        # Our forward
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
