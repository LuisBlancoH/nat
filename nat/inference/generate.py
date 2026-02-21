"""
Text generation with self-modification active.

This module provides :func:`generate` — an autoregressive token-by-token
generation loop that runs **through the full NAT forward pass** so that
the adaptive layers continue to self-modify while generating.

Standard HuggingFace ``model.generate()`` cannot be used because it bypasses
our manual layer-by-layer forward pass (which is where the adaptive /
consolidation layers are inserted).  We therefore implement our own
sampling loop here.

Supported decoding strategies
------------------------------
- **Greedy** (``temperature=0`` or ``do_sample=False``)
- **Temperature sampling** (``temperature > 0, do_sample=True``)
- **Top-k sampling** (``top_k > 0``)
- **Top-p (nucleus) sampling** (``top_p < 1.0``)
- **Repetition penalty**
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Generation config                                                    #
# ------------------------------------------------------------------ #

@dataclass
class GenerationConfig:
    """Parameters controlling autoregressive generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1
    stop_tokens: list[int] = field(default_factory=list)
    stop_strings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ #
# Generation result                                                    #
# ------------------------------------------------------------------ #

@dataclass
class GenerationResult:
    """Container for generation output and metadata."""

    text: str
    token_ids: list[int]
    num_tokens_generated: int
    prompt_tokens: int
    stop_reason: str                   # "max_tokens" | "stop_token" | "stop_string" | "eos"
    adaptation_steps: int              # how many times adapt() ran during generation
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"GenerationResult(tokens={self.num_tokens_generated}, "
            f"stop={self.stop_reason!r})"
        )


# ------------------------------------------------------------------ #
# Sampling helpers                                                     #
# ------------------------------------------------------------------ #

def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature (higher → more random)."""
    if temperature <= 0:
        return logits
    return logits / temperature


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero-out everything outside the top-k logits."""
    if k <= 0:
        return logits
    top_k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    threshold = values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling — keep smallest set of tokens with cumulative prob ≥ p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original order
    return sorted_logits.scatter(-1, sorted_indices, sorted_logits)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """Penalise tokens that have already appeared."""
    if penalty == 1.0 or not input_ids:
        return logits
    unique_ids = list(set(input_ids))
    score = logits[..., unique_ids]
    # Reduce score of already-seen tokens
    score = torch.where(score > 0, score / penalty, score * penalty)
    logits[..., unique_ids] = score
    return logits


# ------------------------------------------------------------------ #
# Core generation loop                                                 #
# ------------------------------------------------------------------ #

@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    gen_config: GenerationConfig | None = None,
    tokenizer=None,
    *,
    return_diagnostics: bool = False,
) -> GenerationResult:
    """
    Autoregressive generation with NAT self-modification active.

    Unlike HuggingFace's ``model.generate()``, this loop runs through
    the full NAT forward pass so the adaptive layers continue to learn
    while generating.

    Parameters
    ----------
    model : NATModel
        Must already have a session started (``model.start_session(1)``).
    input_ids : LongTensor, shape ``(1, prompt_len)``
        Tokenised prompt.  Batch size **must be 1** for generation.
    gen_config : GenerationConfig | None
        Generation parameters.  Defaults to ``GenerationConfig()``.
    tokenizer
        Optional tokenizer for stop-string detection and decoding.
    return_diagnostics : bool
        If True, include per-step diagnostics in the result.

    Returns
    -------
    GenerationResult
    """
    if gen_config is None:
        gen_config = GenerationConfig()

    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, (
        f"generate() expects input_ids of shape (1, seq_len), "
        f"got {input_ids.shape}"
    )

    device = input_ids.device
    prompt_len = input_ids.shape[1]

    # Resolve EOS token
    eos_token_id: int | None = None
    if tokenizer is not None and hasattr(tokenizer, "eos_token_id"):
        eos_token_id = tokenizer.eos_token_id

    # Build the set of stop token ids
    stop_ids = set(gen_config.stop_tokens)
    if eos_token_id is not None:
        stop_ids.add(eos_token_id)

    # ----- Process the prompt -----
    # Run the full prompt through the model so adaptive layers learn from it.
    # We process in chunks to handle long prompts.
    chunk_size = getattr(model.config, "seq_len", 1024)
    for start in range(0, prompt_len, chunk_size):
        chunk = input_ids[:, start : start + chunk_size]
        output = model(chunk)

    # The last output gives us logits for the next token
    next_logits = output["logits"][:, -1, :]  # (1, vocab_size)

    # ----- Autoregressive loop -----
    generated_ids: list[int] = []
    all_ids = input_ids[0].tolist()  # flat list for repetition penalty
    step_diagnostics: list[dict] = []
    stop_reason = "max_tokens"
    adaptation_count = 0

    for step in range(gen_config.max_new_tokens):
        logits = next_logits.clone()

        # Apply sampling transforms
        logits = _apply_repetition_penalty(logits, all_ids, gen_config.repetition_penalty)
        logits = _apply_temperature(logits, gen_config.temperature)
        logits = _apply_top_k(logits, gen_config.top_k)
        logits = _apply_top_p(logits, gen_config.top_p)

        # Sample or argmax
        if gen_config.do_sample and gen_config.temperature > 0:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)      # (1, 1)

        token_id = next_token.item()
        generated_ids.append(token_id)
        all_ids.append(token_id)

        # Check stop conditions
        if token_id in stop_ids:
            stop_reason = "eos" if token_id == eos_token_id else "stop_token"
            break

        # Check stop strings
        if gen_config.stop_strings and tokenizer is not None:
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            for stop_str in gen_config.stop_strings:
                if stop_str in generated_text:
                    stop_reason = "stop_string"
                    # Trim the generated text/ids to before the stop string
                    idx = generated_text.index(stop_str)
                    generated_text = generated_text[:idx]
                    # Re-tokenise to get accurate ids (simpler than tracking offsets)
                    generated_ids = tokenizer.encode(
                        generated_text, add_special_tokens=False
                    )
                    break
            if stop_reason == "stop_string":
                break

        # Forward the new token through NAT (adaptive layers learn)
        output = model(next_token)
        next_logits = output["logits"][:, -1, :]

        # Track adaptation steps
        # (The model's internal step counter determines when adapt() fires)
        if return_diagnostics:
            step_diagnostics.append(model.diagnostics())

    # Count total adaptation steps from the model's counter
    adaptation_count = model._step_counter // model.adapt_every_n

    # Decode final text
    if tokenizer is not None:
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    else:
        text = str(generated_ids)

    diag = {}
    if return_diagnostics:
        diag["per_step"] = step_diagnostics
    diag.update(model.diagnostics())

    return GenerationResult(
        text=text,
        token_ids=generated_ids,
        num_tokens_generated=len(generated_ids),
        prompt_tokens=prompt_len,
        stop_reason=stop_reason,
        adaptation_steps=adaptation_count,
        diagnostics=diag,
    )


# ------------------------------------------------------------------ #
# High-level convenience                                               #
# ------------------------------------------------------------------ #

@torch.no_grad()
def generate_text(
    model,
    prompt: str,
    tokenizer,
    gen_config: GenerationConfig | None = None,
    *,
    start_session: bool = True,
) -> str:
    """
    End-to-end convenience: prompt string in → generated string out.

    Optionally starts a new session before generating (so fast weights
    are fresh).  If ``start_session=False``, the model continues from
    its current fast-weight state, which is useful when you've already
    fed context via :func:`SessionManager.feed`.

    Parameters
    ----------
    model : NATModel
    prompt : str
    tokenizer
        HuggingFace tokenizer.
    gen_config : GenerationConfig | None
    start_session : bool
        Whether to call ``model.start_session(1)`` before generation.

    Returns
    -------
    str
        The generated text (prompt NOT included).
    """
    if start_session:
        model.start_session(1)

    device = next(model.parameters()).device
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"].to(device)

    result = generate(
        model,
        input_ids,
        gen_config=gen_config,
        tokenizer=tokenizer,
    )
    return result.text


@torch.no_grad()
def generate_with_context(
    model,
    context: str,
    prompt: str,
    tokenizer,
    gen_config: GenerationConfig | None = None,
) -> GenerationResult:
    """
    Two-phase generation: first feed context (model learns), then generate.

    This is the primary use-case for NAT — the model reads the context,
    its adaptive layers learn from it, and then the generation benefits
    from that in-context learning via fast weights.

    Parameters
    ----------
    model : NATModel
    context : str
        Text for the model to "study" before generating.
    prompt : str
        The prompt to generate from (appended after context processing).
    tokenizer
        HuggingFace tokenizer.
    gen_config : GenerationConfig | None

    Returns
    -------
    GenerationResult
    """
    device = next(model.parameters()).device
    chunk_size = getattr(model.config, "seq_len", 1024)

    # Start a fresh session
    model.start_session(1)

    # Phase 1: feed context in chunks (model adapts fast weights)
    ctx_ids = tokenizer(
        context,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True,
    )["input_ids"].to(device)

    for start in range(0, ctx_ids.shape[1], chunk_size):
        chunk = ctx_ids[:, start : start + chunk_size]
        model(chunk)

    logger.info(f"Context fed: {ctx_ids.shape[1]} tokens")

    # Phase 2: generate from prompt
    prompt_ids = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,  # context already has BOS
    )["input_ids"].to(device)

    result = generate(
        model,
        prompt_ids,
        gen_config=gen_config,
        tokenizer=tokenizer,
    )
    return result
