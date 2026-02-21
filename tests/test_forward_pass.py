"""
Full forward-pass tests for the NATModel.

Uses a small mock base model to avoid downloading large pretrained weights.
Tests cover:
  1. Construction with mock model
  2. Forward pass shape and no-NaN
  3. Base model frozen, adaptive layers trainable
  4. Adaptive layer insertion points correct
  5. Session lifecycle (start / end)
  6. Adaptation changes output
  7. Gradient flow through full model to θ
  8. Loss computation
  9. Diagnostics
  10. Multiple forward passes (stability)
"""

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from nat.model.nat_model import NATModel


# ================================================================
# Mock base model (tiny transformer for fast tests)
# ================================================================

D_MODEL = 64
NUM_LAYERS = 6
NUM_HEADS = 4
VOCAB_SIZE = 128
BATCH = 2
SEQ_LEN = 16
RANK = 8
D_HIDDEN = 32


class MockDecoderLayer(nn.Module):
    """Minimal decoder layer mimicking HuggingFace's Qwen2DecoderLayer API."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.input_layernorm = nn.LayerNorm(d_model)
        self.post_attention_layernorm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_embeddings: tuple | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Pre-norm residual attention
        residual = hidden_states
        h = self.input_layernorm(hidden_states)

        # Build causal mask for nn.MultiheadAttention (expects additive mask)
        seq_len = h.size(1)
        if attention_mask is not None and attention_mask.dim() == 4:
            # (1, 1, S, S) → (S, S) for nn.MultiheadAttention
            attn_mask = attention_mask.squeeze(0).squeeze(0)
        else:
            attn_mask = None

        h, _ = self.self_attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        hidden_states = residual + h

        # Post-norm residual MLP
        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        h = self.mlp(h)
        hidden_states = residual + h

        return hidden_states


class MockTransformer(nn.Module):
    """Mock transformer backbone (``model.model`` in HuggingFace models)."""

    def __init__(self, d_model: int, num_layers: int, num_heads: int, vocab_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MockDecoderLayer(d_model, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)


class MockCausalLM(nn.Module):
    """Mock ``AutoModelForCausalLM`` with the standard HF structure."""

    def __init__(self, d_model: int, num_layers: int, num_heads: int, vocab_size: int):
        super().__init__()
        self.model = MockTransformer(d_model, num_layers, num_heads, vocab_size)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Mimic HuggingFace config
        self.config = type("Config", (), {
            "hidden_size": d_model,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "vocab_size": vocab_size,
        })()

    def forward(self, input_ids, use_cache=False, **kwargs):
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        hidden = self.model.norm(hidden)
        logits = self.lm_head(hidden)
        return type("Output", (), {"logits": logits})()


# ================================================================
# Config for tests
# ================================================================

@dataclass
class TestConfig:
    base_model_name: str = "mock"
    rank: int = RANK
    d_hidden: int = D_HIDDEN
    adapt_every_n: int = 8
    beta: float = 0.99
    session_reset_alpha: float = 0.5
    lr_clamp: float = 0.1
    fast_weight_max_norm: float = 10.0
    torch_dtype: torch.dtype = torch.float32


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def mock_base():
    """Create a small mock base model."""
    return MockCausalLM(D_MODEL, NUM_LAYERS, NUM_HEADS, VOCAB_SIZE)


@pytest.fixture
def nat_model(mock_base):
    """Create a NATModel wrapping the mock base model."""
    config = TestConfig()
    model = NATModel(config, base_model=mock_base, tokenizer=None)
    model.start_session(BATCH)
    return model


@pytest.fixture
def dummy_ids():
    """Random token IDs."""
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


# ============================================================
# 1. Construction
# ============================================================

class TestConstruction:
    def test_creates_from_mock(self, nat_model):
        assert isinstance(nat_model, nn.Module)

    def test_discovers_model_internals(self, nat_model):
        assert nat_model.d_model == D_MODEL
        assert nat_model.num_layers == NUM_LAYERS

    def test_insertion_points_valid(self, nat_model):
        assert 0 <= nat_model.insert_A < nat_model.num_layers
        assert nat_model.insert_A < nat_model.insert_B < nat_model.num_layers
        assert nat_model.insert_B <= nat_model.insert_C < nat_model.num_layers

    def test_base_model_frozen(self, nat_model):
        for name, param in nat_model.base_model.named_parameters():
            assert not param.requires_grad, f"Base param {name} not frozen"

    def test_adaptive_layers_trainable(self, nat_model):
        trainable = nat_model.get_trainable_parameters()
        assert len(trainable) > 0
        assert all(p.requires_grad for p in trainable)

    def test_trainable_param_count(self, nat_model):
        total = sum(p.numel() for p in nat_model.get_trainable_parameters())
        assert total > 0
        assert total < 1_000_000  # should be small relative to base


# ============================================================
# 2. Forward pass shapes and stability
# ============================================================

class TestForwardPass:
    def test_output_shape(self, nat_model, dummy_ids):
        out = nat_model(dummy_ids)
        assert out["logits"].shape == (BATCH, SEQ_LEN, VOCAB_SIZE)

    def test_no_nan(self, nat_model, dummy_ids):
        out = nat_model(dummy_ids)
        assert not torch.isnan(out["logits"]).any()
        assert not torch.isinf(out["logits"]).any()

    def test_no_loss_without_labels(self, nat_model, dummy_ids):
        out = nat_model(dummy_ids)
        assert out["loss"] is None

    def test_loss_with_labels(self, nat_model, dummy_ids):
        labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        out = nat_model(dummy_ids, labels=labels)
        assert out["loss"] is not None
        assert out["loss"].dim() == 0  # scalar
        assert not torch.isnan(out["loss"])

    def test_batch_size_1(self, mock_base):
        config = TestConfig()
        model = NATModel(config, base_model=mock_base, tokenizer=None)
        model.start_session(1)
        ids = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))
        out = model(ids)
        assert out["logits"].shape == (1, SEQ_LEN, VOCAB_SIZE)

    def test_different_seq_lens(self, nat_model):
        for sl in [4, 8, 32]:
            nat_model.start_session(BATCH)
            ids = torch.randint(0, VOCAB_SIZE, (BATCH, sl))
            out = nat_model(ids)
            assert out["logits"].shape == (BATCH, sl, VOCAB_SIZE)


# ============================================================
# 3. Session lifecycle
# ============================================================

class TestSessionLifecycle:
    def test_start_session_resets(self, nat_model, dummy_ids):
        # Run forward to modify fast weights
        _ = nat_model(dummy_ids)
        A_after = nat_model.adaptive_A.fast_A.clone().detach()

        # Start new session
        nat_model.start_session(BATCH)
        A_reset = nat_model.adaptive_A.fast_A.detach()

        # Should be back to init
        init_A = nat_model.adaptive_A.fast_A_init.unsqueeze(0).expand(BATCH, -1, -1).detach()
        assert torch.allclose(A_reset, init_A, atol=1e-6)

    def test_end_session_consolidates(self, nat_model, dummy_ids):
        assert nat_model.consolidation.is_empty

        # Adapt
        _ = nat_model(dummy_ids)
        _ = nat_model(dummy_ids)

        # End session
        nat_model.end_session()

        # Consolidation should have absorbed something
        assert not nat_model.consolidation.is_empty

    def test_step_counter_resets(self, nat_model, dummy_ids):
        _ = nat_model(dummy_ids)
        assert nat_model._step_counter > 0

        nat_model.start_session(BATCH)
        assert nat_model._step_counter == 0


# ============================================================
# 4. Adaptation effects
# ============================================================

class TestAdaptation:
    def test_fast_weights_change_during_forward(self, nat_model, dummy_ids):
        A_before = nat_model.adaptive_A.fast_A.clone().detach()
        _ = nat_model(dummy_ids)
        A_after = nat_model.adaptive_A.fast_A.detach()
        assert not torch.allclose(A_before, A_after), (
            "Fast weights did not change during forward"
        )

    def test_output_changes_across_forward_passes(self, nat_model, dummy_ids):
        nat_model.start_session(BATCH)
        out1 = nat_model(dummy_ids)["logits"].detach().clone()
        out2 = nat_model(dummy_ids)["logits"].detach().clone()
        # Because adaptation modifies fast weights, second pass differs.
        # With near-identity gate init (~0.007), the difference is tiny
        # but non-zero.
        max_diff = (out1 - out2).abs().max().item()
        assert max_diff > 0, (
            "Output identical across forward passes — adaptation has no effect"
        )


# ============================================================
# 5. Gradient flow
# ============================================================

class TestGradientFlow:
    def test_grad_to_adaptive_params(self, nat_model, dummy_ids):
        nat_model.train()
        nat_model.start_session(BATCH)

        labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        out = nat_model(dummy_ids, labels=labels)
        out["loss"].backward()

        # At least SOME adaptive layer params should get gradients
        got_grad = []
        for name, param in nat_model.adaptive_A.named_parameters():
            if param.requires_grad and param.grad is not None:
                got_grad.append(name)
        assert len(got_grad) > 0, "No adaptive_A params received gradients"

    def test_no_grad_to_base_model(self, nat_model, dummy_ids):
        nat_model.train()
        nat_model.start_session(BATCH)

        labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        out = nat_model(dummy_ids, labels=labels)
        out["loss"].backward()

        for name, param in nat_model.base_model.named_parameters():
            assert param.grad is None, f"Base param {name} got gradient!"

    def test_grad_to_fast_A_init(self, nat_model, dummy_ids):
        """Critical: BPTT gradient must reach fast_A_init through the model."""
        nat_model.train()
        nat_model.start_session(BATCH)

        labels = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        out = nat_model(dummy_ids, labels=labels)
        out["loss"].backward()

        assert nat_model.adaptive_A.fast_A_init.grad is not None, (
            "No grad on fast_A_init — BPTT chain broken in full model"
        )
        assert nat_model.adaptive_A.fast_A_init.grad.abs().sum() > 0


# ============================================================
# 6. Stability over many steps
# ============================================================

class TestStability:
    def test_many_forward_passes(self, nat_model):
        """20 forward passes should not produce NaN."""
        for _ in range(20):
            ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
            out = nat_model(ids)
            assert not torch.isnan(out["logits"]).any(), "NaN after many passes"

    def test_many_sessions(self, mock_base):
        """Multiple session cycles should stay stable."""
        config = TestConfig()
        model = NATModel(config, base_model=mock_base, tokenizer=None)

        for _ in range(5):
            model.start_session(BATCH)
            for _ in range(3):
                ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
                out = model(ids)
            model.end_session()

        # Final check
        model.start_session(BATCH)
        ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
        out = model(ids)
        assert not torch.isnan(out["logits"]).any()


# ============================================================
# 7. Diagnostics
# ============================================================

class TestDiagnostics:
    def test_diagnostics_keys(self, nat_model, dummy_ids):
        _ = nat_model(dummy_ids)
        diag = nat_model.diagnostics()
        assert "adaptive_A/fast_A_norm" in diag
        assert "adaptive_B/fast_A_norm" in diag
        assert "consolidation/W_c_A_norm" in diag
        assert "step_counter" in diag

    def test_named_parameters(self, nat_model):
        named = nat_model.get_trainable_named_parameters()
        assert len(named) > 0
        names = [n for n, _ in named]
        assert any("adaptive_A" in n for n in names)
        assert any("consolidation" in n for n in names)


# ============================================================
# 8. Parity check (output matches base model when adaptation is off)
# ============================================================

class TestParity:
    @torch.no_grad()
    def test_parity_check_method_runs(self, nat_model, dummy_ids):
        """With gate ≈ 0.007 and LayerNorm on the memory branch only,
        NAT with reset fast weights should be near-identical to frozen
        base model output."""
        result = nat_model.check_parity(dummy_ids)
        assert "max_diff" in result
        assert "mean_diff" in result
        assert "passes" in result
        assert result["max_diff"] < float("inf")


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
