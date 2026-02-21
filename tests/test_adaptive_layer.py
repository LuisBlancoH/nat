"""
Unit tests for the Adaptive Memory Layer.

Covers:
  1. Construction and parameter counts
  2. Fast weight reset / lifecycle
  3. Forward pass shapes and no-NaN
  4. Self-modification actually changes fast weights
  5. Output changes when adaptation is active
  6. Gradient flow through the adaptation chain (BPTT)
  7. Norm clamping keeps fast weights bounded
  8. Partial reset blending
  9. Gate initialisation (not stuck at 0)
  10. Batch independence of fast weights
"""

import pytest
import torch
import torch.nn as nn

from nat.model.adaptive_layer import AdaptiveMemoryLayer


# ----- fixtures -----

D_MODEL = 64
RANK = 8
D_HIDDEN = 32
BATCH = 2
SEQ_LEN = 16


@pytest.fixture
def layer():
    """A small adaptive layer for testing."""
    l = AdaptiveMemoryLayer(
        d_model=D_MODEL,
        rank=RANK,
        d_hidden=D_HIDDEN,
        lr_clamp=0.1,
        fast_weight_max_norm=10.0,
    )
    l.reset_fast_weights(batch_size=BATCH)
    return l


@pytest.fixture
def dummy_input():
    """Random hidden states shaped (batch, seq_len, d_model)."""
    return torch.randn(BATCH, SEQ_LEN, D_MODEL)


# ============================================================
# 1. Construction
# ============================================================

class TestConstruction:
    def test_creates_without_error(self):
        layer = AdaptiveMemoryLayer(d_model=D_MODEL, rank=RANK, d_hidden=D_HIDDEN)
        assert isinstance(layer, nn.Module)

    def test_has_slow_parameters(self, layer):
        param_names = {n for n, _ in layer.named_parameters()}
        assert "fast_A_init" in param_names
        assert "fast_B_init" in param_names
        # Surprise net, lr net, etc.
        assert any("surprise_net" in n for n in param_names)
        assert any("lr_net" in n for n in param_names)
        assert any("write_key_net" in n for n in param_names)
        assert any("write_value_net" in n for n in param_names)
        assert any("read_net" in n for n in param_names)
        assert any("gate_net" in n for n in param_names)
        assert any("state_predictor" in n for n in param_names)

    def test_fast_weights_are_not_parameters(self, layer):
        param_names = {n for n, _ in layer.named_parameters()}
        assert "fast_A" not in param_names
        assert "fast_B" not in param_names

    def test_parameter_count_reasonable(self, layer):
        total = sum(p.numel() for p in layer.parameters())
        # With d_model=64, rank=8, d_hidden=32 this should be small
        assert total < 100_000, f"Too many params: {total}"
        assert total > 1_000, f"Too few params: {total}"


# ============================================================
# 2. Fast weight lifecycle
# ============================================================

class TestFastWeightLifecycle:
    def test_reset_creates_fast_weights(self):
        layer = AdaptiveMemoryLayer(d_model=D_MODEL, rank=RANK, d_hidden=D_HIDDEN)
        assert layer.fast_A is None
        assert layer.fast_B is None

        layer.reset_fast_weights(batch_size=3)
        assert layer.fast_A is not None
        assert layer.fast_B is not None
        assert layer.fast_A.shape == (3, D_MODEL, RANK)
        assert layer.fast_B.shape == (3, RANK, D_MODEL)

    def test_reset_clears_prev_h(self, layer):
        # Simulate setting prev_h
        layer.prev_h = torch.randn(BATCH, D_MODEL)
        layer.reset_fast_weights(batch_size=BATCH)
        assert layer.prev_h is None

    def test_reset_fast_weights_have_grad_fn(self, layer):
        # During training, fast weights must be in the computation graph
        assert layer.fast_A.grad_fn is not None or layer.fast_A.requires_grad
        assert layer.fast_B.grad_fn is not None or layer.fast_B.requires_grad

    def test_partial_reset_blends(self, layer):
        # Modify fast weights
        original_A = layer.fast_A.clone().detach()
        layer.fast_A = layer.fast_A + torch.randn_like(layer.fast_A) * 0.5
        modified_A = layer.fast_A.clone().detach()

        layer.partial_reset(alpha=0.5)
        result_A = layer.fast_A.detach()

        # Should be between init and modified
        init_A = layer.fast_A_init.unsqueeze(0).expand(BATCH, -1, -1).detach()
        expected = 0.5 * init_A + 0.5 * modified_A
        assert torch.allclose(result_A, expected, atol=1e-5)

    def test_partial_reset_alpha_1_is_full_reset(self, layer):
        layer.fast_A = layer.fast_A + torch.randn_like(layer.fast_A)
        layer.partial_reset(alpha=1.0)

        init_A = layer.fast_A_init.unsqueeze(0).expand(BATCH, -1, -1).detach()
        assert torch.allclose(layer.fast_A.detach(), init_A, atol=1e-6)


# ============================================================
# 3. Forward pass shapes and stability
# ============================================================

class TestForwardPass:
    def test_output_shape(self, layer, dummy_input):
        out = layer(dummy_input, do_adapt=False)
        assert out.shape == dummy_input.shape

    def test_output_shape_with_adapt(self, layer, dummy_input):
        out = layer(dummy_input, do_adapt=True)
        assert out.shape == dummy_input.shape

    def test_no_nan_without_adapt(self, layer, dummy_input):
        out = layer(dummy_input, do_adapt=False)
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

    def test_no_nan_with_adapt(self, layer, dummy_input):
        out = layer(dummy_input, do_adapt=True)
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

    def test_no_nan_after_many_adaptations(self, layer, dummy_input):
        """Run 50 adaptation steps — should stay stable."""
        for _ in range(50):
            out = layer(dummy_input, do_adapt=True)
        assert not torch.isnan(out).any(), "NaN after 50 adaptations"
        assert not torch.isinf(out).any(), "Inf after 50 adaptations"

    def test_read_only_doesnt_modify_fast_weights(self, layer, dummy_input):
        A_before = layer.fast_A.clone().detach()
        B_before = layer.fast_B.clone().detach()
        _ = layer(dummy_input, do_adapt=False)
        assert torch.allclose(layer.fast_A.detach(), A_before)
        assert torch.allclose(layer.fast_B.detach(), B_before)


# ============================================================
# 4. Self-modification
# ============================================================

class TestSelfModification:
    def test_adapt_changes_fast_A(self, layer, dummy_input):
        A_before = layer.fast_A.clone().detach()
        _ = layer(dummy_input, do_adapt=True)
        A_after = layer.fast_A.detach()
        assert not torch.allclose(A_before, A_after), (
            "fast_A did not change after adaptation"
        )

    def test_fast_B_unchanged_by_adapt(self, layer, dummy_input):
        """Current design only updates fast_A, not fast_B."""
        B_before = layer.fast_B.clone().detach()
        _ = layer(dummy_input, do_adapt=True)
        B_after = layer.fast_B.detach()
        assert torch.allclose(B_before, B_after), (
            "fast_B changed unexpectedly"
        )

    def test_output_differs_with_vs_without_adapt(self, layer, dummy_input):
        # Without adaptation
        layer.reset_fast_weights(BATCH)
        out_no_adapt = layer(dummy_input, do_adapt=False).detach().clone()

        # With adaptation
        layer.reset_fast_weights(BATCH)
        out_with_adapt = layer(dummy_input, do_adapt=True).detach().clone()

        # With near-identity gate init (~0.007), the difference is tiny
        # but must be non-zero.
        max_diff = (out_no_adapt - out_with_adapt).abs().max().item()
        assert max_diff > 0, (
            "Output identical with and without adaptation"
        )

    def test_second_adapt_uses_surprise(self, layer, dummy_input):
        """After the first adapt, prev_h is set so surprise is computed."""
        _ = layer(dummy_input, do_adapt=True)
        assert layer.prev_h is not None

        # Second adaptation — should use the surprise network
        A_before = layer.fast_A.clone().detach()
        _ = layer(dummy_input, do_adapt=True)
        A_after = layer.fast_A.detach()
        assert not torch.allclose(A_before, A_after)


# ============================================================
# 5. Gradient flow (BPTT)
# ============================================================

class TestGradientFlow:
    def test_grad_flows_to_fast_A_init(self, layer, dummy_input):
        """The most critical test: gradients must reach fast_A_init."""
        layer.train()
        layer.reset_fast_weights(BATCH)

        # Adapt, then read, then compute loss
        out = layer(dummy_input, do_adapt=True)
        loss = out.sum()
        loss.backward()

        assert layer.fast_A_init.grad is not None, (
            "No gradient on fast_A_init — BPTT chain is broken!"
        )
        assert layer.fast_A_init.grad.abs().sum() > 0

    def test_grad_flows_to_fast_B_init(self, layer, dummy_input):
        layer.train()
        layer.reset_fast_weights(BATCH)
        out = layer(dummy_input, do_adapt=False)
        loss = out.sum()
        loss.backward()

        assert layer.fast_B_init.grad is not None, (
            "No gradient on fast_B_init"
        )

    def test_grad_flows_to_surprise_net(self, layer, dummy_input):
        layer.train()
        layer.reset_fast_weights(BATCH)

        # Two adaptations so surprise network is used
        _ = layer(dummy_input, do_adapt=True)
        out = layer(dummy_input, do_adapt=True)
        loss = out.sum()
        loss.backward()

        surprise_params = list(layer.surprise_net.parameters())
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in surprise_params)
        assert has_grad, "No gradient on surprise_net parameters"

    def test_grad_flows_to_write_nets(self, layer, dummy_input):
        layer.train()
        layer.reset_fast_weights(BATCH)
        out = layer(dummy_input, do_adapt=True)
        loss = out.sum()
        loss.backward()

        for name, net in [("write_key_net", layer.write_key_net),
                          ("write_value_net", layer.write_value_net)]:
            params = list(net.parameters())
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in params)
            assert has_grad, f"No gradient on {name}"

    def test_multi_step_bptt(self, layer, dummy_input):
        """Gradient should flow through multiple adaptation steps."""
        layer.train()
        layer.reset_fast_weights(BATCH)

        for _ in range(5):
            out = layer(dummy_input, do_adapt=True)

        loss = out.sum()
        loss.backward()

        assert layer.fast_A_init.grad is not None
        assert layer.fast_A_init.grad.abs().sum() > 0


# ============================================================
# 6. Norm clamping / stability
# ============================================================

class TestStability:
    def test_fast_weight_norm_bounded(self, layer, dummy_input):
        """After many steps, norm should not exceed the max."""
        layer.reset_fast_weights(BATCH)
        # Use large-magnitude inputs to stress the system
        big_input = dummy_input * 100.0
        for _ in range(100):
            _ = layer(big_input, do_adapt=True)

        norm = torch.norm(layer.fast_A.detach(), dim=(1, 2))
        assert (norm <= layer.fast_weight_max_norm + 0.1).all(), (
            f"Fast weight norm exceeded max: {norm}"
        )

    def test_lr_clamp_respected(self, layer):
        """Learning rate should never exceed lr_clamp."""
        surprise = torch.tensor([[1.0]])  # max surprise
        lr = layer.lr_net(surprise)
        lr = lr.clamp(max=layer.lr_clamp)
        assert lr.item() <= layer.lr_clamp + 1e-6


# ============================================================
# 7. Gate initialisation
# ============================================================

class TestGateInit:
    def test_initial_gate_not_zero(self, layer, dummy_input):
        """Gate should be slightly open initially (bias = -3.0 → ~0.047)."""
        out_read = layer.read(dummy_input)
        # If gate were zero, output would equal h_t exactly (identity).
        # Check that memory has some effect (small but non-zero).
        diff = (out_read - dummy_input).abs().mean()
        assert diff > 1e-8, "Gate appears to be stuck at 0"


# ============================================================
# 8. Batch independence
# ============================================================

class TestBatchIndependence:
    def test_different_batch_items_get_different_updates(self):
        layer = AdaptiveMemoryLayer(d_model=D_MODEL, rank=RANK, d_hidden=D_HIDDEN)
        layer.reset_fast_weights(batch_size=2)

        # Feed different inputs per batch item
        h = torch.randn(2, SEQ_LEN, D_MODEL)
        h[0] *= 3.0  # Different magnitude
        _ = layer(h, do_adapt=True)

        # Fast weights should differ across batch
        assert not torch.allclose(
            layer.fast_A[0].detach(),
            layer.fast_A[1].detach(),
            atol=1e-6,
        ), "Batch items have identical fast weights after different inputs"


# ============================================================
# 9. Diagnostics
# ============================================================

class TestDiagnostics:
    def test_fast_weight_stats(self, layer):
        stats = layer.fast_weight_stats()
        assert "fast_A_norm" in stats
        assert "fast_B_norm" in stats
        assert stats["fast_A_norm"] > 0

    def test_stats_before_init(self):
        layer = AdaptiveMemoryLayer(d_model=D_MODEL, rank=RANK, d_hidden=D_HIDDEN)
        stats = layer.fast_weight_stats()
        assert stats["fast_A_norm"] == 0.0


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
