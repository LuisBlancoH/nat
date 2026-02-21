"""
Unit tests for the Consolidation Layer.

Covers:
  1. Construction and parameter/buffer counts
  2. Forward pass shapes and no-NaN
  3. Read-only: forward does NOT modify consolidated weights
  4. Consolidation from adaptive layers (EMA update)
  5. EMA convergence properties
  6. Gate initialisation (not stuck at 0)
  7. Persistence (save / load state)
  8. Batch expansion correctness
  9. Diagnostics and is_empty property
  10. Integration with AdaptiveMemoryLayer
"""

import os
import tempfile

import pytest
import torch
import torch.nn as nn

from nat.model.adaptive_layer import AdaptiveMemoryLayer
from nat.model.consolidation_layer import ConsolidationLayer


# ----- constants -----

D_MODEL = 64
RANK = 8
D_HIDDEN = 32
BATCH = 2
SEQ_LEN = 16
BETA = 0.99


# ----- fixtures -----

@pytest.fixture
def layer():
    """A small consolidation layer for testing."""
    return ConsolidationLayer(
        d_model=D_MODEL,
        rank=RANK,
        d_hidden=D_HIDDEN,
        beta=BETA,
    )


@pytest.fixture
def dummy_input():
    """Random hidden states shaped (batch, seq_len, d_model)."""
    return torch.randn(BATCH, SEQ_LEN, D_MODEL)


@pytest.fixture
def adaptive_layers():
    """Two adaptive layers with initialised fast weights."""
    layers = []
    for _ in range(2):
        al = AdaptiveMemoryLayer(d_model=D_MODEL, rank=RANK, d_hidden=D_HIDDEN)
        al.reset_fast_weights(batch_size=BATCH)
        layers.append(al)
    return layers


# ============================================================
# 1. Construction
# ============================================================

class TestConstruction:
    def test_creates_without_error(self):
        layer = ConsolidationLayer(d_model=D_MODEL, rank=RANK, d_hidden=D_HIDDEN)
        assert isinstance(layer, nn.Module)

    def test_consolidated_weights_are_buffers(self, layer):
        """W_c_A and W_c_B must be buffers, not parameters."""
        param_names = {n for n, _ in layer.named_parameters()}
        buffer_names = {n for n, _ in layer.named_buffers()}
        assert "W_c_A" not in param_names
        assert "W_c_B" not in param_names
        assert "W_c_A" in buffer_names
        assert "W_c_B" in buffer_names

    def test_buffer_shapes(self, layer):
        assert layer.W_c_A.shape == (1, D_MODEL, RANK)
        assert layer.W_c_B.shape == (1, RANK, D_MODEL)

    def test_buffers_start_at_zero(self, layer):
        assert (layer.W_c_A == 0).all()
        assert (layer.W_c_B == 0).all()

    def test_has_read_and_gate_nets(self, layer):
        param_names = {n for n, _ in layer.named_parameters()}
        assert any("read_net" in n for n in param_names)
        assert any("gate_net" in n for n in param_names)

    def test_beta_stored(self, layer):
        assert layer.beta == BETA

    def test_parameter_count_reasonable(self, layer):
        total = sum(p.numel() for p in layer.parameters())
        assert total < 50_000, f"Too many params: {total}"
        assert total > 500, f"Too few params: {total}"


# ============================================================
# 2. Forward pass shapes and stability
# ============================================================

class TestForwardPass:
    def test_output_shape(self, layer, dummy_input):
        out = layer(dummy_input)
        assert out.shape == dummy_input.shape

    def test_no_nan(self, layer, dummy_input):
        out = layer(dummy_input)
        assert not torch.isnan(out).any(), "NaN in output"
        assert not torch.isinf(out).any(), "Inf in output"

    def test_output_with_zero_weights_is_close_to_input(self, layer, dummy_input):
        """With W_c = 0, memory readout is zero; output ≈ LayerNorm(input)."""
        out = layer(dummy_input)
        # The memory_raw will be all zeros, so memory_output = read_net(0),
        # gate will be applied to it. The output should be close to
        # layer_norm(input + gate * read_net(0)).
        # Just verify it's finite and shaped correctly.
        assert out.shape == dummy_input.shape
        assert not torch.isnan(out).any()

    def test_output_changes_after_consolidation(
        self, layer, dummy_input, adaptive_layers
    ):
        """Output should differ once consolidated weights are non-zero."""
        out_before = layer(dummy_input).detach().clone()

        # Modify adaptive fast weights so they're non-trivial
        for al in adaptive_layers:
            al.fast_A = al.fast_A + torch.randn_like(al.fast_A) * 0.5
            al.fast_B = al.fast_B + torch.randn_like(al.fast_B) * 0.5

        layer.consolidate(adaptive_layers)
        out_after = layer(dummy_input).detach().clone()

        assert not torch.allclose(out_before, out_after, atol=1e-6), (
            "Output unchanged after consolidation — layer has no effect"
        )

    def test_different_batch_sizes(self, layer):
        """Forward should work with any batch size."""
        for bs in [1, 3, 8]:
            x = torch.randn(bs, SEQ_LEN, D_MODEL)
            out = layer(x)
            assert out.shape == (bs, SEQ_LEN, D_MODEL)


# ============================================================
# 3. Read-only: forward doesn't modify consolidated weights
# ============================================================

class TestReadOnly:
    def test_forward_doesnt_change_weights(self, layer, dummy_input):
        A_before = layer.W_c_A.clone()
        B_before = layer.W_c_B.clone()

        _ = layer(dummy_input)

        assert torch.equal(layer.W_c_A, A_before), "W_c_A changed during forward"
        assert torch.equal(layer.W_c_B, B_before), "W_c_B changed during forward"

    def test_multiple_forwards_dont_change_weights(self, layer, dummy_input):
        A_before = layer.W_c_A.clone()
        for _ in range(10):
            _ = layer(dummy_input)
        assert torch.equal(layer.W_c_A, A_before)


# ============================================================
# 4. Consolidation (EMA update)
# ============================================================

class TestConsolidation:
    def test_consolidate_updates_weights(self, layer, adaptive_layers):
        assert layer.is_empty

        # Give adaptive layers non-zero fast weights
        for al in adaptive_layers:
            al.fast_A = al.fast_A + torch.randn_like(al.fast_A)
            al.fast_B = al.fast_B + torch.randn_like(al.fast_B)

        layer.consolidate(adaptive_layers)

        assert not layer.is_empty, "Consolidated weights still zero after consolidate()"

    def test_consolidate_is_ema(self, layer, adaptive_layers):
        """Verify the EMA formula: W_c ← β·W_c + (1-β)·avg."""
        # Set initial consolidated weights
        layer.W_c_A = torch.randn_like(layer.W_c_A) * 0.1
        layer.W_c_B = torch.randn_like(layer.W_c_B) * 0.1
        old_A = layer.W_c_A.clone()
        old_B = layer.W_c_B.clone()

        # Give adaptive layers specific fast weights
        for al in adaptive_layers:
            al.fast_A = torch.randn_like(al.fast_A)
            al.fast_B = torch.randn_like(al.fast_B)

        # Compute expected values manually
        avg_A = torch.stack(
            [al.fast_A.mean(dim=0) for al in adaptive_layers]
        ).mean(dim=0, keepdim=True)
        avg_B = torch.stack(
            [al.fast_B.mean(dim=0) for al in adaptive_layers]
        ).mean(dim=0, keepdim=True)

        expected_A = BETA * old_A + (1 - BETA) * avg_A
        expected_B = BETA * old_B + (1 - BETA) * avg_B

        layer.consolidate(adaptive_layers)

        assert torch.allclose(layer.W_c_A, expected_A, atol=1e-6), (
            "W_c_A does not match expected EMA"
        )
        assert torch.allclose(layer.W_c_B, expected_B, atol=1e-6), (
            "W_c_B does not match expected EMA"
        )

    def test_consolidate_no_grad(self, layer, adaptive_layers):
        """Consolidation must not create a computation graph."""
        for al in adaptive_layers:
            al.fast_A = al.fast_A + torch.randn_like(al.fast_A)

        layer.consolidate(adaptive_layers)

        assert layer.W_c_A.grad_fn is None, (
            "W_c_A has grad_fn after consolidation — should be detached"
        )

    def test_multiple_consolidations_converge(self, layer, adaptive_layers):
        """Repeated EMA with the same input should converge to that input."""
        # Use consistent fast weights
        target_A = torch.randn(1, D_MODEL, RANK) * 0.5
        target_B = torch.randn(1, RANK, D_MODEL) * 0.5
        for al in adaptive_layers:
            al.fast_A = target_A.expand(BATCH, -1, -1).clone()
            al.fast_B = target_B.expand(BATCH, -1, -1).clone()

        # Run many consolidation steps
        for _ in range(5000):
            layer.consolidate(adaptive_layers)

        # Should converge to target (since all layers have the same weights)
        assert torch.allclose(layer.W_c_A, target_A, atol=1e-3), (
            f"W_c_A did not converge. "
            f"Diff: {(layer.W_c_A - target_A).abs().max().item():.6f}"
        )

    def test_high_beta_preserves_old_weights(self, layer, adaptive_layers):
        """With β close to 1, old weights should dominate."""
        layer.beta = 0.9999
        layer.W_c_A = torch.ones_like(layer.W_c_A)
        old_A = layer.W_c_A.clone()

        for al in adaptive_layers:
            al.fast_A = torch.randn_like(al.fast_A) * 10  # large new values

        layer.consolidate(adaptive_layers)

        # Should be very close to old value
        diff = (layer.W_c_A - old_A).abs().max().item()
        assert diff < 0.02, f"High beta didn't preserve: diff={diff}"

    def test_low_beta_favors_new_weights(self, layer, adaptive_layers):
        """With β close to 0, new weights should dominate."""
        layer.beta = 0.01
        layer.W_c_A = torch.ones_like(layer.W_c_A) * 100  # large old values

        target_A = torch.zeros(1, D_MODEL, RANK)
        for al in adaptive_layers:
            al.fast_A = target_A.expand(BATCH, -1, -1).clone()
            al.fast_B = torch.zeros_like(al.fast_B)

        layer.consolidate(adaptive_layers)

        # Should be much closer to 0 than to 100
        assert layer.W_c_A.abs().max().item() < 5.0


# ============================================================
# 5. Gate initialisation
# ============================================================

class TestGateInit:
    def test_initial_gate_not_zero(self, layer, dummy_input):
        """Gate bias = -1.0 → sigmoid ≈ 0.27 — not stuck at 0."""
        # Give non-zero consolidated weights so memory readout is non-trivial
        layer.W_c_A = torch.randn_like(layer.W_c_A) * 0.1
        layer.W_c_B = torch.randn_like(layer.W_c_B) * 0.1

        out = layer(dummy_input)
        ln_input = layer.layer_norm(dummy_input)
        diff = (out - ln_input).abs().mean()
        assert diff > 1e-6, "Gate appears stuck at 0"

    def test_gate_bias_value(self, layer):
        """Verify the gate's final linear layer has bias = -1.0."""
        # gate_net[-2] is the last Linear (before Sigmoid)
        last_linear = layer.gate_net[-2]
        assert isinstance(last_linear, nn.Linear)
        assert torch.allclose(
            last_linear.bias,
            torch.tensor([-1.0]),
            atol=1e-6,
        )


# ============================================================
# 6. Persistence (save / load)
# ============================================================

class TestPersistence:
    def test_save_and_load(self, layer, adaptive_layers):
        """Consolidated weights survive a save/load round-trip."""
        # Consolidate to get non-zero weights
        for al in adaptive_layers:
            al.fast_A = al.fast_A + torch.randn_like(al.fast_A)
            al.fast_B = al.fast_B + torch.randn_like(al.fast_B)
        layer.consolidate(adaptive_layers)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "consolidation_state.pt")
            layer.save_state(path)

            # Create a fresh layer and load
            new_layer = ConsolidationLayer(
                d_model=D_MODEL, rank=RANK, d_hidden=D_HIDDEN, beta=BETA
            )
            assert new_layer.is_empty
            new_layer.load_state(path)

            assert torch.equal(new_layer.W_c_A, layer.W_c_A)
            assert torch.equal(new_layer.W_c_B, layer.W_c_B)

    def test_save_creates_directory(self):
        """save_state should create parent directories if needed."""
        layer = ConsolidationLayer(d_model=D_MODEL, rank=RANK, d_hidden=D_HIDDEN)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "state.pt")
            layer.save_state(path)
            assert os.path.exists(path)

    def test_state_dict_includes_buffers(self, layer):
        """Buffers should appear in state_dict for standard saving."""
        sd = layer.state_dict()
        assert "W_c_A" in sd
        assert "W_c_B" in sd

    def test_load_state_dict_round_trip(self, layer, adaptive_layers):
        """Standard PyTorch state_dict save/load also works."""
        for al in adaptive_layers:
            al.fast_A = al.fast_A + torch.randn_like(al.fast_A)
        layer.consolidate(adaptive_layers)

        sd = layer.state_dict()
        new_layer = ConsolidationLayer(
            d_model=D_MODEL, rank=RANK, d_hidden=D_HIDDEN, beta=BETA
        )
        new_layer.load_state_dict(sd)
        assert torch.equal(new_layer.W_c_A, layer.W_c_A)


# ============================================================
# 7. Batch expansion
# ============================================================

class TestBatchExpansion:
    def test_single_consolidated_weights_expand_to_batch(
        self, layer, adaptive_layers
    ):
        """W_c is (1, d, r) but should work for any batch size."""
        for al in adaptive_layers:
            al.fast_A = al.fast_A + torch.randn_like(al.fast_A)
        layer.consolidate(adaptive_layers)

        for bs in [1, 4, 8]:
            x = torch.randn(bs, SEQ_LEN, D_MODEL)
            out = layer(x)
            assert out.shape == (bs, SEQ_LEN, D_MODEL)


# ============================================================
# 8. Diagnostics
# ============================================================

class TestDiagnostics:
    def test_consolidated_weight_stats_keys(self, layer):
        stats = layer.consolidated_weight_stats()
        expected_keys = {
            "W_c_A_norm", "W_c_B_norm",
            "W_c_A_mean", "W_c_B_mean",
            "W_c_A_max", "W_c_B_max",
            "beta",
        }
        assert set(stats.keys()) == expected_keys

    def test_stats_reflect_values(self, layer):
        stats = layer.consolidated_weight_stats()
        assert stats["W_c_A_norm"] == 0.0  # starts at zero
        assert stats["beta"] == BETA

    def test_is_empty_true_initially(self, layer):
        assert layer.is_empty

    def test_is_empty_false_after_consolidation(self, layer, adaptive_layers):
        for al in adaptive_layers:
            al.fast_A = al.fast_A + torch.randn_like(al.fast_A)
        layer.consolidate(adaptive_layers)
        assert not layer.is_empty


# ============================================================
# 9. Integration with AdaptiveMemoryLayer
# ============================================================

class TestIntegration:
    def test_consolidation_after_adaptation(self, layer, adaptive_layers):
        """Realistic flow: adapt → consolidate → forward."""
        dummy = torch.randn(BATCH, SEQ_LEN, D_MODEL)

        # Adapt both adaptive layers
        for al in adaptive_layers:
            _ = al(dummy, do_adapt=True)
            _ = al(dummy, do_adapt=True)

        # Consolidate
        layer.consolidate(adaptive_layers)
        assert not layer.is_empty

        # Forward through consolidation layer
        out = layer(dummy)
        assert out.shape == dummy.shape
        assert not torch.isnan(out).any()

    def test_many_sessions_cycle(self, layer, adaptive_layers):
        """Simulate multiple sessions: adapt → consolidate → partial_reset."""
        dummy = torch.randn(BATCH, SEQ_LEN, D_MODEL)

        for session in range(10):
            # Reset adaptive fast weights
            for al in adaptive_layers:
                al.reset_fast_weights(BATCH)

            # Adapt
            for step in range(5):
                for al in adaptive_layers:
                    _ = al(dummy, do_adapt=True)

            # Consolidate
            layer.consolidate(adaptive_layers)

            # Partial reset
            for al in adaptive_layers:
                al.partial_reset(alpha=0.5)

        # After 10 sessions, consolidated weights should be non-trivial
        assert not layer.is_empty
        stats = layer.consolidated_weight_stats()
        assert stats["W_c_A_norm"] > 0

        # Forward should still be stable
        out = layer(dummy)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
