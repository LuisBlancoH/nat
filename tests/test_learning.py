"""
Tests for end-to-end learning through the NATModel.

Verifies that:
  1. Outer-loop training (SGD on θ) reduces loss over steps
  2. BPTT through the adaptive self-modification graph is functional
  3. Session lifecycle (start → forward → end → consolidate) integrates properly
  4. Consolidation actually contributes to later sessions
"""

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass

from nat.model.nat_model import NATModel


# ================================================================
# Reuse mock base model from test_forward_pass
# ================================================================

from tests.test_forward_pass import (
    MockCausalLM,
    D_MODEL,
    NUM_LAYERS,
    NUM_HEADS,
    VOCAB_SIZE,
    BATCH,
    SEQ_LEN,
    RANK,
    D_HIDDEN,
)


@dataclass
class TestConfig:
    base_model_name: str = "mock"
    rank: int = RANK
    d_hidden: int = D_HIDDEN
    adapt_every_n: int = 4
    beta: float = 0.99
    session_reset_alpha: float = 0.5
    lr_clamp: float = 0.1
    fast_weight_max_norm: float = 10.0
    torch_dtype: torch.dtype = torch.float32


@pytest.fixture
def base_model():
    return MockCausalLM(D_MODEL, NUM_LAYERS, NUM_HEADS, VOCAB_SIZE)


@pytest.fixture
def make_model(base_model):
    """Factory fixture so each test gets a fresh NATModel."""
    def _make():
        config = TestConfig()
        model = NATModel(config, base_model=base_model, tokenizer=None)
        return model
    return _make


@pytest.fixture
def target_ids():
    """Fixed target so loss is comparable across steps."""
    torch.manual_seed(42)
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


@pytest.fixture
def input_ids():
    torch.manual_seed(123)
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


# ============================================================
# 1. Outer-loop training reduces loss
# ============================================================

class TestOuterLoopTraining:
    def test_loss_decreases_with_training(self, make_model, input_ids, target_ids):
        """Train for a few steps and verify loss goes down."""
        model = make_model()
        model.train()

        trainable = model.get_trainable_parameters()
        optimizer = torch.optim.Adam(trainable, lr=1e-3)

        losses = []
        for step in range(10):
            model.start_session(BATCH)
            optimizer.zero_grad()

            # Run two forward passes (adaptation + use)
            _ = model(input_ids)  # adapt
            out = model(input_ids, labels=target_ids)  # evaluate

            out["loss"].backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            losses.append(out["loss"].item())

        # Loss should trend down — compare first 3 avg to last 3 avg
        early = sum(losses[:3]) / 3
        late = sum(losses[-3:]) / 3
        assert late < early, (
            f"Loss did not decrease: early avg {early:.4f} → late avg {late:.4f}"
        )

    def test_optimizer_updates_adaptive_params(self, make_model, input_ids, target_ids):
        """After one optimizer step, adaptive params should change."""
        model = make_model()
        model.train()
        model.start_session(BATCH)

        trainable = model.get_trainable_parameters()
        optimizer = torch.optim.SGD(trainable, lr=0.01)

        before = {n: p.clone().detach() for n, p in model.get_trainable_named_parameters()}

        optimizer.zero_grad()
        out = model(input_ids, labels=target_ids)
        out["loss"].backward()
        optimizer.step()

        changed_any = False
        for name, param in model.get_trainable_named_parameters():
            if not torch.allclose(before[name], param.detach(), atol=1e-7):
                changed_any = True
                break

        assert changed_any, "No trainable parameter changed after optimizer step"


# ============================================================
# 2. BPTT through self-modification
# ============================================================

class TestBPTT:
    def test_grad_magnitude_nonzero(self, make_model, input_ids, target_ids):
        """Gradients should have nonzero magnitude, showing BPTT works."""
        model = make_model()
        model.train()
        model.start_session(BATCH)

        out = model(input_ids, labels=target_ids)
        out["loss"].backward()

        grad_norms = {}
        for name, param in model.get_trainable_named_parameters():
            if param.grad is not None:
                grad_norms[name] = param.grad.norm().item()

        assert len(grad_norms) > 0, "No gradients at all"
        assert any(v > 1e-8 for v in grad_norms.values()), (
            f"All gradient norms near zero: {grad_norms}"
        )

    def test_second_forward_gradient_depends_on_first(self, make_model, input_ids, target_ids):
        """If we do two forward passes, loss from second pass should
        produce gradients that reflect the self-modification from the first."""
        model = make_model()
        model.train()
        model.start_session(BATCH)

        # First forward: self-modify
        _ = model(input_ids)

        # Second forward: use adapted weights
        out = model(input_ids, labels=target_ids)
        out["loss"].backward()

        # fast_A_init is now a frozen zero buffer; check a real slow param
        grad = model.adaptive_A.write_key_net[0].weight.grad
        assert grad is not None, "write_key_net has no gradient after 2-pass BPTT"
        assert grad.abs().sum() > 0


# ============================================================
# 3. Consolidation integration
# ============================================================

class TestConsolidationIntegration:
    def test_consolidation_absorbs_after_session(self, make_model, input_ids):
        """After end_session the fast weights should have been partially reset."""
        model = make_model()
        model.eval()

        model.start_session(BATCH)
        for _ in range(5):
            _ = model(input_ids)

        # Capture fast weights before reset
        A_before = model.adaptive_A.fast_A.detach().clone()

        model.end_session()

        # After partial reset, fast weights should have moved toward init
        A_after = model.adaptive_A.fast_A.detach()
        init_A = model.adaptive_A.fast_A_init.unsqueeze(0).expand_as(A_after).detach()
        dist_before = (A_before - init_A).norm().item()
        dist_after = (A_after - init_A).norm().item()
        assert dist_after < dist_before, "end_session partial reset did not move fast weights toward init"

    def test_consolidation_affects_output(self, make_model, input_ids):
        """Fast-weight adaptation changes the output within a session."""
        model = make_model()
        model.eval()

        model.start_session(BATCH)
        out_before = model(input_ids)["logits"].detach().clone()

        for _ in range(20):
            _ = model(input_ids)

        out_after = model(input_ids)["logits"].detach().clone()
        max_diff = (out_before - out_after).abs().max().item()
        assert max_diff > 1e-5, (
            f"Adaptation had no effect on output within session "
            f"(max_diff={max_diff:.2e})"
        )

    def test_multiple_consolidation_cycles(self, make_model, input_ids):
        """Running several consolidation cycles should remain stable."""
        model = make_model()
        model.eval()

        for cycle in range(5):
            model.start_session(BATCH)
            for _ in range(3):
                _ = model(input_ids)
            model.end_session()

        # Final check
        model.start_session(BATCH)
        out = model(input_ids)
        assert not torch.isnan(out["logits"]).any(), (
            "NaN after multiple consolidation cycles"
        )


# ============================================================
# 4. Full training loop with consolidation
# ============================================================

class TestFullTrainingLoop:
    def test_train_with_session_cycling(self, make_model, input_ids, target_ids):
        """Simulate realistic training: multiple sessions with consolidation.

        Each training step needs its own fresh session (start_session) so the
        autograd graph is clean — partial_reset detaches, so we cannot backward
        through multiple steps in a single session without retain_graph.
        """
        model = make_model()
        model.train()

        trainable = model.get_trainable_parameters()
        optimizer = torch.optim.Adam(trainable, lr=1e-3)

        all_losses = []

        for session in range(4):
            for step in range(3):
                model.start_session(BATCH)
                optimizer.zero_grad()
                out = model(input_ids, labels=target_ids)
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                all_losses.append(out["loss"].item())
            model.end_session()

        # Should have run without errors
        assert len(all_losses) == 12
        # No NaN losses
        assert all(not (l != l) for l in all_losses), f"NaN loss: {all_losses}"


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
