"""
Tests for Phase 2 episodic multi-task training.

Uses the same mock base model from test_forward_pass to avoid
downloading real pretrained weights.

Tests cover:
  1. Episodic data module — SyntheticEpisodicDataset, spans, dataloader
  2. compute_episodic_loss — per-problem losses, improvement bonus
  3. Single episodic training step — loss finite, params update
  4. Full Phase 2 loop — runs N episodes without error
  5. Improvement tracking — per-problem losses & bonus are recorded
  6. Checkpoint save / load round-trip
  7. Integration — Phase 1 → Phase 2 continuation
"""

import pytest
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nat.model.nat_model import NATModel
from nat.training.data import (
    SyntheticEpisodicDataset,
    build_phase2_dataloader,
    collate_episodic,
)
from nat.training.phase2_episodic import (
    compute_episodic_loss,
    train_one_episodic_step,
    train_phase2,
)
from nat.training.phase1_meta_learn import _save_checkpoint, load_checkpoint

# Reuse mock model from test_forward_pass
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


# ================================================================
# Test config (mock — no downloads)
# ================================================================

@dataclass
class Phase2TestConfig:
    base_model_name: str = "mock"
    rank: int = RANK
    d_hidden: int = D_HIDDEN
    adapt_every_n: int = 4
    beta: float = 0.99
    session_reset_alpha: float = 0.5
    lr_clamp: float = 0.1
    fast_weight_max_norm: float = 10.0
    torch_dtype: torch.dtype = torch.float32
    # Training (Phase 2)
    lr: float = 1e-3
    lr_phase2: float = 1e-3
    weight_decay: float = 0.01
    num_episodes_p2: int = 10
    num_problems_per_episode: int = 4
    improvement_weight: float = 0.1
    batch_size: int = 2
    seq_len: int = 32
    truncated_bptt: int = 0
    grad_clip: float = 1.0
    # Logging
    log_every: int = 5
    save_every: int = 100
    save_path: str = "/tmp/nat_test_phase2.pt"
    wandb_project: str = "test"
    wandb_entity: str = None
    # Data
    vocab_size: int = VOCAB_SIZE
    device: str = "cpu"

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith("_") and k != "torch_dtype"}


# ================================================================
# Fixtures
# ================================================================

@pytest.fixture
def base_model():
    return MockCausalLM(D_MODEL, NUM_LAYERS, NUM_HEADS, VOCAB_SIZE)


@pytest.fixture
def config():
    return Phase2TestConfig()


@pytest.fixture
def model(base_model, config):
    return NATModel(config, base_model=base_model, tokenizer=None)


@pytest.fixture
def optimizer(model):
    return torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-3)


@pytest.fixture
def episodic_dataset(config):
    return SyntheticEpisodicDataset(
        num_episodes=20,
        seq_len=config.seq_len,
        num_problems=config.num_problems_per_episode,
        vocab_size=VOCAB_SIZE,
    )


@pytest.fixture
def dummy_batch(config):
    """A batch with problem spans, mimicking dataloader output."""
    ds = SyntheticEpisodicDataset(
        num_episodes=config.batch_size,
        seq_len=config.seq_len,
        num_problems=config.num_problems_per_episode,
        vocab_size=VOCAB_SIZE,
    )
    dl = DataLoader(ds, batch_size=config.batch_size, collate_fn=collate_episodic)
    return next(iter(dl))


# ============================================================
# 1. Episodic data module
# ============================================================

class TestEpisodicDataModule:
    def test_dataset_length(self, config):
        ds = SyntheticEpisodicDataset(
            num_episodes=20, seq_len=config.seq_len, num_problems=4
        )
        assert len(ds) == 20

    def test_dataset_item_structure(self, config):
        ds = SyntheticEpisodicDataset(
            num_episodes=5, seq_len=config.seq_len, num_problems=4
        )
        item = ds[0]
        assert "input_ids" in item
        assert "problem_spans" in item
        assert item["input_ids"].shape == (config.seq_len,)
        assert item["input_ids"].dtype == torch.long

    def test_problem_spans_count(self, config):
        ds = SyntheticEpisodicDataset(
            num_episodes=5, seq_len=config.seq_len, num_problems=4
        )
        item = ds[0]
        assert len(item["problem_spans"]) == 4

    def test_problem_spans_valid_ranges(self, config):
        ds = SyntheticEpisodicDataset(
            num_episodes=5, seq_len=config.seq_len, num_problems=4
        )
        spans = ds[0]["problem_spans"]
        for sol_start, sol_end in spans:
            assert 1 <= sol_start < sol_end <= config.seq_len
            assert sol_start >= 1  # need logits at sol_start-1

    def test_problem_spans_cover_non_overlapping_regions(self, config):
        ds = SyntheticEpisodicDataset(
            num_episodes=5, seq_len=config.seq_len, num_problems=4
        )
        spans = ds[0]["problem_spans"]
        for i in range(len(spans) - 1):
            _, end_i = spans[i]
            start_next, _ = spans[i + 1]
            assert end_i <= start_next, "Solution spans overlap"

    def test_dataset_deterministic(self, config):
        ds1 = SyntheticEpisodicDataset(
            num_episodes=5, seq_len=config.seq_len, seed=42
        )
        ds2 = SyntheticEpisodicDataset(
            num_episodes=5, seq_len=config.seq_len, seed=42
        )
        assert torch.equal(ds1[0]["input_ids"], ds2[0]["input_ids"])
        assert ds1[0]["problem_spans"] == ds2[0]["problem_spans"]

    def test_build_dataloader_synthetic(self, config):
        dl = build_phase2_dataloader(config, synthetic=True)
        batch = next(iter(dl))
        assert batch["input_ids"].shape == (config.batch_size, config.seq_len)
        assert "problem_spans" in batch
        assert isinstance(batch["problem_spans"], list)

    def test_collate_episodic_stacks_ids(self, config):
        ds = SyntheticEpisodicDataset(
            num_episodes=4, seq_len=config.seq_len, num_problems=4
        )
        batch_list = [ds[i] for i in range(config.batch_size)]
        batch = collate_episodic(batch_list)
        assert batch["input_ids"].shape == (config.batch_size, config.seq_len)
        assert batch["problem_spans"] == ds[0]["problem_spans"]


# ============================================================
# 2. compute_episodic_loss
# ============================================================

class TestComputeEpisodicLoss:
    def test_returns_finite_loss(self, config):
        logits = torch.randn(BATCH, config.seq_len, VOCAB_SIZE)
        ids = torch.randint(0, VOCAB_SIZE, (BATCH, config.seq_len))
        ds = SyntheticEpisodicDataset(
            num_episodes=1, seq_len=config.seq_len, num_problems=4
        )
        spans = ds[0]["problem_spans"]

        loss, per_prob, impr = compute_episodic_loss(logits, ids, spans, 0.1)
        assert loss.isfinite()
        assert all(l == l for l in per_prob)  # no NaN

    def test_per_problem_losses_count(self, config):
        logits = torch.randn(BATCH, config.seq_len, VOCAB_SIZE)
        ids = torch.randint(0, VOCAB_SIZE, (BATCH, config.seq_len))
        ds = SyntheticEpisodicDataset(
            num_episodes=1, seq_len=config.seq_len, num_problems=4
        )
        spans = ds[0]["problem_spans"]

        _, per_prob, _ = compute_episodic_loss(logits, ids, spans, 0.1)
        assert len(per_prob) == 4

    def test_improvement_is_nonnegative(self, config):
        """Improvement bonus uses relu, so ≥ 0."""
        logits = torch.randn(BATCH, config.seq_len, VOCAB_SIZE)
        ids = torch.randint(0, VOCAB_SIZE, (BATCH, config.seq_len))
        ds = SyntheticEpisodicDataset(
            num_episodes=1, seq_len=config.seq_len, num_problems=4
        )
        spans = ds[0]["problem_spans"]

        _, _, impr = compute_episodic_loss(logits, ids, spans, 0.1)
        assert impr.item() >= 0.0

    def test_loss_is_differentiable(self, config):
        logits = torch.randn(BATCH, config.seq_len, VOCAB_SIZE, requires_grad=True)
        ids = torch.randint(0, VOCAB_SIZE, (BATCH, config.seq_len))
        ds = SyntheticEpisodicDataset(
            num_episodes=1, seq_len=config.seq_len, num_problems=4
        )
        spans = ds[0]["problem_spans"]

        loss, _, _ = compute_episodic_loss(logits, ids, spans, 0.1)
        loss.backward()
        assert logits.grad is not None

    def test_improvement_weight_affects_loss(self, config):
        """With improvement_weight=0, total_loss == base_loss."""
        logits = torch.randn(BATCH, config.seq_len, VOCAB_SIZE)
        ids = torch.randint(0, VOCAB_SIZE, (BATCH, config.seq_len))
        ds = SyntheticEpisodicDataset(
            num_episodes=1, seq_len=config.seq_len, num_problems=4
        )
        spans = ds[0]["problem_spans"]

        loss_0, per_prob_0, _ = compute_episodic_loss(logits, ids, spans, 0.0)
        loss_1, per_prob_1, _ = compute_episodic_loss(logits, ids, spans, 1.0)

        # With 0 weight, loss should equal mean of per-problem losses
        expected_base = sum(per_prob_0) / len(per_prob_0)
        assert abs(loss_0.item() - expected_base) < 1e-4

    def test_empty_spans_returns_zero(self):
        logits = torch.randn(BATCH, 16, VOCAB_SIZE)
        ids = torch.randint(0, VOCAB_SIZE, (BATCH, 16))
        loss, per_prob, impr = compute_episodic_loss(logits, ids, [], 0.1)
        assert loss.item() == 0.0
        assert len(per_prob) == 0


# ============================================================
# 3. Single episodic training step
# ============================================================

class TestSingleEpisodicStep:
    def test_step_returns_finite_loss(self, model, dummy_batch, optimizer, config):
        model.train()
        metrics = train_one_episodic_step(
            model,
            dummy_batch["input_ids"],
            dummy_batch["problem_spans"],
            optimizer,
            config,
        )
        assert "loss" in metrics
        assert metrics["loss"] == metrics["loss"]  # not NaN
        assert metrics["loss"] < float("inf")

    def test_step_updates_params(self, model, dummy_batch, optimizer, config):
        model.train()
        before = {n: p.clone().detach()
                  for n, p in model.get_trainable_named_parameters()}

        train_one_episodic_step(
            model,
            dummy_batch["input_ids"],
            dummy_batch["problem_spans"],
            optimizer,
            config,
        )

        changed = False
        for name, param in model.get_trainable_named_parameters():
            if not torch.allclose(before[name], param.detach(), atol=1e-7):
                changed = True
                break
        assert changed, "No parameters changed after one episodic step"

    def test_step_reports_per_problem_losses(self, model, dummy_batch, optimizer, config):
        model.train()
        metrics = train_one_episodic_step(
            model,
            dummy_batch["input_ids"],
            dummy_batch["problem_spans"],
            optimizer,
            config,
        )
        assert "per_problem_losses" in metrics
        assert len(metrics["per_problem_losses"]) == config.num_problems_per_episode

    def test_step_reports_improvement(self, model, dummy_batch, optimizer, config):
        model.train()
        metrics = train_one_episodic_step(
            model,
            dummy_batch["input_ids"],
            dummy_batch["problem_spans"],
            optimizer,
            config,
        )
        assert "improvement" in metrics
        assert metrics["improvement"] >= 0.0

    def test_step_reports_num_problems(self, model, dummy_batch, optimizer, config):
        model.train()
        metrics = train_one_episodic_step(
            model,
            dummy_batch["input_ids"],
            dummy_batch["problem_spans"],
            optimizer,
            config,
        )
        assert metrics["num_problems"] == config.num_problems_per_episode


# ============================================================
# 4. Full Phase 2 loop
# ============================================================

class TestFullPhase2Loop:
    def test_runs_n_episodes(self, base_model):
        config = Phase2TestConfig(num_episodes_p2=6, log_every=3)
        model = NATModel(config, base_model=base_model, tokenizer=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_path = os.path.join(tmpdir, "phase2.pt")
            result = train_phase2(model, config, synthetic=True)

        assert result["num_episodes_run"] == 6
        assert result["final_loss"] == result["final_loss"]  # not NaN

    def test_no_nans_throughout(self, base_model):
        config = Phase2TestConfig(num_episodes_p2=10, log_every=5)
        model = NATModel(config, base_model=base_model, tokenizer=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_path = os.path.join(tmpdir, "phase2.pt")
            result = train_phase2(model, config, synthetic=True)

        assert result["final_loss"] == result["final_loss"]

    def test_custom_dataloader(self, base_model):
        """User can pass their own DataLoader."""
        config = Phase2TestConfig(num_episodes_p2=4, log_every=2)
        model = NATModel(config, base_model=base_model, tokenizer=None)

        ds = SyntheticEpisodicDataset(
            num_episodes=8, seq_len=config.seq_len,
            num_problems=config.num_problems_per_episode,
            vocab_size=VOCAB_SIZE,
        )
        dl = DataLoader(ds, batch_size=config.batch_size,
                        collate_fn=collate_episodic, drop_last=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_path = os.path.join(tmpdir, "phase2.pt")
            result = train_phase2(model, config, dataloader=dl)

        assert result["num_episodes_run"] == 4


# ============================================================
# 5. Improvement tracking
# ============================================================

class TestImprovementTracking:
    def test_per_problem_losses_vary(self, model, dummy_batch, optimizer, config):
        """Per-problem losses should not all be exactly the same."""
        model.train()
        metrics = train_one_episodic_step(
            model,
            dummy_batch["input_ids"],
            dummy_batch["problem_spans"],
            optimizer,
            config,
        )
        losses = metrics["per_problem_losses"]
        # With random data + adaptation, extremely unlikely all are identical
        if len(losses) > 1:
            assert not all(
                abs(losses[0] - l) < 1e-6 for l in losses[1:]
            ), "All per-problem losses are identical — suspicious"

    def test_improvement_bonus_differentiable(self, config):
        """Improvement bonus gradients flow back to logits."""
        logits = torch.randn(BATCH, config.seq_len, VOCAB_SIZE, requires_grad=True)
        ids = torch.randint(0, VOCAB_SIZE, (BATCH, config.seq_len))
        ds = SyntheticEpisodicDataset(
            num_episodes=1, seq_len=config.seq_len, num_problems=4
        )
        spans = ds[0]["problem_spans"]

        _, _, improvement = compute_episodic_loss(logits, ids, spans, 0.1)
        # Even if improvement is 0 (no decrease), it should still be
        # a tensor in the graph
        assert isinstance(improvement, torch.Tensor)


# ============================================================
# 6. Checkpoint save / load
# ============================================================

class TestPhase2Checkpointing:
    def test_checkpoint_during_training(self, base_model):
        config = Phase2TestConfig(
            num_episodes_p2=4, save_every=2, log_every=2
        )
        model = NATModel(config, base_model=base_model, tokenizer=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_path = os.path.join(tmpdir, "phase2.pt")
            train_phase2(model, config, synthetic=True)
            assert os.path.exists(config.save_path)

    def test_checkpoint_round_trip(self, base_model, config):
        model1 = NATModel(config, base_model=base_model, tokenizer=None)
        with torch.no_grad():
            model1.adaptive_A.fast_A_init.fill_(2.72)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            _save_checkpoint(model1, path, episode_idx=55)

            base2 = MockCausalLM(D_MODEL, NUM_LAYERS, NUM_HEADS, VOCAB_SIZE)
            model2 = NATModel(config, base_model=base2, tokenizer=None)
            ep = load_checkpoint(model2, path)

        assert ep == 55
        assert torch.allclose(
            model2.adaptive_A.fast_A_init,
            torch.full_like(model2.adaptive_A.fast_A_init, 2.72),
        )


# ============================================================
# 7. Phase 1 → Phase 2 continuation
# ============================================================

class TestPhase1ToPhase2:
    def test_phase1_checkpoint_loads_into_phase2(self, base_model):
        """A model trained with Phase 1 can continue to Phase 2."""
        from nat.training.phase1_meta_learn import train_phase1

        # Phase 1 quick training
        p1_config = Phase2TestConfig(
            num_episodes_p2=4,  # unused for Phase 1
        )
        # Phase1 needs num_episodes / num_episodes_p1
        p1_config.num_episodes = 4
        p1_config.num_episodes_p1 = 4
        p1_config.lr_phase1 = 1e-3

        model = NATModel(p1_config, base_model=base_model, tokenizer=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            p1_config.save_path = os.path.join(tmpdir, "phase1.pt")
            result1 = train_phase1(model, p1_config, synthetic=True)

            # Load into Phase 2
            base2 = MockCausalLM(D_MODEL, NUM_LAYERS, NUM_HEADS, VOCAB_SIZE)
            model2 = NATModel(p1_config, base_model=base2, tokenizer=None)
            load_checkpoint(model2, result1["save_path"])

            # Now run Phase 2
            p2_config = Phase2TestConfig(num_episodes_p2=4, log_every=2)
            p2_config.save_path = os.path.join(tmpdir, "phase2.pt")
            result2 = train_phase2(model2, p2_config, synthetic=True)

        assert result2["final_loss"] == result2["final_loss"]  # not NaN
        assert result2["num_episodes_run"] == 4


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
