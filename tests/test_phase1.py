"""
Tests for Phase 1 episodic multi-domain meta-learning.

Uses the same mock base model from test_forward_pass to avoid
downloading real pretrained weights.

Tests cover:
  1. Data module — SyntheticEpisodeDataset / SyntheticEpisodicDataset
  2. Single episodic step — loss is finite, gradients flow
  3. Truncated BPTT — fast weights are detached correctly
  4. compute_episodic_loss — per-problem losses, improvement bonus
  5. Full Phase 1 loop — runs N episodes without error
  6. Checkpoint save / load round-trip
  7. Cosine schedule — learning rate decays
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
    SyntheticEpisodeDataset,
    SyntheticEpisodicDataset,
    build_phase1_dataloader,
    collate_episodic,
)
from nat.training.phase1_episodic import (
    compute_episodic_loss,
    train_one_episodic_step,
    train_phase1,
)
from nat.training.train_utils import (
    save_checkpoint as _save_checkpoint,
    load_checkpoint,
    maybe_truncate as _maybe_truncate,
)

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
class Phase1TestConfig:
    base_model_name: str = "mock"
    rank: int = RANK
    d_hidden: int = D_HIDDEN
    adapt_every_n: int = 4
    beta: float = 0.99
    session_reset_alpha: float = 0.5
    lr_clamp: float = 0.1
    fast_weight_max_norm: float = 10.0
    torch_dtype: torch.dtype = torch.float32
    # Training (Phase 1 = episodic)
    lr: float = 1e-3
    lr_phase1: float = 1e-3
    weight_decay: float = 0.01
    num_episodes_p1: int = 10
    num_problems_per_episode: int = 4
    improvement_weight: float = 0.1
    adapt_problems_p1: int = 2
    batch_size: int = 2
    seq_len: int = 32
    truncated_bptt: int = 0  # no truncation by default
    grad_clip: float = 1.0
    # Logging
    log_every: int = 5
    save_every: int = 100
    save_path: str = "/tmp/nat_test_phase1.pt"
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
    return Phase1TestConfig()


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
# 1. Data module
# ============================================================

class TestDataModule:
    def test_synthetic_dataset_length(self, config):
        ds = SyntheticEpisodeDataset(
            num_episodes=20, seq_len=config.seq_len, vocab_size=VOCAB_SIZE
        )
        assert len(ds) == 20

    def test_synthetic_dataset_item_shape(self, config):
        ds = SyntheticEpisodeDataset(
            num_episodes=5, seq_len=config.seq_len, vocab_size=VOCAB_SIZE
        )
        item = ds[0]
        assert "input_ids" in item
        assert item["input_ids"].shape == (config.seq_len,)
        assert item["input_ids"].dtype == torch.long

    def test_synthetic_dataset_deterministic(self, config):
        ds1 = SyntheticEpisodeDataset(num_episodes=5, seq_len=16, seed=42)
        ds2 = SyntheticEpisodeDataset(num_episodes=5, seq_len=16, seed=42)
        assert torch.equal(ds1[0]["input_ids"], ds2[0]["input_ids"])

    def test_episodic_dataset_has_spans(self, config):
        ds = SyntheticEpisodicDataset(
            num_episodes=5, seq_len=config.seq_len,
            num_problems=config.num_problems_per_episode,
        )
        item = ds[0]
        assert "problem_spans" in item
        assert len(item["problem_spans"]) == config.num_problems_per_episode

    def test_build_dataloader_synthetic(self, config):
        dl = build_phase1_dataloader(config, synthetic=True)
        batch = next(iter(dl))
        assert batch["input_ids"].shape == (config.batch_size, config.seq_len)
        assert "problem_spans" in batch

    def test_dataloader_multiple_batches(self, config):
        config.num_episodes_p1 = 10
        dl = build_phase1_dataloader(config, synthetic=True)
        batches = list(dl)
        # 10 episodes / batch_size 2, drop_last=True → 5 batches
        assert len(batches) == 5


# ============================================================
# 2. Single episodic step
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
        expected_eval = config.num_problems_per_episode - config.adapt_problems_p1
        assert len(metrics["per_problem_losses"]) == expected_eval

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

    def test_gradient_reaches_theta(self, model, dummy_batch, optimizer, config):
        """Verify BPTT reaches slow parameters (fast_A_init)."""
        model.train()
        train_one_episodic_step(
            model,
            dummy_batch["input_ids"],
            dummy_batch["problem_spans"],
            optimizer,
            config,
        )

        # Since optimizer already stepped, verify gradient flow directly
        model.train()
        model.start_session(config.batch_size)
        ids = torch.randint(0, VOCAB_SIZE, (config.batch_size, config.adapt_every_n))
        _ = model(ids)
        ids2 = torch.randint(0, VOCAB_SIZE, (config.batch_size, config.adapt_every_n))
        out = model(ids2, labels=ids2)
        out["loss"].backward()

        assert model.adaptive_A.fast_A_init.grad is not None, \
            "BPTT broken: fast_A_init has no gradient"


# ============================================================
# 3. Truncated BPTT
# ============================================================

class TestTruncatedBPTT:
    def test_truncation_detaches_fast_weights(self, model, config):
        """After truncation, fast_A.grad_fn should be None (detached)."""
        model.train()
        model.start_session(config.batch_size)
        ids = torch.randint(0, VOCAB_SIZE, (config.batch_size, config.adapt_every_n))
        _ = model(ids)

        # Verify fast_A is in graph
        assert model.adaptive_A.fast_A.grad_fn is not None

        # Truncate
        config_tbptt = Phase1TestConfig(truncated_bptt=1)
        _maybe_truncate(model, 1, config_tbptt)

        # Now it should be detached but with requires_grad
        assert model.adaptive_A.fast_A.grad_fn is None
        assert model.adaptive_A.fast_A.requires_grad

    def test_truncation_skip_when_disabled(self, model, config):
        model.train()
        model.start_session(config.batch_size)
        ids = torch.randint(0, VOCAB_SIZE, (config.batch_size, config.adapt_every_n))
        _ = model(ids)

        grad_fn_before = model.adaptive_A.fast_A.grad_fn
        _maybe_truncate(model, 1, config)  # truncated_bptt=0 → disabled
        grad_fn_after = model.adaptive_A.fast_A.grad_fn

        assert grad_fn_before is grad_fn_after, "Truncation fired when disabled"


# ============================================================
# 4. compute_episodic_loss
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

    def test_empty_spans_returns_zero(self):
        logits = torch.randn(BATCH, 16, VOCAB_SIZE)
        ids = torch.randint(0, VOCAB_SIZE, (BATCH, 16))
        loss, per_prob, impr = compute_episodic_loss(logits, ids, [], 0.1)
        assert loss.item() == 0.0
        assert len(per_prob) == 0


# ============================================================
# 5. Full Phase 1 loop
# ============================================================

class TestFullPhase1Loop:
    def test_runs_n_episodes(self, base_model):
        config = Phase1TestConfig(num_episodes_p1=6, log_every=3)
        model = NATModel(config, base_model=base_model, tokenizer=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_path = os.path.join(tmpdir, "phase1.pt")
            result = train_phase1(model, config, synthetic=True)

        assert result["num_episodes_run"] == 6
        assert result["final_loss"] == result["final_loss"]  # not NaN

    def test_no_nans_throughout(self, base_model):
        config = Phase1TestConfig(num_episodes_p1=10, log_every=5)
        model = NATModel(config, base_model=base_model, tokenizer=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_path = os.path.join(tmpdir, "phase1.pt")
            result = train_phase1(model, config, synthetic=True)

        assert result["final_loss"] == result["final_loss"]

    def test_custom_dataloader(self, base_model):
        config = Phase1TestConfig(num_episodes_p1=4, log_every=2)
        model = NATModel(config, base_model=base_model, tokenizer=None)

        ds = SyntheticEpisodicDataset(
            num_episodes=8, seq_len=config.seq_len,
            num_problems=config.num_problems_per_episode,
            vocab_size=VOCAB_SIZE,
        )
        dl = DataLoader(ds, batch_size=config.batch_size,
                        collate_fn=collate_episodic, drop_last=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_path = os.path.join(tmpdir, "phase1.pt")
            result = train_phase1(model, config, dataloader=dl)

        assert result["num_episodes_run"] == 4


# ============================================================
# 6. Checkpoint save / load
# ============================================================

class TestCheckpointing:
    def test_save_creates_file(self, model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.pt")
            _save_checkpoint(model, path, episode_idx=42)
            assert os.path.exists(path)

    def test_save_load_round_trip(self, base_model, config):
        model1 = NATModel(config, base_model=base_model, tokenizer=None)
        model1.start_session(BATCH)

        with torch.no_grad():
            model1.adaptive_A.fast_A_init.fill_(3.14)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            _save_checkpoint(model1, path, episode_idx=99)

            base2 = MockCausalLM(D_MODEL, NUM_LAYERS, NUM_HEADS, VOCAB_SIZE)
            model2 = NATModel(config, base_model=base2, tokenizer=None)
            ep = load_checkpoint(model2, path)

        assert ep == 99
        assert torch.allclose(
            model2.adaptive_A.fast_A_init,
            torch.full_like(model2.adaptive_A.fast_A_init, 3.14),
        )

    def test_checkpoint_during_training(self, base_model):
        config = Phase1TestConfig(
            num_episodes_p1=4,
            save_every=2, log_every=2,
        )
        model = NATModel(config, base_model=base_model, tokenizer=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_path = os.path.join(tmpdir, "phase1.pt")
            train_phase1(model, config, synthetic=True)
            assert os.path.exists(config.save_path)


# ============================================================
# 7. Learning rate schedule
# ============================================================

class TestSchedule:
    def test_cosine_lr_decays(self, base_model):
        config = Phase1TestConfig(
            num_episodes_p1=20,
            log_every=100,
        )
        model = NATModel(config, base_model=base_model, tokenizer=None)
        model.train()

        trainable = model.get_trainable_parameters()
        optimizer = torch.optim.AdamW(trainable, lr=config.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20
        )

        lrs = [scheduler.get_last_lr()[0]]
        for _ in range(20):
            scheduler.step()
            lrs.append(scheduler.get_last_lr()[0])

        assert lrs[10] < lrs[0]


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
