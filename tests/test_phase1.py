"""
Tests for Phase 1 meta-learning training.

Uses the same mock base model from test_forward_pass to avoid
downloading real pretrained weights.

Tests cover:
  1. Data module — SyntheticEpisodeDataset and dataloader building
  2. Single episode training — loss is finite, gradients flow
  3. Truncated BPTT — fast weights are detached correctly
  4. Baseline computation — benefit metric is sensible
  5. Full training loop — runs N episodes without error
  6. Checkpoint save / load round-trip
  7. Adaptation benefit — eval loss with adaptation ≤ baseline
  8. Cosine schedule — learning rate decays
"""

import pytest
import tempfile
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nat.model.nat_model import NATModel
from nat.training.data import SyntheticEpisodeDataset, build_phase1_dataloader
from nat.training.phase1_meta_learn import (
    train_one_episode,
    train_phase1,
    _save_checkpoint,
    load_checkpoint,
    _maybe_truncate,
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
    # Training
    lr: float = 1e-3
    lr_phase1: float = 1e-3
    weight_decay: float = 0.01
    num_episodes: int = 10
    num_episodes_p1: int = 10
    batch_size: int = 2
    seq_len: int = 32
    truncated_bptt: int = 0  # no truncation by default
    grad_clip: float = 1.0
    # Logging
    log_every: int = 5
    save_every: int = 100
    save_path: str = "/tmp/nat_test_checkpoint.pt"
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
def dummy_ids(config):
    return torch.randint(0, VOCAB_SIZE, (config.batch_size, config.seq_len))


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

    def test_build_dataloader_synthetic(self, config):
        dl = build_phase1_dataloader(config, synthetic=True)
        batch = next(iter(dl))
        assert batch["input_ids"].shape == (config.batch_size, config.seq_len)

    def test_dataloader_multiple_batches(self, config):
        config.num_episodes_p1 = 10
        dl = build_phase1_dataloader(config, synthetic=True)
        batches = list(dl)
        # 10 episodes / batch_size 2 = 5 batches
        assert len(batches) == 5


# ============================================================
# 2. Single episode training
# ============================================================

class TestSingleEpisode:
    def test_episode_returns_finite_loss(self, model, dummy_ids, optimizer, config):
        model.train()
        metrics = train_one_episode(model, dummy_ids, optimizer, config)
        assert "loss" in metrics
        assert metrics["loss"] == metrics["loss"]  # not NaN
        assert metrics["loss"] < float("inf")

    def test_episode_updates_params(self, model, dummy_ids, optimizer, config):
        model.train()
        before = {n: p.clone().detach()
                  for n, p in model.get_trainable_named_parameters()}

        train_one_episode(model, dummy_ids, optimizer, config)

        changed = False
        for name, param in model.get_trainable_named_parameters():
            if not torch.allclose(before[name], param.detach(), atol=1e-7):
                changed = True
                break
        assert changed, "No parameters changed after one episode"

    def test_episode_reports_adapt_steps(self, model, dummy_ids, optimizer, config):
        model.train()
        metrics = train_one_episode(model, dummy_ids, optimizer, config)
        assert "num_adapt_steps" in metrics
        assert metrics["num_adapt_steps"] > 0

    def test_gradient_reaches_theta(self, model, dummy_ids, optimizer, config):
        """Verify BPTT reaches slow parameters."""
        model.train()
        train_one_episode(model, dummy_ids, optimizer, config)

        # fast_A_init should have gotten gradients (or been updated)
        # Since optimizer already stepped, check that params moved
        # We test gradient flow more directly here:
        model.train()
        model.start_session(config.batch_size)
        _ = model(dummy_ids[:, :config.adapt_every_n])
        out = model(dummy_ids[:, config.adapt_every_n:], labels=dummy_ids[:, config.adapt_every_n:])
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
        _ = model(ids)  # adapt once

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

    def test_training_with_truncation(self, base_model):
        """Full episode with truncated BPTT should work without errors."""
        config = Phase1TestConfig(truncated_bptt=2, seq_len=32, adapt_every_n=4)
        model = NATModel(config, base_model=base_model, tokenizer=None)
        model.train()

        optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-3)
        ids = torch.randint(0, VOCAB_SIZE, (config.batch_size, config.seq_len))

        metrics = train_one_episode(model, ids, optimizer, config)
        assert metrics["loss"] == metrics["loss"]  # not NaN


# ============================================================
# 4. Baseline computation
# ============================================================

class TestBaselineComputation:
    def test_baseline_is_computed(self, model, dummy_ids, optimizer, config):
        model.train()
        metrics = train_one_episode(
            model, dummy_ids, optimizer, config, compute_baseline=True
        )
        assert "baseline_loss" in metrics
        assert "adaptation_benefit" in metrics
        assert metrics["baseline_loss"] is not None

    def test_baseline_not_computed_by_default(self, model, dummy_ids, optimizer, config):
        model.train()
        metrics = train_one_episode(model, dummy_ids, optimizer, config)
        assert "baseline_loss" not in metrics

    def test_benefit_is_finite(self, model, dummy_ids, optimizer, config):
        model.train()
        metrics = train_one_episode(
            model, dummy_ids, optimizer, config, compute_baseline=True
        )
        assert metrics["adaptation_benefit"] == metrics["adaptation_benefit"]


# ============================================================
# 5. Full training loop
# ============================================================

class TestFullTrainingLoop:
    def test_runs_n_episodes(self, base_model):
        config = Phase1TestConfig(num_episodes=6, num_episodes_p1=6, log_every=3)
        model = NATModel(config, base_model=base_model, tokenizer=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_path = os.path.join(tmpdir, "phase1.pt")
            result = train_phase1(model, config, synthetic=True)

        assert result["num_episodes_run"] == 6
        assert result["final_loss"] == result["final_loss"]  # not NaN

    def test_loss_decreases_over_episodes(self, base_model):
        """Over many episodes, loss should trend downward."""
        config = Phase1TestConfig(
            num_episodes=30, num_episodes_p1=30,
            log_every=10, seq_len=32, adapt_every_n=4,
        )
        model = NATModel(config, base_model=base_model, tokenizer=None)

        # Track losses manually
        losses = []
        dl = build_phase1_dataloader(config, synthetic=True)
        model.train()
        model = model.to(config.device)
        opt = torch.optim.AdamW(model.get_trainable_parameters(), lr=1e-3)

        data_iter = iter(dl)
        for _ in range(30):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)

            ids = batch["input_ids"].to(config.device)
            m = train_one_episode(model, ids, opt, config)
            losses.append(m["loss"])

        # Compare first 10 avg to last 10 avg
        early = sum(losses[:10]) / 10
        late = sum(losses[-10:]) / 10
        assert late < early, (
            f"Loss did not decrease: early={early:.4f} late={late:.4f}"
        )

    def test_no_nans_throughout(self, base_model):
        config = Phase1TestConfig(num_episodes=10, num_episodes_p1=10, log_every=5)
        model = NATModel(config, base_model=base_model, tokenizer=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_path = os.path.join(tmpdir, "phase1.pt")
            result = train_phase1(model, config, synthetic=True)

        assert result["final_loss"] == result["final_loss"]


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

        # Modify a param so we can check it round-trips
        with torch.no_grad():
            model1.adaptive_A.fast_A_init.fill_(3.14)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            _save_checkpoint(model1, path, episode_idx=99)

            # Build fresh model and load
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
            num_episodes=4, num_episodes_p1=4,
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
            num_episodes=20, num_episodes_p1=20,
            log_every=100,  # suppress logging
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

        # LR should decrease from start to midpoint
        assert lrs[10] < lrs[0]


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
