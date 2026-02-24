"""
Tests for NAT inference module — session management and generation.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nat.model.nat_model import NATModel
from nat.inference.session import SessionManager, SessionInfo
from nat.inference.generate import (
    GenerationConfig,
    GenerationResult,
    generate,
    _apply_temperature,
    _apply_top_k,
    _apply_top_p,
    _apply_repetition_penalty,
)


# ================================================================
# Shared constants & mock model
# ================================================================

D_MODEL = 64
NUM_LAYERS = 6
VOCAB_SIZE = 256
BATCH = 1
SEQ_LEN = 16


class MockLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(d, d)
        self.ff = nn.Linear(d, d)

    def forward(self, x, **kwargs):
        return (self.ff(x),)


class MockTransformer(nn.Module):
    def __init__(self, d: int, n_layers: int, vocab: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([MockLayer(d) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.config = type("C", (), {"hidden_size": d})()

    def forward(self, input_ids, use_cache=False, **kwargs):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            out = layer(hidden)
            hidden = out[0] if isinstance(out, tuple) else out
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return type("Output", (), {"logits": logits})()


@dataclass
class MockConfig:
    base_model_name: str = "mock"
    rank: int = 16
    d_hidden: int = 32
    adapt_every_n: int = 4
    beta: float = 0.99
    session_reset_alpha: float = 0.5
    lr_clamp: float = 0.1
    fast_weight_max_norm: float = 10.0
    seq_len: int = 32
    device: str = "cpu"
    torch_dtype: torch.dtype = torch.float32
    save_dir: str = "/tmp/nat_test_ckpt"


@pytest.fixture
def model():
    mock = MockTransformer(D_MODEL, NUM_LAYERS, VOCAB_SIZE)
    cfg = MockConfig()
    return NATModel(cfg, base_model=mock)


@pytest.fixture
def config():
    return MockConfig()


@pytest.fixture
def input_ids():
    return torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))


# ================================================================
# 1. Sampling helpers
# ================================================================


class TestSamplingHelpers:
    def test_temperature(self):
        logits = torch.randn(1, VOCAB_SIZE)
        out = _apply_temperature(logits.clone(), 0.5)
        assert out.shape == logits.shape
        # temperature=0 should return unchanged logits
        out0 = _apply_temperature(logits.clone(), 0.0)
        assert torch.equal(out0, logits)

    def test_top_k(self):
        logits = torch.randn(1, VOCAB_SIZE)
        out = _apply_top_k(logits.clone(), 10)
        # At most 10 non-inf values
        finite_count = (out > float("-inf")).sum().item()
        assert finite_count <= 10

    def test_top_k_zero_noop(self):
        logits = torch.randn(1, VOCAB_SIZE)
        out = _apply_top_k(logits.clone(), 0)
        assert torch.equal(out, logits)

    def test_top_p(self):
        logits = torch.randn(1, VOCAB_SIZE)
        out = _apply_top_p(logits.clone(), 0.9)
        assert out.shape == logits.shape

    def test_top_p_one_noop(self):
        logits = torch.randn(1, VOCAB_SIZE)
        out = _apply_top_p(logits.clone(), 1.0)
        assert torch.equal(out, logits)

    def test_repetition_penalty(self):
        logits = torch.randn(1, VOCAB_SIZE)
        seen = [1, 2, 3]
        out = _apply_repetition_penalty(logits.clone(), seen, 1.2)
        assert out.shape == logits.shape

    def test_repetition_penalty_one_noop(self):
        logits = torch.randn(1, VOCAB_SIZE)
        out = _apply_repetition_penalty(logits.clone(), [1, 2], 1.0)
        assert torch.equal(out, logits)


# ================================================================
# 2. Generation
# ================================================================


class TestGenerate:
    def test_greedy(self, model, input_ids):
        model.start_session(1)
        cfg = GenerationConfig(max_new_tokens=10, do_sample=False, temperature=0.0)
        result = generate(model, input_ids, gen_config=cfg)
        assert isinstance(result, GenerationResult)
        assert result.num_tokens_generated > 0
        assert result.prompt_tokens == SEQ_LEN
        assert result.stop_reason in ("max_tokens", "eos", "stop_token")

    def test_sampling(self, model, input_ids):
        model.start_session(1)
        cfg = GenerationConfig(
            max_new_tokens=10, temperature=0.8, top_k=20, top_p=0.9, do_sample=True
        )
        result = generate(model, input_ids, gen_config=cfg)
        assert result.num_tokens_generated > 0

    def test_stop_tokens(self, model, input_ids):
        model.start_session(1)
        cfg = GenerationConfig(max_new_tokens=50, do_sample=False, stop_tokens=[5])
        result = generate(model, input_ids, gen_config=cfg)
        # Should complete without error
        assert result.num_tokens_generated > 0

    def test_diagnostics_returned(self, model, input_ids):
        model.start_session(1)
        cfg = GenerationConfig(max_new_tokens=5, do_sample=False)
        result = generate(model, input_ids, gen_config=cfg, return_diagnostics=True)
        assert "per_step" in result.diagnostics
        assert len(result.diagnostics["per_step"]) > 0

    def test_default_config(self, model, input_ids):
        model.start_session(1)
        result = generate(model, input_ids)
        assert result.num_tokens_generated > 0

    def test_result_repr(self, model, input_ids):
        model.start_session(1)
        cfg = GenerationConfig(max_new_tokens=3, do_sample=False)
        result = generate(model, input_ids, gen_config=cfg)
        repr_str = repr(result)
        assert "GenerationResult" in repr_str

    def test_batch_size_assertion(self, model):
        bad_ids = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN))
        model.start_session(2)
        cfg = GenerationConfig(max_new_tokens=3, do_sample=False)
        with pytest.raises(AssertionError):
            generate(model, bad_ids, gen_config=cfg)


# ================================================================
# 3. SessionManager
# ================================================================


class TestSessionManager:
    def test_lifecycle(self, model, config):
        mgr = SessionManager(model, config)
        assert not mgr.session_active
        assert mgr.session_count == 0

        mgr.start_session()
        assert mgr.session_active
        assert mgr.session_count == 1

        info = mgr.end_session()
        assert isinstance(info, SessionInfo)
        assert info.session_id == 1
        assert not mgr.session_active

    def test_double_start_raises(self, model, config):
        mgr = SessionManager(model, config)
        mgr.start_session()
        with pytest.raises(RuntimeError, match="already active"):
            mgr.start_session()

    def test_end_without_start_raises(self, model, config):
        mgr = SessionManager(model, config)
        with pytest.raises(RuntimeError, match="No active session"):
            mgr.end_session()

    def test_feed_tokens(self, model, config):
        mgr = SessionManager(model, config)
        mgr.start_session()
        result = mgr.feed_tokens(torch.randint(0, VOCAB_SIZE, (1, 32)))
        assert result["tokens_processed"] == 32
        assert result["chunks"] >= 1
        info = mgr.end_session()
        assert info.tokens_processed == 32

    def test_feed_tokens_1d(self, model, config):
        """1D input_ids should be auto-unsqueezed."""
        mgr = SessionManager(model, config)
        mgr.start_session()
        result = mgr.feed_tokens(torch.randint(0, VOCAB_SIZE, (32,)))
        assert result["tokens_processed"] == 32
        mgr.end_session()

    def test_feed_without_session_raises(self, model, config):
        mgr = SessionManager(model, config)
        with pytest.raises(RuntimeError, match="No active session"):
            mgr.feed_tokens(torch.randint(0, VOCAB_SIZE, (1, 16)))

    def test_multi_session(self, model, config):
        mgr = SessionManager(model, config)
        for i in range(3):
            mgr.start_session()
            mgr.feed_tokens(torch.randint(0, VOCAB_SIZE, (1, 16)))
            info = mgr.end_session()
            assert info.session_id == i + 1
        assert len(mgr.history) == 3
        assert mgr.session_count == 3

    def test_save_load_consolidated(self, model, config):
        mgr = SessionManager(model, config)
        mgr.start_session()
        mgr.feed_tokens(torch.randint(0, VOCAB_SIZE, (1, 32)))
        mgr.end_session()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "consolidated.pt"
            mgr.save_consolidated(path)
            assert path.exists()
            mgr.load_consolidated(path)

    def test_save_history(self, model, config):
        mgr = SessionManager(model, config)
        mgr.start_session()
        mgr.feed_tokens(torch.randint(0, VOCAB_SIZE, (1, 16)))
        mgr.end_session()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "history.json"
            mgr.save_history(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["session_count"] == 1
            assert len(data["sessions"]) == 1

    def test_diagnostics(self, model, config):
        mgr = SessionManager(model, config)
        d = mgr.diagnostics()
        assert "session_count" in d
        assert "consolidation_empty" in d

    def test_summary(self, model, config):
        mgr = SessionManager(model, config)
        s = mgr.summary()
        assert "SessionManager" in s


# ================================================================
# 4. SessionInfo
# ================================================================


class TestSessionInfo:
    def test_to_dict(self):
        info = SessionInfo(
            session_id=1,
            tokens_processed=100,
            adaptation_steps=10,
            fast_A_norm_start=0.5,
            fast_A_norm_end=1.5,
            fast_B_norm_start=0.3,
            fast_B_norm_end=0.8,
        )
        d = info.to_dict()
        assert d["session_id"] == 1
        assert d["tokens_processed"] == 100
        assert d["adaptation_steps"] == 10
