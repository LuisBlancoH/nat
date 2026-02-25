"""
Tests for Phase 1 training loop — verifies the full episode pipeline
works end-to-end with a tiny model on CPU.

Uses the same tiny Qwen3 model (4 layers, d_model=64) as test_nat_model.py.

Run with: python -m tests.test_phase1
"""

import math
import os
import tempfile

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from model.nat_model import NATv2Model
from training.data import EpisodeDataset, TopicGroup
from training.phase1_adaptation import Phase1Config, compute_loss, train_phase1

# ── Tiny model config ────────────────────────────────────────────────
TINY_D_MODEL = 64
TINY_LAYERS = 4
TINY_VOCAB = 256

BATCH = 2
SEQ_LEN = 128
CHUNK_SIZE = 16
NUM_CHUNKS = SEQ_LEN // CHUNK_SIZE  # 8
NUM_ADAPT = 6


def make_tiny_base_model():
    """Create a tiny Qwen3 model (~0.16M params) for testing."""
    config = AutoConfig.from_pretrained(
        "Qwen/Qwen3-4B", local_files_only=True,
    )
    config.hidden_size = TINY_D_MODEL
    config.num_hidden_layers = TINY_LAYERS
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.intermediate_size = 128
    config.vocab_size = TINY_VOCAB
    config.head_dim = TINY_D_MODEL // 4
    return AutoModelForCausalLM.from_config(config)


def make_model():
    """Create NATv2Model with tiny base model."""
    base = make_tiny_base_model()
    return NATv2Model(
        base_model=base,
        layer_A=1,
        layer_B=2,
        dtype=torch.float32,
    )


def make_dataset():
    """Create a tiny random-token dataset for testing."""
    tokens = torch.randint(0, TINY_VOCAB, (10000,))
    topic = TopicGroup(dataset="test", topic_key="test_topic", tokens=tokens)
    return EpisodeDataset(topics=[topic], seq_len=SEQ_LEN)


# ── Test 1: Full episode pipeline ────────────────────────────────────
def test_episode_pipeline():
    """Run a single episode manually and verify all mechanics."""
    torch.manual_seed(42)
    model = make_model()
    dataset = make_dataset()

    optimizer = torch.optim.AdamW(
        list(model.theta_params()), lr=1e-3, weight_decay=0.01,
    )

    input_ids, domain_idx = dataset.sample_batch(BATCH)
    model.start_episode(BATCH, "cpu")
    optimizer.zero_grad()

    chunk_losses = []
    eval_losses = []

    for chunk_idx in range(NUM_CHUNKS):
        start = chunk_idx * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk_ids = input_ids[:, start:end]

        adapt = chunk_idx < NUM_ADAPT
        outputs = model(chunk_ids, adapt=adapt, chunk_idx=chunk_idx)

        loss = compute_loss(outputs.logits, chunk_ids)
        chunk_losses.append(loss.item())

        if not adapt:
            eval_losses.append(loss)

        del outputs
        if adapt:
            del loss

    eval_loss = torch.stack(eval_losses).mean()
    eval_loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(model.theta_params()), max_norm=1.0,
    )

    optimizer.step()

    # Verify no NaN/Inf
    assert all(
        math.isfinite(l) for l in chunk_losses
    ), f"Non-finite chunk losses: {chunk_losses}"
    assert math.isfinite(
        eval_loss.item()
    ), f"Non-finite eval loss: {eval_loss.item()}"
    assert math.isfinite(
        grad_norm.item()
    ), f"Non-finite grad norm: {grad_norm.item()}"

    # Verify correct number of chunks
    assert len(chunk_losses) == NUM_CHUNKS
    assert len(eval_losses) == NUM_CHUNKS - NUM_ADAPT

    # Verify adaptation benefit is a real number (may be positive or negative)
    benefit = chunk_losses[0] - chunk_losses[-1]
    assert math.isfinite(benefit), f"Non-finite benefit: {benefit}"

    # Verify gradients reached neuron params
    grads_nonzero = sum(
        1 for p in model.theta_params()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    total_theta = sum(1 for _ in model.theta_params())
    assert grads_nonzero > 0, "No gradients reached θ parameters"

    print(
        f"PASS: Test 1 — Episode pipeline: "
        f"eval_loss={eval_loss.item():.4f}, "
        f"benefit={benefit:+.4f}, "
        f"grad_norm={grad_norm.item():.2f}, "
        f"grads={grads_nonzero}/{total_theta}"
    )
    print(
        f"  chunk_losses={[f'{l:.3f}' for l in chunk_losses]}"
    )


# ── Test 2: Multi-episode training ───────────────────────────────────
def test_multi_episode():
    """Run several episodes and verify parameters change."""
    torch.manual_seed(42)
    model = make_model()
    dataset = make_dataset()

    optimizer = torch.optim.AdamW(
        list(model.theta_params()), lr=1e-3, weight_decay=0.01,
    )

    # Record initial params
    initial_params = {
        name: p.data.clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    num_episodes = 5
    losses = []

    for ep in range(num_episodes):
        input_ids, _ = dataset.sample_batch(BATCH)
        model.start_episode(BATCH, "cpu")
        optimizer.zero_grad()

        eval_losses = []
        for chunk_idx in range(NUM_CHUNKS):
            start = chunk_idx * CHUNK_SIZE
            end = start + CHUNK_SIZE
            chunk_ids = input_ids[:, start:end]
            adapt = chunk_idx < NUM_ADAPT

            outputs = model(chunk_ids, adapt=adapt, chunk_idx=chunk_idx)
            loss = compute_loss(outputs.logits, chunk_ids)

            if not adapt:
                eval_losses.append(loss)
            del outputs
            if adapt:
                del loss

        eval_loss = torch.stack(eval_losses).mean()
        eval_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.theta_params()), max_norm=1.0,
        )
        optimizer.step()

        losses.append(eval_loss.item())
        del eval_loss, eval_losses

    # All losses should be finite
    assert all(math.isfinite(l) for l in losses), f"Non-finite losses: {losses}"

    # Parameters should have changed
    changed = 0
    for name, p in model.named_parameters():
        if p.requires_grad and name in initial_params:
            if not torch.equal(p.data, initial_params[name]):
                changed += 1
    assert changed > 0, "No parameters changed during training"

    print(
        f"PASS: Test 2 — Multi-episode: "
        f"{num_episodes} episodes, "
        f"params changed={changed}, "
        f"losses={[f'{l:.4f}' for l in losses]}"
    )


# ── Test 3: train_phase1() integration ───────────────────────────────
def test_train_phase1_integration():
    """Verify the train_phase1 function runs end-to-end."""
    torch.manual_seed(42)
    model = make_model()
    dataset = make_dataset()

    with tempfile.TemporaryDirectory() as tmpdir:
        config = Phase1Config(
            num_episodes=3,
            batch_size=BATCH,
            seq_len=SEQ_LEN,
            chunk_size=CHUNK_SIZE,
            num_adapt_chunks=NUM_ADAPT,
            lr=1e-3,
            warmup_episodes=1,
            log_interval=1,
            frozen_eval_interval=2,
            save_interval=3,
            output_dir=tmpdir,
            use_wandb=False,
            verify_episodes=0,
        )

        train_phase1(config, model=model, dataset=dataset)

        # Verify outputs exist
        assert os.path.exists(os.path.join(tmpdir, "config.json"))
        assert os.path.exists(os.path.join(tmpdir, "train_log.jsonl"))
        assert os.path.exists(
            os.path.join(tmpdir, "checkpoint_3.pt")
        )

        # Verify log contents
        import json
        with open(os.path.join(tmpdir, "train_log.jsonl")) as f:
            lines = f.readlines()

        # Should have training logs + frozen eval logs
        assert len(lines) >= 3, f"Expected >= 3 log lines, got {len(lines)}"

        # Parse and verify a training log entry
        entry = json.loads(lines[0])
        assert "eval_loss" in entry
        assert "adaptation_benefit" in entry
        assert "chunk_losses" in entry
        assert len(entry["chunk_losses"]) == NUM_CHUNKS

    print("PASS: Test 3 — train_phase1() integration")


# ── Test 4: Checkpoint save/load roundtrip ────────────────────────────
def test_checkpoint_roundtrip():
    """Save checkpoint, load into fresh model, verify params match."""
    torch.manual_seed(42)
    model = make_model()
    dataset = make_dataset()

    optimizer = torch.optim.AdamW(
        list(model.theta_params()), lr=1e-3,
    )

    # Train for a few episodes
    for _ in range(3):
        input_ids, _ = dataset.sample_batch(BATCH)
        model.start_episode(BATCH, "cpu")
        optimizer.zero_grad()

        eval_losses = []
        for chunk_idx in range(NUM_CHUNKS):
            start = chunk_idx * CHUNK_SIZE
            end = start + CHUNK_SIZE
            chunk_ids = input_ids[:, start:end]
            adapt = chunk_idx < NUM_ADAPT

            outputs = model(chunk_ids, adapt=adapt, chunk_idx=chunk_idx)
            loss = compute_loss(outputs.logits, chunk_ids)
            if not adapt:
                eval_losses.append(loss)
            del outputs
            if adapt:
                del loss

        eval_loss = torch.stack(eval_losses).mean()
        eval_loss.backward()
        optimizer.step()
        del eval_loss, eval_losses

    # Save checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    saved_state = {
        k: v.clone()
        for k, v in model.state_dict().items()
        if "base_model" not in k
    }

    torch.save(
        {
            "episode": 3,
            "model_state_dict": saved_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "running_eval_loss": 5.0,
            "running_benefit": 0.1,
        },
        ckpt_path,
    )

    # Create fresh model and load checkpoint
    model2 = make_model()
    ckpt = torch.load(ckpt_path, weights_only=False)
    model2.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Verify neuron params match
    for name, p in model.named_parameters():
        if "base_model" not in name and p.requires_grad:
            p2 = dict(model2.named_parameters())[name]
            assert torch.equal(p.data, p2.data), \
                f"Param mismatch after load: {name}"

    os.unlink(ckpt_path)
    print("PASS: Test 4 — Checkpoint save/load roundtrip")


# ── Test 5: BPTT gradient chain across chunks ────────────────────────
def test_bptt_gradient_chain():
    """
    Verify that eval loss gradients flow back through adapt chunk
    state updates (mem_A, W_mod).
    """
    torch.manual_seed(42)
    model = make_model()

    # Force low thresholds so projection write paths are active
    model.fast_neuron_B.fixed_threshold = 0.0

    model.start_episode(BATCH, "cpu")
    model.zero_grad()

    input_ids = torch.randint(0, TINY_VOCAB, (BATCH, SEQ_LEN))

    eval_losses = []
    for chunk_idx in range(NUM_CHUNKS):
        start = chunk_idx * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk_ids = input_ids[:, start:end]
        adapt = chunk_idx < NUM_ADAPT

        outputs = model(chunk_ids, adapt=adapt, chunk_idx=chunk_idx)
        loss = compute_loss(outputs.logits, chunk_ids)

        if not adapt:
            eval_losses.append(loss)
        del outputs
        if adapt:
            del loss

    eval_loss = torch.stack(eval_losses).mean()
    eval_loss.backward()

    # Check that neuron B has gradients in write networks
    # (which only execute during adapt chunks). If BPTT works, eval loss
    # gradients flow back through state tensors to adapt chunk computations.
    neuron = model.fast_neuron_B
    write_grad = False
    for name, p in neuron.named_parameters():
        if name.startswith("write_key_net") or name.startswith(
            "write_value_net"
        ):
            if p.grad is not None and p.grad.abs().sum() > 0:
                write_grad = True
                break
    assert write_grad, \
        "fast_neuron_B: write networks have no gradient (BPTT broken)"

    print("PASS: Test 5 — BPTT gradient chain from eval loss to adapt chunks")


# ── Test 6: EpisodeDataset sampling ──────────────────────────────────
def test_dataset_sampling():
    """Verify EpisodeDataset produces correct shapes."""
    tokens_a = torch.randint(0, 100, (5000,))
    tokens_b = torch.randint(0, 100, (3000,))

    topics = [
        TopicGroup(dataset="ds_a", topic_key="topic_a", tokens=tokens_a),
        TopicGroup(dataset="ds_b", topic_key="topic_b", tokens=tokens_b),
    ]
    dataset = EpisodeDataset(topics=topics, seq_len=128)

    assert dataset.num_topics == 2

    input_ids, topic_indices = dataset.sample_batch(4)
    assert input_ids.shape == (4, 128), f"Shape: {input_ids.shape}"
    assert len(topic_indices) == 4
    assert all(0 <= idx < 2 for idx in topic_indices)

    # All token values should be in range
    assert input_ids.min() >= 0
    assert input_ids.max() < 100

    print(
        f"PASS: Test 6 — Dataset sampling: "
        f"shape={input_ids.shape}, "
        f"topics={topic_indices}"
    )


# ── Test 7: Topic grouping and filtering ─────────────────────────────
def test_topic_grouping():
    """Verify TopicGroup filtering and multi-topic sampling."""
    seq_len = 128
    # Create topics: one large enough, one too small, one exactly at threshold
    topics = [
        TopicGroup(dataset="math", topic_key="Algebra_L1",
                   tokens=torch.randint(0, 100, (5000,))),
        TopicGroup(dataset="math", topic_key="Algebra_L2",
                   tokens=torch.randint(0, 100, (50,))),   # too small
        TopicGroup(dataset="deepmind", topic_key="linear_1d",
                   tokens=torch.randint(0, 100, (128,))),  # exactly seq_len
    ]

    dataset = EpisodeDataset(topics=topics, seq_len=seq_len)

    # Should have filtered out the too-small topic
    assert dataset.num_topics == 2, f"Expected 2 topics, got {dataset.num_topics}"

    # Verify the surviving topics
    surviving_keys = {t.topic_key for t in dataset.topics}
    assert "Algebra_L1" in surviving_keys
    assert "linear_1d" in surviving_keys
    assert "Algebra_L2" not in surviving_keys

    # Sample and verify all samples come from valid topics
    input_ids, topic_indices = dataset.sample_batch(8)
    assert input_ids.shape == (8, seq_len)
    assert all(0 <= idx < 2 for idx in topic_indices)

    # Verify both topics can be sampled (probabilistic but with 8 samples likely)
    unique_topics = set(topic_indices)
    assert len(unique_topics) >= 1  # at least one topic sampled

    print(
        f"PASS: Test 7 — Topic grouping: "
        f"{dataset.num_topics} topics survived, "
        f"sampled from {len(unique_topics)} unique topics"
    )


if __name__ == "__main__":
    test_episode_pipeline()
    test_multi_episode()
    test_train_phase1_integration()
    test_checkpoint_roundtrip()
    test_bptt_gradient_chain()
    test_dataset_sampling()
    test_topic_grouping()
    print("\nAll Phase 1 tests passed!")
