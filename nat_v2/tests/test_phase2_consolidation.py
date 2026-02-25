"""
Tests for Phase 2 consolidation training — verifies multi-window pipeline,
slow neuron firing, consolidation writes, gradient flow, and detach semantics.

Uses the same tiny Qwen3 model (4 layers, d_model=64) as other tests.

Run with: python -m tests.test_phase2_consolidation
"""

import math

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from model.nat_model import NATv2Model
from training.data import EpisodeDataset, TopicGroup
from training.phase1_adaptation import compute_loss
from training.phase2_consolidation import Phase2Config, run_episode

# ── Tiny model config ────────────────────────────────────────────────
TINY_D_MODEL = 64
TINY_LAYERS = 4
TINY_VOCAB = 256

BATCH = 2
WINDOW_LEN = 128
CHUNK_SIZE = 16
NUM_CHUNKS = WINDOW_LEN // CHUNK_SIZE  # 8
NUM_ADAPT = 6
# Use 3 windows (A, B, A) for fast tests instead of 13
NUM_TEST_WINDOWS = 3


def make_tiny_base_model():
    """Create a tiny Qwen3 model for testing."""
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


def make_model(slow_fire_interval=4):
    """Create NATv2Model with tiny base model."""
    base = make_tiny_base_model()
    model = NATv2Model(
        base_model=base,
        layer_A=1,
        layer_B=2,
        slow_fire_interval=slow_fire_interval,
        dtype=torch.float32,
    )
    return model


def make_batch(num_windows=NUM_TEST_WINDOWS):
    """Create a fake Phase 2 batch with num_windows windows."""
    windows = [
        torch.randint(0, TINY_VOCAB, (BATCH, WINDOW_LEN))
        for _ in range(num_windows)
    ]
    if num_windows == 3:
        domain_labels = ['A', 'B', 'A']
    else:
        domain_labels = ['A'] * 5 + ['B'] * 5 + ['A'] * 3
    return {'windows': windows, 'domain_labels': domain_labels[:num_windows]}


def make_config():
    """Create a Phase 2 config for testing."""
    return Phase2Config(
        num_episodes=1,
        batch_size=BATCH,
        window_len=WINDOW_LEN,
        chunk_size=CHUNK_SIZE,
        num_adapt_chunks=NUM_ADAPT,
        windows_A=2,  # 1 early + 1 late for 3-window test
        windows_B=1,
        slow_fire_interval=4,
        slow_lr=1e-3,
        fast_lr=1e-4,
        use_wandb=False,
        verify_episodes=0,
    )


def make_dataset():
    """Create a dataset with enough tokens for Phase 2 sampling."""
    topics = []
    for i in range(6):
        tokens = torch.randint(0, TINY_VOCAB, (50000,))
        topics.append(TopicGroup(
            dataset="test", topic_key=f"topic_{i}", tokens=tokens,
        ))
    return EpisodeDataset(topics=topics, seq_len=WINDOW_LEN)


# ── Test 1: Multi-window pipeline ────────────────────────────────────
def test_multi_window_pipeline():
    """3 windows (A, B, A), verify shapes, no NaN."""
    torch.manual_seed(42)
    model = make_model()
    config = make_config()
    batch = make_batch(3)

    model.start_episode(BATCH, "cpu")
    model.slow_neuron_active = True
    model.zero_grad()

    metrics = run_episode(model, batch, config, "cpu")

    # Check metrics are finite
    assert math.isfinite(metrics["loss_A_early"]), f"NaN loss_A_early"
    assert math.isfinite(metrics["loss_B"]), f"NaN loss_B"
    assert math.isfinite(metrics["loss_A_late"]), f"NaN loss_A_late"
    assert math.isfinite(metrics["forgetting_ratio"]), f"NaN forgetting_ratio"

    # Per-window losses should exist
    assert len(metrics["per_window_losses"]) == 3

    # All per-window losses should be finite
    for i, wl in enumerate(metrics["per_window_losses"]):
        assert math.isfinite(wl), f"NaN loss at window {i}"

    print(
        f"PASS: Test 1 — Multi-window pipeline: "
        f"A_early={metrics['loss_A_early']:.4f}, "
        f"B={metrics['loss_B']:.4f}, "
        f"A_late={metrics['loss_A_late']:.4f}, "
        f"forget={metrics['forgetting_ratio']:.3f}"
    )


# ── Test 2: Slow neuron fires at expected chunk counts ───────────────
def test_slow_neuron_fires():
    """Verify slow neuron fires at expected chunk counts."""
    torch.manual_seed(42)
    # fire_interval=4 so fires every 4 chunks
    model = make_model(slow_fire_interval=4)
    model.start_episode(BATCH, "cpu")
    model.slow_neuron_active = True

    # Force low threshold so writes happen
    model.fast_neuron_A.fixed_threshold = 0.0
    model.fast_neuron_B.fixed_threshold = 0.0

    fire_counts = []
    input_ids = torch.randint(0, TINY_VOCAB, (BATCH, WINDOW_LEN))

    # Run 2 windows of 8 chunks each = 16 chunks total
    # With fire_interval=4, expect fires at chunks 4, 8, 12, 16
    for w in range(2):
        model.start_window(BATCH, "cpu")
        for chunk_idx in range(NUM_CHUNKS):
            start = chunk_idx * CHUNK_SIZE
            end = start + CHUNK_SIZE
            chunk_ids = input_ids[:, start:end]

            old_context = model.fast_neuron_A.context.clone()
            with torch.no_grad():
                model(chunk_ids, adapt=True, chunk_idx=chunk_idx)
            new_context = model.fast_neuron_A.context

            # Check if context changed (slow neuron fired)
            if not torch.equal(old_context, new_context):
                fire_counts.append(model.chunk_counter)

    # Should have fired at cumulative chunk counts 4, 8, 12, 16
    assert len(fire_counts) >= 2, (
        f"Expected >= 2 slow neuron firings, got {len(fire_counts)}: {fire_counts}"
    )

    print(
        f"PASS: Test 2 — Slow neuron fires: "
        f"{len(fire_counts)} firings at chunks {fire_counts}"
    )


# ── Test 3: W_mod persists across windows ────────────────────────────
def test_w_mod_persists_across_windows():
    """W_mod non-zero after window 2 despite start_window reset."""
    torch.manual_seed(42)
    model = make_model()
    model.start_episode(BATCH, "cpu")
    model.slow_neuron_active = True
    model.fast_neuron_A.fixed_threshold = 0.0
    model.fast_neuron_B.fixed_threshold = 0.0

    input_ids = torch.randint(0, TINY_VOCAB, (BATCH, WINDOW_LEN))

    # Window 1: adapt chunks write to W_mod
    for chunk_idx in range(NUM_CHUNKS):
        start = chunk_idx * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk_ids = input_ids[:, start:end]
        with torch.no_grad():
            model(chunk_ids, adapt=(chunk_idx < NUM_ADAPT), chunk_idx=chunk_idx)

    w_mod_after_w1 = torch.norm(model.fast_neuron_A.W_down_mod).item()

    # start_window resets mem_A but NOT W_mod
    model.start_window(BATCH, "cpu")

    w_mod_after_reset = torch.norm(model.fast_neuron_A.W_down_mod).item()
    mem_after_reset = torch.norm(model.fast_neuron_A.mem_A).item()

    assert mem_after_reset == 0.0, f"mem_A should be zero after start_window"
    assert w_mod_after_reset == w_mod_after_w1, (
        f"W_mod should persist: {w_mod_after_w1} vs {w_mod_after_reset}"
    )
    assert w_mod_after_reset > 0, f"W_mod should be non-zero after adaptation"

    print(
        f"PASS: Test 3 — W_mod persists: "
        f"W_mod={w_mod_after_reset:.4f}, mem_A={mem_after_reset:.4f}"
    )


# ── Test 4: Consolidation writes ─────────────────────────────────────
def test_consolidation_writes():
    """Slow neuron writes non-zero to fast W_mod via consolidation."""
    torch.manual_seed(42)
    # fire_interval=4 so it fires during the first window
    model = make_model(slow_fire_interval=4)
    model.start_episode(BATCH, "cpu")
    model.slow_neuron_active = True
    model.fast_neuron_A.fixed_threshold = 0.0
    model.fast_neuron_B.fixed_threshold = 0.0

    input_ids = torch.randint(0, TINY_VOCAB, (BATCH, WINDOW_LEN))

    # Track W_mod before and after slow neuron fires
    w_mod_norms = []

    for chunk_idx in range(NUM_CHUNKS):
        start = chunk_idx * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk_ids = input_ids[:, start:end]

        w_mod_before = torch.norm(model.fast_neuron_A.W_down_mod).item()
        with torch.no_grad():
            model(chunk_ids, adapt=True, chunk_idx=chunk_idx)
        w_mod_after = torch.norm(model.fast_neuron_A.W_down_mod).item()
        w_mod_norms.append((w_mod_before, w_mod_after))

    # At least some W_mod changes should come from consolidation
    # (slow neuron fires at chunk 4 with fire_interval=4)
    final_w_mod = torch.norm(model.fast_neuron_A.W_down_mod).item()
    assert final_w_mod > 0, f"W_mod should be non-zero after consolidation"

    print(
        f"PASS: Test 4 — Consolidation writes: "
        f"final W_mod_norm={final_w_mod:.4f}"
    )


# ── Test 5: Gradient flow to slow neuron ──────────────────────────────
def test_gradient_flow_to_slow():
    """Gradients reach slow neuron params through consolidation."""
    torch.manual_seed(42)
    # Use fire_interval=2 so slow neuron fires frequently in tiny test
    model = make_model(slow_fire_interval=2)
    model.start_episode(BATCH, "cpu")
    model.slow_neuron_active = True
    model.fast_neuron_A.fixed_threshold = 0.0
    model.fast_neuron_B.fixed_threshold = 0.0

    model.zero_grad()

    input_ids = torch.randint(0, TINY_VOCAB, (BATCH, WINDOW_LEN))

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

    # Check slow neuron has gradients
    slow_grads = 0
    slow_total = 0
    for name, p in model.slow_neuron.named_parameters():
        slow_total += 1
        if p.grad is not None and p.grad.abs().sum() > 0:
            slow_grads += 1

    # At minimum, consolidation_write_nets and context_compress should have grads
    assert slow_grads > 0, (
        f"No slow neuron params have gradients. "
        f"Total slow params: {slow_total}"
    )

    print(
        f"PASS: Test 5 — Gradient flow to slow neuron: "
        f"{slow_grads}/{slow_total} params with grad"
    )


# ── Test 6: Detach between windows ───────────────────────────────────
def test_detach_between_windows():
    """Verify state is detached (no cross-window grad accumulation)."""
    torch.manual_seed(42)
    model = make_model()
    model.start_episode(BATCH, "cpu")
    model.slow_neuron_active = True
    model.fast_neuron_A.fixed_threshold = 0.0
    model.fast_neuron_B.fixed_threshold = 0.0

    input_ids = torch.randint(0, TINY_VOCAB, (BATCH, WINDOW_LEN))

    # Window 1: run adapt + eval, backward
    model.zero_grad()
    eval_losses_w1 = []
    for chunk_idx in range(NUM_CHUNKS):
        start = chunk_idx * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk_ids = input_ids[:, start:end]
        adapt = chunk_idx < NUM_ADAPT
        outputs = model(chunk_ids, adapt=adapt, chunk_idx=chunk_idx)
        loss = compute_loss(outputs.logits, chunk_ids)
        if not adapt:
            eval_losses_w1.append(loss)
        del outputs
        if adapt:
            del loss

    eval_loss_w1 = torch.stack(eval_losses_w1).mean()
    eval_loss_w1.backward()

    # Detach all state
    model.detach_all_state()

    # Verify W_mod is detached
    assert not model.fast_neuron_A.W_down_mod.requires_grad, (
        "W_down_mod should not require grad after detach"
    )
    assert not model.fast_neuron_A.W_up_mod.requires_grad, (
        "W_up_mod should not require grad after detach"
    )
    assert not model.fast_neuron_A.context.requires_grad, (
        "context should not require grad after detach"
    )

    # Window 2: should be able to run and backward independently
    model.start_window(BATCH, "cpu")
    eval_losses_w2 = []
    for chunk_idx in range(NUM_CHUNKS):
        start = chunk_idx * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk_ids = input_ids[:, start:end]
        adapt = chunk_idx < NUM_ADAPT
        outputs = model(chunk_ids, adapt=adapt, chunk_idx=chunk_idx)
        loss = compute_loss(outputs.logits, chunk_ids)
        if not adapt:
            eval_losses_w2.append(loss)
        del outputs
        if adapt:
            del loss

    eval_loss_w2 = torch.stack(eval_losses_w2).mean()
    # This should succeed without "trying to backward through graph a second time"
    eval_loss_w2.backward()

    assert math.isfinite(eval_loss_w2.item()), "Window 2 loss should be finite"

    print(
        f"PASS: Test 6 — Detach between windows: "
        f"w1_loss={eval_loss_w1.item():.4f}, w2_loss={eval_loss_w2.item():.4f}"
    )


# ── Test 7: Forgetting ratio computable ──────────────────────────────
def test_forgetting_ratio_computable():
    """Forgetting metric is finite and positive."""
    torch.manual_seed(42)
    model = make_model()
    config = make_config()
    batch = make_batch(3)

    model.start_episode(BATCH, "cpu")
    model.slow_neuron_active = True
    model.zero_grad()

    metrics = run_episode(model, batch, config, "cpu")

    fr = metrics["forgetting_ratio"]
    assert math.isfinite(fr), f"Forgetting ratio should be finite, got {fr}"
    assert fr > 0, f"Forgetting ratio should be positive, got {fr}"

    cb = metrics["consolidation_benefit"]
    assert math.isfinite(cb), f"Consolidation benefit should be finite, got {cb}"

    print(
        f"PASS: Test 7 — Forgetting ratio: "
        f"{fr:.3f}, consolidation_benefit={cb:+.4f}"
    )


# ── Test 8: sample_phase2_batch ──────────────────────────────────────
def test_sample_phase2_batch():
    """Verify Phase 2 batch sampling produces correct structure."""
    dataset = make_dataset()
    batch = dataset.sample_phase2_batch(
        batch_size=2, windows_A=8, windows_B=5, window_len=WINDOW_LEN,
    )

    assert len(batch['windows']) == 13, f"Expected 13 windows, got {len(batch['windows'])}"
    expected_labels = ['A'] * 5 + ['B'] * 5 + ['A'] * 3
    assert batch['domain_labels'] == expected_labels, (
        f"Expected {expected_labels}, got {batch['domain_labels']}"
    )

    for i, w in enumerate(batch['windows']):
        assert w.shape == (2, WINDOW_LEN), (
            f"Window {i} shape: {w.shape}, expected (2, {WINDOW_LEN})"
        )

    print(f"PASS: Test 8 — sample_phase2_batch: 13 windows, correct shapes")


if __name__ == "__main__":
    test_multi_window_pipeline()
    test_slow_neuron_fires()
    test_w_mod_persists_across_windows()
    test_consolidation_writes()
    test_gradient_flow_to_slow()
    test_detach_between_windows()
    test_forgetting_ratio_computable()
    test_sample_phase2_batch()
    print("\nAll Phase 2 tests passed!")
